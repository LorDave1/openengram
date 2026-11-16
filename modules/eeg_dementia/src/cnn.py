"""Small convolutional network (EEGNet-style) on raw EEG windows, subject-level CV.

Recordings are resampled to 125 Hz, z-scored per channel per person, and cut into
4 s windows (19 x 500). Each fold trains on people in the training split only;
predictions are averaged per person. Runs on Apple-silicon GPU (MPS) when available.
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import mne
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

SF, WIN = 125, 500
TASKS = {"AD_vs_CN": ["A", "C"], "FTD_vs_CN": ["F", "C"], "AD_vs_FTD": ["A", "F"], "3class": ["A", "F", "C"]}


def load(root, cache):
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return z["X"], z["subj"], z["group"]
    parts = pd.read_csv(root / "participants.tsv", sep="\t")
    Xs, ss, gs = [], [], []
    for r in parts.itertuples():
        p = root / "derivatives" / r.participant_id / "eeg" / f"{r.participant_id}_task-eyesclosed_eeg.set"
        raw = mne.io.read_raw_eeglab(p, preload=True, verbose="error").resample(SF, verbose="error")
        x = raw.get_data()
        x = (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-12)
        n = x.shape[1] // WIN
        Xs.append(x[:, : n * WIN].reshape(19, n, WIN).transpose(1, 0, 2).astype(np.float32))
        ss += [r.participant_id] * n; gs += [r.Group] * n
    X, subj, group = np.concatenate(Xs), np.array(ss), np.array(gs)
    np.savez(cache, X=X, subj=subj, group=group)
    return X, subj, group


class EEGNet(nn.Module):
    def __init__(self, n_classes, ch=19, F1=8, D=2, F2=16, drop=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, F1, (1, 63), padding=(0, 31), bias=False), nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (ch, 1), groups=F1, bias=False), nn.BatchNorm2d(F1 * D), nn.ELU(),
            nn.AvgPool2d((1, 4)), nn.Dropout(drop),
            nn.Conv2d(F1 * D, F1 * D, (1, 15), padding=(0, 7), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, 1, bias=False), nn.BatchNorm2d(F2), nn.ELU(),
            nn.AvgPool2d((1, 8)), nn.Dropout(drop), nn.Flatten())
        self.fc = nn.Linear(F2 * (WIN // 32), n_classes)

    def forward(self, x):
        return self.fc(self.net(x.unsqueeze(1)))


def train_fold(Xtr, ytr, Xte, n_classes, dev, epochs, seed):
    torch.manual_seed(seed)
    m = EEGNet(n_classes).to(dev)
    opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=1e-2)
    w = torch.tensor(len(ytr) / (n_classes * np.bincount(ytr, minlength=n_classes)), dtype=torch.float32, device=dev)
    lossf = nn.CrossEntropyLoss(weight=w)
    Xt, yt = torch.from_numpy(Xtr), torch.from_numpy(ytr)
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(perm), 128):
            b = perm[i:i + 128]
            xb = Xt[b].to(dev)
            xb = xb * (1 + 0.1 * torch.randn(len(b), 1, 1, device=dev))  # amplitude jitter
            loss = lossf(m(xb), yt[b].to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
    m.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(Xte), 512):
            out.append(torch.softmax(m(torch.from_numpy(Xte[i:i + 512]).to(dev)), 1).cpu().numpy())
    return np.concatenate(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/ds004504"))
    ap.add_argument("--cache", type=Path, default=Path("data/raw125.npz"))
    ap.add_argument("--out", type=Path, default=Path("results/cnn.json"))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(4)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    X, subj, group = load(a.root, a.cache)
    print("data", X.shape, dev, flush=True)
    res = {}
    for task, classes in TASKS.items():
        sel = np.isin(group, classes)
        Xk, g = X[sel], subj[sel]
        y = np.array([classes.index(c) for c in group[sel]])
        ymap = dict(zip(g, y)); scores = []
        for rep in range(a.repeats):
            probs = []
            for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=rep).split(Xk, y, g):
                assert not set(g[tr]) & set(g[te])
                p = train_fold(Xk[tr], y[tr], Xk[te], len(classes), dev, a.epochs, rep)
                probs.append(pd.DataFrame(p, index=g[te]).groupby(level=0).mean())
            P = pd.concat(probs); yt = np.array([ymap[s] for s in P.index]); Pm = P.to_numpy()
            auc = roc_auc_score(yt, Pm[:, 1]) if len(classes) == 2 else roc_auc_score(yt, Pm, multi_class="ovr")
            scores.append((accuracy_score(yt, Pm.argmax(1)), balanced_accuracy_score(yt, Pm.argmax(1)), auc))
        s = np.array(scores)
        res[task] = {k: [round(float(s[:, i].mean()), 3), round(float(s[:, i].std()), 3)] for i, k in enumerate(["acc", "bal_acc", "auc"])}
        print(task, res[task], flush=True)
        a.out.write_text(json.dumps(res, indent=2))
