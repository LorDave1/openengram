"""Nested, subject-level evaluation of v2 features.

Outer loop: StratifiedGroupKFold over people (5 folds x 3 repeats) -> honest score.
Inner loop: 3-fold grouped CV on the *training people only* picks one of the
candidate models. The test people never influence the choice.
Training uses every 4th window (windows overlap 50%, so this loses little and is
4x faster); testing uses all windows, averaged into one probability per person.
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TASKS = {"AD_vs_CN": ["A", "C"], "FTD_vs_CN": ["F", "C"], "AD_vs_FTD": ["A", "F"], "3class": ["A", "F", "C"]}
SPECTRAL = ("rel_", "theta_alpha", "slow_fast", "peak_freq")
LOG_COLS = ("rel_", "theta_alpha", "slow_fast")


def base_models(spec_idx):
    return {
        "logreg_spectral": (spec_idx, lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))),
        "logreg_all": (None, lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=3000))),
        "boosting_all": (None, lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                                                     l2_regularization=1.0, random_state=0)),
    }


def fit_predict(X, y, g, w, tr, te, spec_idx):
    """Train every base model on people in tr, return per-person probabilities for people in te."""
    trs = tr[w[tr] % 4 == 0]
    probs = {}
    for name, (cols, mk) in base_models(spec_idx).items():
        Xa = X if cols is None else X[:, cols]
        p = mk().fit(Xa[trs], y[trs]).predict_proba(Xa[te])
        probs[name] = pd.DataFrame(p, index=g[te]).groupby(level=0).mean()
    probs["ensemble"] = (probs["logreg_all"] + probs["boosting_all"]) / 2
    return probs


def bal_acc(probs, ymap):
    return balanced_accuracy_score([ymap[s] for s in probs.index], probs.to_numpy().argmax(1))


def run_task(df, classes, feat, spec_idx, repeats):
    d = df[df.group.isin(classes)]
    X = d[feat].to_numpy(np.float32)
    y = d.group.map({c: i for i, c in enumerate(classes)}).to_numpy()
    g, w = d.subject.to_numpy(), d.window.to_numpy()
    ymap = dict(zip(g, y))
    per_model = {}; nested = []; picks = []
    for rep in range(repeats):
        outer = StratifiedGroupKFold(5, shuffle=True, random_state=rep)
        rep_probs = {}; rep_nested = []
        for tr, te in outer.split(X, y, g):
            assert not set(g[tr]) & set(g[te])
            # inner model selection on training people only
            inner = StratifiedGroupKFold(3, shuffle=True, random_state=100 + rep)
            inner_scores = {}
            for itr, ite in inner.split(X[tr], y[tr], g[tr]):
                for k, p in fit_predict(X, y, g, w, tr[itr], tr[ite], spec_idx).items():
                    inner_scores.setdefault(k, []).append(bal_acc(p, ymap))
            best = max(inner_scores, key=lambda k: np.mean(inner_scores[k]))
            picks.append(best)
            outer_probs = fit_predict(X, y, g, w, tr, te, spec_idx)
            for k, p in outer_probs.items():
                rep_probs.setdefault(k, []).append(p)
            rep_nested.append(outer_probs[best])
        for k, plist in rep_probs.items():
            per_model.setdefault(k, []).append(pd.concat(plist))
        nested.append(pd.concat(rep_nested))

    def summarize(plist):
        s = []
        for P in plist:
            yt = np.array([ymap[x] for x in P.index]); Pm = P.to_numpy()
            auc = roc_auc_score(yt, Pm[:, 1]) if len(classes) == 2 else roc_auc_score(yt, Pm, multi_class="ovr")
            s.append((accuracy_score(yt, Pm.argmax(1)), balanced_accuracy_score(yt, Pm.argmax(1)), auc))
        s = np.array(s)
        return {k: [round(float(s[:, i].mean()), 3), round(float(s[:, i].std()), 3)] for i, k in enumerate(["acc", "bal_acc", "auc"])}

    n = {c: int(sum(1 for v in ymap.values() if v == i)) for i, c in enumerate(classes)}
    return {"n_people": n, "chance_acc": round(max(n.values()) / sum(n.values()), 3),
            "nested": summarize(nested), "picks": {k: picks.count(k) for k in set(picks)},
            "per_model_outer_only": {k: summarize(v) for k, v in per_model.items()}}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, default=Path("data/features_v2.parquet"))
    ap.add_argument("--out", type=Path, default=Path("results/v2_nested.json"))
    ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args()
    df = pd.read_parquet(a.features)
    feat = [c for c in df.columns if c not in ("subject", "group", "window")]
    for c in feat:
        if c.startswith(LOG_COLS):
            df[c] = np.log(df[c] + 1e-6)
    spec_idx = [i for i, c in enumerate(feat) if c.startswith(SPECTRAL)]
    res = {}
    for task, classes in TASKS.items():
        res[task] = run_task(df, classes, feat, spec_idx, a.repeats)
        print(task, json.dumps(res[task]), flush=True)
        a.out.write_text(json.dumps(res, indent=2))
