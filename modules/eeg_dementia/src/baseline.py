"""Baseline classifiers with strict subject-level splits (no leakage).

Two setups:
  A) "subject":  average each person's windows into one feature vector -> 1 row per person.
  B) "window":   train on individual 4s windows, then average predicted probabilities
                 per person. Splits are by person (StratifiedGroupKFold), so no
                 person ever appears in both train and test.
Repeated 5-fold CV (10 repeats) over people. Reports accuracy, balanced accuracy, AUC.
"""
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TASKS = {"AD_vs_CN": ["A", "C"], "FTD_vs_CN": ["F", "C"], "AD_vs_FTD": ["A", "F"], "3class": ["A", "F", "C"]}


def models():
    return {
        "logreg": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)),
        "forest": lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=3, n_jobs=4, random_state=0),
    }


def evaluate(df, classes, mode, make_model, repeats=10):
    df = df[df.group.isin(classes)]
    feat = [c for c in df.columns if c not in ("subject", "group", "window")]
    if mode == "subject":
        df = df.groupby(["subject", "group"], as_index=False)[feat].mean()
    X = np.log(df[feat].to_numpy() + 1e-6)  # band ratios are skewed
    y = df.group.map({c: i for i, c in enumerate(classes)}).to_numpy()
    g = df.subject.to_numpy()
    scores = []
    for rep in range(repeats):
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=rep)
        subj_prob = {}
        for tr, te in cv.split(X, y, g):
            assert not set(g[tr]) & set(g[te])  # leakage guard
            m = make_model().fit(X[tr], y[tr])
            p = m.predict_proba(X[te])
            for s, pp in pd.DataFrame(p, index=g[te]).groupby(level=0).mean().iterrows():
                subj_prob[s] = pp.to_numpy()
        subj = pd.DataFrame({"s": g, "y": y}).drop_duplicates("s").set_index("s")
        P = np.stack([subj_prob[s] for s in subj.index]); yt = subj.y.to_numpy()
        pred = P.argmax(1)
        auc = roc_auc_score(yt, P[:, 1]) if len(classes) == 2 else roc_auc_score(yt, P, multi_class="ovr")
        scores.append((accuracy_score(yt, pred), balanced_accuracy_score(yt, pred), auc))
    s = np.array(scores)
    n = {c: int((subj.y == i).sum()) for i, c in enumerate(classes)}
    chance = max(n.values()) / sum(n.values())
    return {"n_people": n, "chance_acc": round(chance, 3),
            **{k: [round(s[:, i].mean(), 3), round(s[:, i].std(), 3)] for i, k in enumerate(["acc", "bal_acc", "auc"])}}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, default=Path("data/features.parquet"))
    ap.add_argument("--out", type=Path, default=Path("results/baseline.json"))
    a = ap.parse_args()
    df = pd.read_parquet(a.features)
    res = {}
    for task, classes in TASKS.items():
        for mode in ("subject", "window"):
            for name, mk in models().items():
                r = evaluate(df, classes, mode, mk, repeats=10 if mode == "subject" else 3)
                res[f"{task}|{mode}|{name}"] = r
                print(task, mode, name, r, flush=True)
    a.out.write_text(json.dumps(res, indent=2))
