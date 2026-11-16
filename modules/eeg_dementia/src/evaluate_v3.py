"""Nested, subject-level evaluation over all candidate models (v2 + v3).

Folds are made over *people*. Every candidate trains only on the training people of
a fold (including feature scaling, feature selection and the Riemannian tangent-space
reference) and outputs one probability per test person. The inner loop picks a
candidate on training people only, so the outer score is honest.

Usage: python evaluate_v3.py --mode screen   (quick, not nested: for exploring)
       python evaluate_v3.py --mode nested   (honest numbers for the README)
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
import numpy as np
import pandas as pd
from pyriemann.tangentspace import TangentSpace
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TASKS = {"AD_vs_CN": ["A", "C"], "FTD_vs_CN": ["F", "C"], "AD_vs_FTD": ["A", "F"], "3class": ["A", "F", "C"]}
LOG_COLS = ("rel_", "theta_alpha", "slow_fast")
SPECTRAL = ("rel_", "theta_alpha", "slow_fast", "peak_freq")


class Data:
    def __init__(self):
        v2 = pd.read_parquet("data/features_v2.parquet")
        self.v2_cols = [c for c in v2.columns if c not in ("subject", "group", "window")]
        for c in self.v2_cols:
            if c.startswith(LOG_COLS):
                v2[c] = np.log(v2[c] + 1e-6)
        self.v2 = v2
        self.v2_spec = [c for c in self.v2_cols if c.startswith(SPECTRAL)]
        reg = pd.read_parquet("data/features_v3_region.parquet")
        self.reg = reg
        self.reg_cols = [c for c in reg.columns if c not in ("subject", "group", "window")]
        ph = Path("data/features_v2_photic.parquet")  # same people, eyes open + photic stimulation (ds006036)
        if ph.exists():
            eo = pd.read_parquet(ph)
            for c in self.v2_cols:
                if c.startswith(LOG_COLS):
                    eo[c] = np.log(eo[c] + 1e-6)
            ok = np.isfinite(eo[self.v2_cols].to_numpy()).all(1)  # drop flat/zeroed windows (0.5%)
            self.eo = eo[ok].reset_index(drop=True)
        z = np.load("data/features_v3_riemann.npz", allow_pickle=True)
        self.C, self.C_subj = z["C"], z["subj"]


def logreg(C):
    return lambda: make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=4000))


def boost():
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                          l2_regularization=1.0, random_state=0)


def per_person(p, subj):
    return pd.DataFrame(p, index=subj).groupby(level=0).mean()


def tab_model(df, cols, make, every=4):
    def run(train_s, test_s, ymap):
        tr = df[df.subject.isin(train_s) & (df.window % every == 0)]
        te = df[df.subject.isin(test_s)]
        m = make().fit(tr[cols].to_numpy(np.float32), tr.subject.map(ymap).to_numpy())
        return per_person(m.predict_proba(te[cols].to_numpy(np.float32)), te.subject.to_numpy())
    return run


def riemann_model(D, C=1.0, bands=(0, 1, 2, 3), every=2):
    def run(train_s, test_s, ymap):
        idx = np.arange(len(D.C_subj))
        tr = idx[np.isin(D.C_subj, train_s) & (idx % every == 0)]
        te = idx[np.isin(D.C_subj, test_s)]
        Ftr, Fte = [], []
        for b in bands:
            ts = TangentSpace(metric="riemann").fit(D.C[tr, b])  # reference from training people only
            Ftr.append(ts.transform(D.C[tr, b])); Fte.append(ts.transform(D.C[te, b]))
        m = logreg(C)().fit(np.hstack(Ftr), pd.Series(D.C_subj[tr]).map(ymap).to_numpy())
        return per_person(m.predict_proba(np.hstack(Fte)), D.C_subj[te])
    return run


def subj_model(df, cols, make, stats=("median",)):
    """One row per person: median (and optionally IQR) of each feature over that person's windows."""
    def run(train_s, test_s, ymap):
        d = df[df.subject.isin(np.concatenate([train_s, test_s]))]
        g = d.groupby("subject")[cols]
        parts = [g.median()]
        if "iqr" in stats:
            parts.append(g.quantile(0.75) - g.quantile(0.25))
        T = pd.concat(parts, axis=1)
        tr, te = T.loc[train_s], T.loc[test_s]
        m = make().fit(tr.to_numpy(), tr.index.map(ymap).to_numpy())
        return pd.DataFrame(m.predict_proba(te.to_numpy()), index=te.index)
    return run


def logreg_bal(C, penalty="l2"):
    # sklearn >= 1.8: choose L1 vs L2 with l1_ratio (the old `penalty` argument is deprecated)
    if penalty == "l1":
        return lambda: make_pipeline(StandardScaler(), LogisticRegression(C=C, l1_ratio=1.0, solver="saga",
                                                                          class_weight="balanced", max_iter=8000))
    return lambda: make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=8000))


def concat_subj_model(dfs, cols, make):
    """Per person: medians from the eyes-closed AND the eyes-open recording, side by side."""
    def run(train_s, test_s, ymap):
        T = pd.concat([d.groupby("subject")[cols].median().add_suffix(f"__{i}") for i, d in enumerate(dfs)], axis=1)
        tr, te = T.loc[train_s], T.loc[test_s]
        m = make().fit(tr.to_numpy(), tr.index.map(ymap).to_numpy())
        return pd.DataFrame(m.predict_proba(te.to_numpy()), index=te.index)
    return run


def candidates(D, full=False, photic=False):
    if photic:
        c = candidates(D)
        c.update({
            "eo_logreg_all": tab_model(D.eo, D.v2_cols, logreg(0.01), every=2),
            "eo_boosting_all": tab_model(D.eo, D.v2_cols, boost, every=2),
            "eo_subj_logreg_bal": subj_model(D.eo, D.v2_cols, logreg_bal(0.01)),
            "subj_ec_eo_concat_bal": concat_subj_model([D.v2, D.eo], D.v2_cols, logreg_bal(0.005)),
        })
        return c
    if not full:  # the defended candidate set used for the README numbers
        return {
            "v2_logreg_spectral": tab_model(D.v2, D.v2_spec, logreg(0.1)),
            "v2_logreg_all": tab_model(D.v2, D.v2_cols, logreg(0.01)),
            "v2_logreg_all_bal": tab_model(D.v2, D.v2_cols, logreg_bal(0.01)),
            "v2_boosting_all": tab_model(D.v2, D.v2_cols, boost),
            "subj_v2_logreg_bal": subj_model(D.v2, D.v2_cols, logreg_bal(0.01)),
            "subj_v2_l1_bal": subj_model(D.v2, D.v2_cols, logreg_bal(0.3, "l1"), ("median", "iqr")),
        }
    c = {
        "v2_logreg_spectral": tab_model(D.v2, D.v2_spec, logreg(0.1)),
        "v2_logreg_all": tab_model(D.v2, D.v2_cols, logreg(0.01)),
        "v2_boosting_all": tab_model(D.v2, D.v2_cols, boost),
        "v2_select50_logreg": tab_model(D.v2, D.v2_cols, lambda: make_pipeline(
            StandardScaler(), SelectKBest(f_classif, k=50), LogisticRegression(C=0.1, max_iter=4000))),
        "region_logreg": tab_model(D.reg, D.reg_cols, logreg(0.05)),
        "region_boosting": tab_model(D.reg, D.reg_cols, boost),
        "riemann_logreg": riemann_model(D, C=0.05),
    }
    return c


ENSEMBLES_FULL = {"ens_region_riemann": ["region_logreg", "riemann_logreg"],
                  "ens_v2_riemann": ["v2_logreg_all", "v2_boosting_all", "riemann_logreg"],
                  "ens_all": ["v2_logreg_all", "v2_boosting_all", "region_logreg", "region_boosting", "riemann_logreg"]}
ENSEMBLES_MAIN = {"ens_window": ["v2_logreg_all", "v2_boosting_all"],
                  "ens_subj_window": ["subj_v2_logreg_bal", "v2_boosting_all"]}
ENSEMBLES_PHOTIC = {**ENSEMBLES_MAIN,
                    "ens_ec_eo_logreg": ["v2_logreg_all", "eo_logreg_all"],
                    "ens_ec_eo_boosting": ["v2_boosting_all", "eo_boosting_all"],
                    "ens_ec_eo_subj": ["subj_v2_logreg_bal", "eo_subj_logreg_bal"],
                    "ens_ec_eo_window": ["v2_logreg_all", "v2_boosting_all", "eo_logreg_all", "eo_boosting_all"]}
ENSEMBLES = ENSEMBLES_MAIN


def predict_all(cands, train_s, test_s, ymap):
    out = {k: f(train_s, test_s, ymap) for k, f in cands.items()}
    for k, parts in ENSEMBLES.items():
        out[k] = sum(out[p] for p in parts) / len(parts)
    return out


def score(P, ymap, k):
    yt = np.array([ymap[s] for s in P.index]); Pm = P.to_numpy()
    auc = roc_auc_score(yt, Pm[:, 1]) if k == 2 else roc_auc_score(yt, Pm, multi_class="ovr")
    return accuracy_score(yt, Pm.argmax(1)), balanced_accuracy_score(yt, Pm.argmax(1)), auc


def summarize(scores):
    s = np.array(scores)
    return {k: [round(float(s[:, i].mean()), 3), round(float(s[:, i].std()), 3)] for i, k in enumerate(["acc", "bal_acc", "auc"])}


def run_task(D, cands, classes, mode, repeats):
    people = D.v2[D.v2.group.isin(classes)].drop_duplicates("subject")
    S = people.subject.to_numpy(); y = people.group.map({c: i for i, c in enumerate(classes)}).to_numpy()
    ymap = dict(zip(S, y)); k = len(classes)
    per_model, nested, picks = {}, [], []
    for rep in range(repeats):
        rep_outer, rep_nested = {}, []
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=rep).split(S, y):
            outer = predict_all(cands, S[tr], S[te], ymap)
            for m, p in outer.items():
                rep_outer.setdefault(m, []).append(p)
            if mode == "nested":
                inner_scores = {}
                for itr, ite in StratifiedKFold(3, shuffle=True, random_state=100 + rep).split(S[tr], y[tr]):
                    for m, p in predict_all(cands, S[tr][itr], S[tr][ite], ymap).items():
                        inner_scores.setdefault(m, []).append(score(p, ymap, k)[1])
                best = max(inner_scores, key=lambda m: np.mean(inner_scores[m]))
                picks.append(best); rep_nested.append(outer[best])
        for m, plist in rep_outer.items():
            per_model.setdefault(m, []).append(score(pd.concat(plist), ymap, k))
        if mode == "nested":
            nested.append(score(pd.concat(rep_nested), ymap, k))
    n = {c: int((y == i).sum()) for i, c in enumerate(classes)}
    res = {"n_people": n, "chance_acc": round(max(n.values()) / len(y), 3),
           "per_model_outer_only": {m: summarize(v) for m, v in per_model.items()}}
    if mode == "nested":
        res["nested"] = summarize(nested); res["picks"] = {m: picks.count(m) for m in set(picks)}
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["screen", "nested"], default="nested")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--photic", action="store_true", help="also use the eyes-open photic recordings (ds006036)")
    ap.add_argument("--full", action="store_true", help="exploratory set incl. region and Riemannian models")
    a = ap.parse_args()
    out = a.out or Path(f"results/v3_{a.mode}{'_photic' if a.photic else ''}.json")
    if a.full:
        ENSEMBLES = ENSEMBLES_FULL
    if a.photic:
        ENSEMBLES = ENSEMBLES_PHOTIC
    D = Data(); cands = candidates(D, a.full, a.photic); res = {}
    for t in a.tasks.split(","):
        res[t] = run_task(D, cands, TASKS[t], a.mode, a.repeats)
        print(t, json.dumps(res[t]), flush=True)
        out.write_text(json.dumps(res, indent=2))
