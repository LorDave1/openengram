"""v3 feature sets, built to target AD vs FTD.

1) Region features: the 19 channels are averaged into 5 regions (frontal, central,
   temporal, parietal, occipital). FTD mainly affects frontal/temporal areas, AD more
   posterior ones, so we add anterior-minus-posterior contrasts. Connectivity is
   summarised as mean coherence / wPLI within and between regions. ~150 features
   instead of 1,007, which suits a dataset of 88 people.
2) Riemannian features: per 4 s raw window and frequency band, the 19x19 covariance
   matrix (shrinkage, OAS). The tangent-space projection is fitted later inside each
   training fold only (see evaluate_v3.py).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from pyriemann.estimation import Covariances
from scipy.signal import butter, sosfiltfilt

REGIONS = {
    "frontal": ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz"],
    "central": ["C3", "C4", "Cz"],
    "temporal": ["T3", "T4", "T5", "T6"],
    "parietal": ["P3", "P4", "Pz"],
    "occipital": ["O1", "O2"],
}
CH2REG = {c: r for r, cs in REGIONS.items() for c in cs}
PER_CH = ["abs_delta", "abs_theta", "abs_alpha", "abs_beta", "abs_gamma", "rel_delta", "rel_theta", "rel_alpha",
          "rel_beta", "rel_gamma", "theta_alpha", "slow_fast", "peak_freq", "spec_entropy", "aperiodic_slope",
          "hjorth_mob", "hjorth_comp"]
LOGGED = ("rel_", "theta_alpha", "slow_fast")
RIEMANN_BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 25)}


def region_features(v2: pd.DataFrame) -> pd.DataFrame:
    out = {"subject": v2.subject, "group": v2.group, "window": v2.window}
    for f in PER_CH:
        reg = {}
        for r, cs in REGIONS.items():
            x = v2[[f"{f}_{c}" for c in cs]].to_numpy()
            if f.startswith(LOGGED):
                x = np.log(x + 1e-6)
            reg[r] = x.mean(1)
            out[f"{f}__{r}"] = reg[r]
        out[f"{f}__ant_minus_post"] = reg["frontal"] - (reg["parietal"] + reg["occipital"]) / 2
        out[f"{f}__temp_minus_post"] = reg["temporal"] - (reg["parietal"] + reg["occipital"]) / 2
        out[f"{f}__left_minus_right"] = (v2[[f"{f}_{c}" for c in ["Fp1", "F3", "F7", "C3", "T3", "T5", "P3", "O1"]]].mean(1)
                                         - v2[[f"{f}_{c}" for c in ["Fp2", "F4", "F8", "C4", "T4", "T6", "P4", "O2"]]].mean(1)).to_numpy()
    for kind in ["coh_theta", "coh_alpha", "wpli_theta", "wpli_alpha"]:
        cols = [c for c in v2.columns if c.startswith(kind + "_")]
        groups = {}
        for c in cols:
            a, b = c[len(kind) + 1:].split("-")
            key = "__".join(sorted([CH2REG[a], CH2REG[b]]))
            groups.setdefault(key, []).append(c)
        for key, cs in groups.items():
            out[f"{kind}__{key}"] = v2[cs].to_numpy().mean(1)
    return pd.DataFrame(out)


def riemann_covariances(cache: Path):
    z = np.load(cache, allow_pickle=True)
    X, subj, group = z["X"], z["subj"], z["group"]  # (n, 19, 500) at 125 Hz
    covs = []
    for lo, hi in RIEMANN_BANDS.values():
        sos = butter(4, [lo, hi], btype="band", fs=125, output="sos")
        Xf = sosfiltfilt(sos, X, axis=-1).astype(np.float64)
        covs.append(Covariances("oas").fit_transform(Xf))
    return np.stack(covs, 1), subj, group  # (n, bands, 19, 19)


if __name__ == "__main__":
    v2 = pd.read_parquet("data/features_v2.parquet")
    r = region_features(v2)
    r.to_parquet("data/features_v3_region.parquet")
    print("region", r.shape, flush=True)
    C, s, g = riemann_covariances(Path("data/raw125.npz"))
    np.savez("data/features_v3_riemann.npz", C=C, subj=s, group=g)
    print("riemann", C.shape)
