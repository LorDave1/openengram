"""Extract spectral features from ds004504 preprocessed EEG.

Each recording is cut into 4-second windows (50% overlap). For every window and
each of the 19 channels we compute relative band power (delta..gamma) with
Welch's method, plus a few classic dementia markers (theta/alpha ratio,
spectral peak frequency). Output: one row per window, tagged with subject id.
"""
import argparse
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.signal import welch

BANDS = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 25), "gamma": (25, 45)}
WIN_S, STEP_S = 4.0, 2.0


def window_features(x, sfreq):
    f, p = welch(x, fs=sfreq, nperseg=int(2 * sfreq), axis=-1)  # x: (ch, time)
    sel = (f >= 0.5) & (f <= 45)
    total = p[:, sel].sum(axis=1)
    feats = {}
    for name, (lo, hi) in BANDS.items():
        m = (f >= lo) & (f < hi)
        feats[name] = p[:, m].sum(axis=1) / total
    feats["theta_alpha"] = feats["theta"] / feats["alpha"]
    feats["slow_fast"] = (feats["delta"] + feats["theta"]) / (feats["alpha"] + feats["beta"])
    m = (f >= 4) & (f <= 15)
    feats["peak_freq"] = f[m][p[:, m].argmax(axis=1)]
    return feats


def main(root, out):
    parts = pd.read_csv(root / "participants.tsv", sep="\t")
    rows = []
    for _, r in parts.iterrows():
        sid = r.participant_id
        path = root / "derivatives" / sid / "eeg" / f"{sid}_task-eyesclosed_eeg.set"
        raw = mne.io.read_raw_eeglab(path, preload=True, verbose="error")
        sf = raw.info["sfreq"]
        data = raw.get_data()
        ch = raw.ch_names
        win, step = int(WIN_S * sf), int(STEP_S * sf)
        for i, start in enumerate(range(0, data.shape[1] - win, step)):
            feats = window_features(data[:, start:start + win], sf)
            row = {"subject": sid, "group": r.Group, "window": i}
            for fname, vals in feats.items():
                for c, v in zip(ch, vals):
                    row[f"{fname}_{c}"] = v
            rows.append(row)
        print(sid, r.Group, f"{data.shape[1]/sf:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(out)
    print("saved", out, df.shape)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/ds004504"))
    ap.add_argument("--out", type=Path, default=Path("data/features.parquet"))
    a = ap.parse_args()
    main(a.root, a.out)
