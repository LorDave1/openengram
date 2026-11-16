"""Richer EEG features (v2): spectral + complexity + connectivity, per 4 s window.

Per channel: log absolute band power, relative band power, theta/alpha and
slow/fast ratios, alpha peak frequency, spectral entropy, aperiodic (1/f) slope,
Hjorth mobility and complexity.
Per channel pair (171 pairs) in theta and alpha bands: magnitude-squared
coherence and weighted phase lag index (wPLI, robust to volume conduction).
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mne
import numpy as np
import pandas as pd

BANDS = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 25), "gamma": (25, 45)}
CONN_BANDS = {"theta": (4, 8), "alpha": (8, 13)}
WIN_S, STEP_S, SEG_S = 4.0, 2.0, 1.0


def cross_spectra(x, sf):
    """x: (ch, t). 1 s Hann segments, 50% overlap -> complex spectra (seg, ch, f)."""
    n = int(SEG_S * sf)
    starts = range(0, x.shape[1] - n + 1, n // 2)
    segs = np.stack([x[:, s:s + n] for s in starts]) * np.hanning(n)
    return np.fft.rfftfreq(n, 1 / sf), np.fft.rfft(segs, axis=-1)


def window_features(x, sf, iu):
    f, X = cross_spectra(x, sf)
    psd = (np.abs(X) ** 2).mean(0)  # (ch, f)
    sel = (f >= 0.5) & (f <= 45)
    total = psd[:, sel].sum(1)
    out = {}
    for b, (lo, hi) in BANDS.items():
        m = (f >= lo) & (f < hi)
        bp = psd[:, m].sum(1)
        out[f"abs_{b}"] = np.log(bp + 1e-30)
        out[f"rel_{b}"] = bp / total
    out["theta_alpha"] = out["rel_theta"] / out["rel_alpha"]
    out["slow_fast"] = (out["rel_delta"] + out["rel_theta"]) / (out["rel_alpha"] + out["rel_beta"])
    m = (f >= 4) & (f <= 15)
    out["peak_freq"] = f[m][psd[:, m].argmax(1)]
    pn = psd[:, sel] / total[:, None]
    out["spec_entropy"] = -(pn * np.log(pn + 1e-30)).sum(1) / np.log(sel.sum())
    m = ((f >= 2) & (f < 7)) | ((f > 14) & (f <= 40))  # fit outside theta/alpha peaks
    lf, lp = np.log10(f[m]), np.log10(psd[:, m] + 1e-30)
    out["aperiodic_slope"] = np.polyfit(lf, lp.T, 1)[0]
    d1, d2 = np.diff(x, axis=1), np.diff(x, n=2, axis=1)
    v0, v1, v2 = x.var(1), d1.var(1), d2.var(1)
    out["hjorth_mob"] = np.sqrt(v1 / v0)
    out["hjorth_comp"] = np.sqrt(v2 / v1) / out["hjorth_mob"]
    conn = {}
    for b, (lo, hi) in CONN_BANDS.items():
        m = (f >= lo) & (f < hi)
        Xb = X[:, :, m]  # (seg, ch, f)
        S = np.einsum("sif,sjf->sijf", Xb, Xb.conj())  # cross-spectra per segment
        Sm = S.mean(0)
        P = np.real(np.einsum("iif->if", Sm))
        coh = np.abs(Sm) ** 2 / (P[:, None, :] * P[None, :, :])
        im = np.imag(S)
        wpli = np.abs(im.mean(0)) / (np.abs(im).mean(0) + 1e-30)
        conn[f"coh_{b}"] = coh.mean(-1)[iu]
        conn[f"wpli_{b}"] = wpli.mean(-1)[iu]
    return out, conn


def process(args):
    sid, group, path = args
    raw = mne.io.read_raw_eeglab(path, preload=True, verbose="error")
    sf, data, ch = raw.info["sfreq"], raw.get_data(), raw.ch_names
    iu = np.triu_indices(len(ch), 1)
    pairs = [f"{ch[i]}-{ch[j]}" for i, j in zip(*iu)]
    win, step = int(WIN_S * sf), int(STEP_S * sf)
    rows = []
    for w, s in enumerate(range(0, data.shape[1] - win, step)):
        per_ch, conn = window_features(data[:, s:s + win], sf, iu)
        row = {"subject": sid, "group": group, "window": w}
        for k, v in per_ch.items():
            row.update({f"{k}_{c}": val for c, val in zip(ch, v)})
        for k, v in conn.items():
            row.update({f"{k}_{p}": val for p, val in zip(pairs, v)})
        rows.append(row)
    return pd.DataFrame(rows).astype({"window": "int32"})


def main(root, out, workers):
    parts = pd.read_csv(root / "participants.tsv", sep="\t")
    jobs = [(r.participant_id, r.Group, root / "derivatives" / r.participant_id / "eeg" /
             f"{r.participant_id}_task-eyesclosed_eeg.set") for r in parts.itertuples()]
    with ProcessPoolExecutor(workers) as ex:
        dfs = []
        for d in ex.map(process, jobs):
            dfs.append(d)
            print(d.subject.iloc[0], len(d), flush=True)
    df = pd.concat(dfs, ignore_index=True)
    num = df.columns.difference(["subject", "group", "window"])
    df[num] = df[num].astype("float32")
    df.to_parquet(out)
    print("saved", out, df.shape)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/ds004504"))
    ap.add_argument("--out", type=Path, default=Path("data/features_v2.parquet"))
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    main(a.root, a.out, a.workers)
