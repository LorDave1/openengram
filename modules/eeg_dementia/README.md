# eeg_dementia

**Question:** can a routine 15-minute resting EEG (eyes closed) separate Alzheimer's disease (AD)
and frontotemporal dementia (FTD) from healthy aging?

**Data:** OpenNeuro ds004504 (CC0), 88 people: 36 AD, 23 FTD, 29 healthy controls (CN).
We use the authors' preprocessed recordings. See [DATA.md](../../DATA.md).

## Headline results (v2, nested cross-validation)

Accuracy per **person**, mean ± std over repeats. Every split is by person; model choice is made
inside the training people only (nested CV), so these numbers are not tuned on the test people.

| Task | Chance | Age + sex only | **EEG v2 (nested)** | Balanced acc. | AUC |
|---|---|---|---|---|---|
| AD vs healthy | 55% | 59% | **87.2% ± 3.2** | 87.6% | 0.91 |
| FTD vs healthy | 56% | 58% | **76.3% ± 1.8** | 74.2% | 0.83 |
| AD vs FTD | 61% | 58% | 58.2% ± 0.8 | 51.9% | 0.60 |
| 3 groups | 41% | n/a | **63.3% ± 2.7** | 59.2% | 0.78 |

**In plain words:** from brain-wave patterns alone, the model correctly tells an Alzheimer's
patient from a healthy person in about 7 out of 8 people, where guessing gets about half right.
Age and sex alone only reach ~59%, so the EEG is doing the work. FTD vs healthy is about 3 in 4.
Telling AD and FTD *apart* is still no better than chance with these features: that is the main
open problem in this module.

## Methods

### v1 baseline (`features.py`, `baseline.py`)
4 s windows (2 s step, 34,788 total). Per channel (19): relative band power (delta–gamma),
theta/alpha and slow/fast ratios, alpha peak frequency (152 features). Logistic regression and
random forest, repeated 5-fold `StratifiedGroupKFold` by person.

### v2 features (`features_v2.py`), 1,007 per window
- Per channel: log absolute and relative band power, ratios, peak frequency, **spectral entropy**,
  **aperiodic (1/f) slope**, **Hjorth mobility and complexity**.
- Per channel pair (171 pairs), theta and alpha bands: **coherence** and **weighted phase lag index**
  (wPLI, robust to volume conduction).

### v2 evaluation (`evaluate_v2.py`)
Outer: 5-fold × 3 repeats by person. Inner: 3-fold by person on the training people picks one of
logistic regression (spectral only), logistic regression (all), gradient boosting (all), or their
ensemble. Training uses every 4th window (overlapping windows add little). Test windows are
averaged into one probability per person.

### Small CNN on raw EEG (`cnn.py`)
EEGNet-style network on raw 4 s windows at 125 Hz, trained on Apple-silicon GPU (MPS), fixed
20 epochs, 5-fold × 2 repeats by person.

## All results

| Task | v1 best (not nested) | v2 nested | v2 per model (outer only, for comparison) | CNN raw EEG |
|---|---|---|---|---|
| AD vs healthy | 83.6% | **87.2%** | logreg 89.7%, boosting 89.2%, ensemble 86.7% | 85.4% ± 2.3 (AUC 0.92) |
| FTD vs healthy | 75.0% | **76.3%** | ensemble 78.2%, logreg 74.4%, boosting 74.4% | 69.2% ± 3.8 |
| AD vs FTD | 68.9% (bal. 63%) | 58.2% (bal. 52%) | spectral logreg 66.7% (bal. 60%) | 65.3% (bal. 63%) |
| 3 groups | 64.4% | **63.3%** | ensemble 66.7%, boosting 65.9% | 56.2% ± 6.2 |

The "per model" column picks the best model after seeing the test people, so it is optimistic;
the nested column is the honest number. Raw files: `results/baseline.json`, `results/v2_nested.json`,
`results/cnn.json`.

What we learned:
- Connectivity and complexity features help AD vs healthy (+3.6 points, honestly measured).
- The CNN is close but not better than hand-made features on 88 people. More data or pre-training would likely be needed.
- For AD vs FTD, richer features made things worse (more features, same few people). Simpler,
  targeted features or more data are needed.

## Reproduce

```bash
make eeg-features-v2   # ~5 min
make eeg-nested        # ~45 min, low priority
make eeg-cnn           # ~30 min on Apple-silicon GPU
```

## Limitations

- Only 88 people from one hospital, so results may not carry over to other populations or devices.
- The groups differ in sex balance (AD 33% male vs healthy 62%); age and sex alone reach ~59%.
  The EEG result is far above that, but adjusting for these confounds properly is still to do.
- Research only, see [SAFETY.md](../../SAFETY.md).

## Good next contributions

- Confidence intervals by bootstrap over people.
- Features aimed at AD vs FTD (frontal vs posterior asymmetry, frontal theta).
- Pre-trained EEG foundation models, or adding other open dementia EEG datasets for more people.
