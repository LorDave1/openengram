# eeg_dementia

**Question:** can a routine 15-minute resting EEG (eyes closed) separate Alzheimer's disease (AD)
and frontotemporal dementia (FTD) from healthy aging?

**Data:** OpenNeuro ds004504 (CC0), 88 people: 36 AD, 23 FTD, 29 healthy controls (CN).
We use the authors' preprocessed recordings. See [DATA.md](../../DATA.md).

## Method (baseline)

1. `src/features.py`: cut each recording into 4 s windows (2 s step), 34,788 windows total.
   Per window and per channel (19): relative power in delta/theta/alpha/beta/gamma, theta/alpha ratio,
   slow/fast ratio, alpha peak frequency. 152 features.
2. `src/baseline.py`: logistic regression and random forest.
   - *subject*: average each person's windows into one row.
   - *window*: classify windows, then average the probabilities per person.
   - Splits are **by person** (`StratifiedGroupKFold`, 5 folds, repeated with different seeds;
     a code assertion checks no person is in both train and test). All scores are per person.

## Results (accuracy per person, mean ± std over repeats)

| Task | Chance | Best model | Accuracy | Balanced acc. | AUC |
|---|---|---|---|---|---|
| AD vs healthy | 55% | window + forest | **83.6% ± 0.7** | 83.5% | 0.89 |
| AD vs healthy | 55% | window + logreg | 79.5% ± 1.5 | 79.1% | **0.91** |
| FTD vs healthy | 56% | window + forest | **75.0%** | 72.6% | 0.83 |
| AD vs FTD | 61% | window + logreg | 68.9% ± 0.8 | 62.8% | 0.63 |
| 3 groups | 41% | window + forest | **64.4% ± 0.5** | 59.4% | 0.71 |

Full numbers for all 16 combinations: `results/baseline.json` (regenerate with `make eeg-baseline`).

**In plain words:** from brain-wave rhythms alone, the model correctly tells an Alzheimer's
patient from a healthy person about 5 times out of 6, where guessing gets about half right.
The classic signal is that slow rhythms (theta) rise and alpha falls in dementia.
FTD vs healthy is harder (3 out of 4). Telling AD and FTD *apart* barely beats guessing,
which is the hardest and most clinically interesting open problem here.

## Limitations

- Only 88 people from one hospital; numbers will vary on other populations and devices.
- We compared 4 setups and report the best per task, which is mildly optimistic.
  Proper next step: nested cross-validation.
- Research only, see [SAFETY.md](../../SAFETY.md).

## Good first contributions

- Connectivity features (coherence, phase-lag index) and spectral entropy.
- A small CNN/transformer on raw windows (PyTorch, `PYTORCH_ENABLE_MPS_FALLBACK=1` on Apple silicon).
- Nested CV and confidence intervals; age/sex as covariates (patients are older on average).
