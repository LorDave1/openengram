# Datasets

Data files are never committed. Each dataset has a download script in `scripts/`.

| Dataset | Used by | Size | License | Citation |
|---|---|---|---|---|
| OpenNeuro **ds004504** v1.0.9 — resting-state EEG, 88 people: 36 Alzheimer's (A), 23 frontotemporal dementia (F), 29 healthy (C). 19 channels, eyes closed. We use the authors' preprocessed `derivatives/`. | `eeg_dementia` | ~2.9 GB | **CC0** (public domain) | Miltiadous et al. (2023), *Data* 8(6):95, doi:10.3390/data8060095. Dataset doi:10.18112/openneuro.ds004504.v1.0.9 |
| OpenNeuro **ds006036** v1.1.0 — eyes-open EEG with photic stimulation (5–30 Hz) of the **same 88 people** as ds004504. We use the authors' preprocessed `derivatives/eeglab/`. | `eeg_dementia` (v3) | ~1 GB | **CC0** (public domain) | Same authors (Miltiadous et al.). Dataset doi:10.18112/openneuro.ds006036.v1.1.0 |

Even though CC0 doesn't require it, please cite the dataset authors in any write-up.

## Candidates for future modules (check licenses before use)

- **PrimeKG** (Harvard, drug/disease knowledge graph). Check license on Harvard Dataverse.
- **Hetionet** (CC0).
- **Human Connectome Project** structural connectomes: requires accepting HCP data use terms, so not redistributable here.
- **DementiaBank** (speech): requires membership; can't be redistributed.
