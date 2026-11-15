# OpenEngram

**An open, community-built research workspace for understanding and detecting dementia.**

*An engram is the physical trace a memory leaves in the brain. OpenEngram is an open effort to understand how those traces are lost, and how to protect them.*

Founded by David Abramov. Anyone can contribute: code, data analysis, domain knowledge,
literature reviews, or plain-language explanations. You don't need to be a scientist
to help, and scientists are very welcome too.

🌐 **Website:** https://lordave1.github.io/openengram/ (English + עברית)

> ⚠️ **Research only. Not a medical device.** Nothing here diagnoses, treats or gives
> medical advice to anyone. See [SAFETY.md](SAFETY.md).

## Why

Dementia (Alzheimer's disease, frontotemporal dementia and others) affects tens of
millions of people. Much of the useful public data and methods are scattered across
papers and repositories. This project puts reproducible, open pipelines in one place so
people with different skills can build on each other's work.

## Modules

Each module is a self-contained folder with its own README, code and results.

| Module | Status | Question |
|---|---|---|
| [`modules/eeg_dementia`](modules/eeg_dementia) | ✅ baseline working | Can a routine resting EEG tell Alzheimer's / FTD apart from healthy aging? |
| [`modules/tau_spread_simulation`](modules/tau_spread_simulation) | 💡 open for contributors | Simulate how tau protein spreads along the brain's wiring (connectome). |
| [`modules/drug_repurposing`](modules/drug_repurposing) | 💡 open for contributors | Find existing approved drugs that may be relevant, using knowledge graphs (PrimeKG, Hetionet). |
| [`modules/speech_markers`](modules/speech_markers) | 💡 open for contributors | Early signs of cognitive decline in speech and language. |

Want to start a new module? Open an issue describing the question and the open data you'd use.

## Quick start

Requires macOS or Linux, Python 3.12, [uv](https://github.com/astral-sh/uv) and the AWS CLI (for anonymous data download).

```bash
make setup          # create .venv with pinned versions
make data           # download OpenNeuro ds004504 (~2.9 GB, no account needed)
make eeg-features   # extract spectral EEG features
make eeg-baseline   # train and evaluate baseline models
```

## Ground rules

1. **No data leakage.** Train/test splits are always by person, never by recording window.
2. **Open data only**, with its license recorded in [DATA.md](DATA.md). No private patient data, ever.
3. **Report honestly.** Show chance level, spread across repeats, and what didn't work.
4. **Plain language.** Every result has a short explanation a non-expert can follow.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md), [GOVERNANCE.md](GOVERNANCE.md) (roles and how decisions are made) and our [Code of Conduct](CODE_OF_CONDUCT.md).

## License

Code: [Apache License 2.0](LICENSE). Redistributions must keep the [NOTICE](NOTICE) file,
which credits the founder and contributors. Documentation and write-ups: CC BY 4.0.
Datasets keep their own licenses (see [DATA.md](DATA.md)).

---

### בעברית בקצרה

OpenEngram הוא פרויקט פתוח לחקר וזיהוי דמנציה, שהוקם על ידי דוד אברמוב. כל אחד יכול לתרום: קוד, ניתוח נתונים,
ידע מקצועי או הסברים בשפה פשוטה. זה פרויקט מחקר בלבד ולא כלי רפואי.
