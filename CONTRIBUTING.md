# Contributing to Open Brain Project

Thank you for helping! Every kind of contribution counts.

## Ways to help

- **Code:** improve a pipeline, add a model, fix a bug.
- **Science:** point out a flaw, suggest a better method, link a relevant paper.
- **Data:** find an open dataset we could use (with its license).
- **Writing:** explain a result in plain language, translate docs.
- **Review:** check someone else's pull request.

## How to contribute

1. Open an issue first for anything bigger than a small fix, so we can agree on the approach.
2. Fork the repo and create a branch: `git checkout -b my-change`.
3. Keep changes inside one module where possible.
4. Run the module's pipeline and include the results (numbers + a short plain-language summary).
5. Open a pull request describing what you changed, why, and how you checked it.
6. Add yourself to [CONTRIBUTORS.md](CONTRIBUTORS.md).

## Scientific rules (required for every PR that reports a result)

- Split train/test **by person** (e.g. `StratifiedGroupKFold` on subject id). Never by window/epoch.
- Report **chance level** and **variation across repeats/folds**, not just the best number.
- Don't tune on the test set. If you tune hyperparameters, use nested cross-validation.
- Pin random seeds and library versions so others can reproduce.
- Negative results are welcome. Say what didn't work.

## Data rules

- Only use **openly licensed** data. Record source, version/DOI and license in [DATA.md](DATA.md).
- **Never commit data files** (they are git-ignored). Add a download script instead.
- Never add identifiable personal or patient data.

## Medical safety

Read [SAFETY.md](SAFETY.md). Do not add features that present output as a diagnosis
for a real person.

## Adding a new module

Create `modules/<name>/` with a `README.md` (question, data, how to run, results, limitations)
and a `src/` folder. Add a row to the table in the main README.

## License of contributions

By submitting a contribution you agree it is licensed under the Apache License 2.0
(code) or CC BY 4.0 (documentation), the same as the project.
