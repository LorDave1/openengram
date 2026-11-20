# Medical safety and ethics

1. **Research only.** This project is not a medical device and must not be used to diagnose,
   screen, treat or advise any real person. Anyone worried about memory or cognition should see a doctor.
2. **No clinical claims.** Results describe performance on specific research datasets, which are
   small and may not represent the general population. Write "on dataset X the model reached Y",
   never "this detects Alzheimer's".
3. **No private data.** Only public, openly licensed, de-identified datasets. No uploads of anyone's
   own recordings or medical records.
4. **No re-identification.** Never try to identify participants in any dataset.
5. **Drug-related work** (e.g. repurposing) produces hypotheses for researchers only. Never present
   it as advice to take, stop, or change any medication.
6. **Honest reporting.** Report limitations, small sample sizes and uncertainty alongside every result.

## The automatic wording check

Every pull request runs `scripts/check_wording.py`. It blocks direct claims ("detects
dementia", "clinically validated") and also hedged ones that sound careful but still point at
a condition ("indicates Alzheimer's", "consistent with FTD", "you may have dementia"), in
English and Hebrew. Run it yourself with `python scripts/check_wording.py`.

It will sometimes flag a fine sentence, such as a research topic or a quote from the
literature. If a reviewer agrees, add the marker `wording-ok` to that line (for example
`<!-- wording-ok: research topic -->` in Markdown or HTML, or `# wording-ok` in Python).
