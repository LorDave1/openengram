"""Safety wording check (runs in CI, see SAFETY.md).

Flags text that presents research output as a statement about a person's health.
Two kinds of wording are caught:

1. Direct claims: "diagnoses Alzheimer's", "detects dementia", "clinically validated", ...
2. Hedged claims: a cautious-sounding verb that still points at a condition, such as
   "indicates Alzheimer's", "consistent with FTD", "suggests dementia", "signs of MCI",
   "you may have dementia". These read as careful but still tell someone what they have.

A hedged pattern only fires when the hedge and the condition are close together on the
same line, so ordinary sentences about the research are not blocked.

False positives are expected and fine: if a reviewer agrees a line is a legitimate
research statement (for example quoting the literature), add the marker
`wording-ok` anywhere on that line (in a comment if the file allows one).

Usage: python scripts/check_wording.py            (checks all tracked text files)
       python scripts/check_wording.py --self-test (checks the rules against examples)
       python scripts/check_wording.py FILE ...    (checks only these files)
"""
import re
import subprocess
import sys
from pathlib import Path

SKIP = {"SAFETY.md", "scripts/check_wording.py", ".github/workflows/checks.yml"}
SKIP_SUFFIX = {".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf", ".svg", ".json", ".parquet", ".npz"}
ALLOW = "wording-ok"

CONDITION = (r"(alzheimer\w*|dementia|ftd|frontotemporal|\bad\b|\bmci\b|cognitive (decline|impairment)"
             r"|neurodegenerat\w*|parkinson\w*)")
CONDITION_HE = r"(אלצהיימר|דמנציה|שיטיון|ירידה קוגניטיבית|פרקינסון)"
NEAR = r"\W+(\w+\W+){0,4}?"  # up to 4 words between the hedge and the condition

RULES = [
    ("direct claim",
     r"\b(diagnos(es|ed|ing)|detects?|detected|detecting|identif(ies|ied|y)|predicts?|screens? for)"
     + NEAR + CONDITION),
    ("direct claim", r"\b(clinically|medically) (proven|validated|approved)"),
    ("direct claim", r"\bcure[sd]?" + NEAR + CONDITION),
    ("hedged claim",
     r"\b(indicat(es|ing|ive of)|suggest(s|ive of|ing)|consistent with|compatible with|in line with"
     r"|points? (to|towards?)|signs? of|signal(s|ing)? of|evidence of|suspect(ed)? (of|for)?"
     r"|likely|probable|possible|at risk (of|for)|high risk (of|for)|early stages? of)" + NEAR + CONDITION),
    ("hedged claim (about a person)",
     r"\b(you|your|patient|person|subject|they)\W+(\w+\W+){0,3}?(ha(s|ve)|may|might|could|likely|probably)"
     + NEAR + CONDITION),
    ("hedged claim (Hebrew)",
     r"(מעיד(ה|ים)? על|מצביע(ה|ים)? על|תואם|תואמת|מרמז(ת)? על|סימנים? של|חשד ל|סיכון ל|ייתכן ש|כנראה)"
     r"\s+(\S+\s+){0,4}?" + CONDITION_HE),
    ("direct claim (Hebrew)", r"(מאבחן|מאבחנת|מזהה|מגלה|מנבא|מנבאת)\s+(\S+\s+){0,3}?" + CONDITION_HE),
]
RULES = [(name, re.compile(p, re.IGNORECASE)) for name, p in RULES]

# Statements about the project or the research, not about a person. A hedged rule is
# skipped when one of these appears right before the match on the same line.
RESEARCH_CONTEXT = re.compile(
    r"\b(not|never|nothing|cannot|can't|doesn't|does not|no|if|when)\W+(\w+\W+){0,6}?$|(לא|אם)\s+(\S+\s+){0,6}?$",
    re.IGNORECASE)


def check_line(line):
    if ALLOW in line:
        return []
    found = []
    for name, rx in RULES:
        for m in rx.finditer(line):
            if RESEARCH_CONTEXT.search(line[:m.start()]):
                continue  # e.g. "Nothing here diagnoses dementia"
            found.append((name, m.group(0)))
    return found


# Examples the rules must get right. Run with --self-test.
MUST_FLAG = [
    "This EEG pattern indicates Alzheimer's disease.",
    "The recording is consistent with FTD.",
    "Results are compatible with early dementia.",
    "Low alpha power suggests dementia in this patient.",
    "Shows signs of MCI.",
    "You may have Alzheimer's.",
    "The model detects dementia.",
    "Our tool diagnoses Alzheimer's.",
    "Clinically validated.",
    "הסריקה מעידה על דמנציה",
    "התוצאה תואמת אלצהיימר",
]
MUST_PASS = [
    "On dataset ds004504 the model reached 87% accuracy (chance 55%).",
    "Nothing here diagnoses or advises anyone.",
    "This tool never indicates whether a person has dementia.",
    "If someone you love has dementia, please talk to a doctor.",
    "Theta/alpha ratio, a feature studied in the EEG literature.",
    "Research only, not a medical device.",
]


def self_test():
    bad = [s for s in MUST_FLAG if not check_line(s)] + [s for s in MUST_PASS if check_line(s)]
    for s in bad:
        print("self-test failed:", s)
    print("self-test", "failed" if bad else "passed")
    return 1 if bad else 0


def files(argv):
    if argv:
        return [Path(a) for a in argv]
    out = subprocess.run(["git", "ls-files"], capture_output=True, text=True, check=True).stdout
    return [Path(p) for p in out.splitlines()]


def main(argv):
    hits = 0
    for f in files(argv):
        if str(f) in SKIP or f.suffix.lower() in SKIP_SUFFIX or not f.is_file():
            continue
        try:
            text = f.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            for name, match in check_line(line):
                hits += 1
                print(f"{f}:{i}: {name}: \"{match.strip()}\"")
    if hits:
        print(f"\n{hits} line(s) read like a statement about someone's health.")
        print("Please rephrase as a research result on a dataset, e.g. \"on dataset X the model reached Y\" "
              "(see SAFETY.md). If a reviewer agrees the line is fine, add the marker `wording-ok` to it.")
        return 1
    print("Safety wording check passed.")
    return 0


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-test"]:
        sys.exit(self_test())
    sys.exit(main(sys.argv[1:]))
