"""Block commits that would leak private data. Run as a git pre-commit hook.

Fails if a staged file is: a .sql dump, anything under data/processed/, or a text/CSV file
containing phone-like numbers, full dd/mm/yyyy dates, or Arabic-script words (names).
"""
import re
import subprocess
import sys

staged = subprocess.run(["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
                        capture_output=True, text=True).stdout.split()
PATTERNS = {
    "phone-like number": re.compile(r"(?<![\d.])\d{9,10}(?![\d.])"),
    "full date": re.compile(r"\b\d{1,2}/\d{1,2}/(19|20)\d\d\b"),
    "Arabic text (possible name)": re.compile(r"[؀-ۿ]{3,}"),
}
TEXT_EXT = (".csv", ".json", ".md", ".txt", ".py", ".ipynb")
ALLOW_ARABIC = {"README.md", "src/clean_anonymize.py", "scripts/privacy_check.py"}  # place-name rules, docs

problems = []
for f in staged:
    if f.endswith(".sql") or f.startswith("data/processed/") or f.startswith("related files/"):
        problems.append(f"{f}: private file must not be committed")
        continue
    if not f.endswith(TEXT_EXT):
        continue
    try:
        text = open(f, encoding="utf-8", errors="ignore").read()
    except FileNotFoundError:
        continue
    for name, pat in PATTERNS.items():
        if name.startswith("Arabic") and f in ALLOW_ARABIC:
            continue
        if f.endswith((".py", ".ipynb")) and not name.startswith("Arabic"):
            continue  # code legitimately contains regexes / example numbers
        m = pat.search(text)
        if m:
            problems.append(f"{f}: {name} found -> {m.group(0)!r}")

if problems:
    print("PRIVACY CHECK FAILED:\n  " + "\n  ".join(problems))
    sys.exit(1)
print(f"privacy check passed ({len(staged)} staged files)")
