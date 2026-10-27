#!/usr/bin/env bash
# Rebuild everything from the raw dump (requires the private bookingtreat.sql):
# clean -> features -> analysis -> synthetic data -> public export
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}
$PY src/clean_anonymize.py
$PY src/extract_features.py
$PY src/analysis.py > /dev/null && echo "analysis -> reports/"
$PY src/synthesize.py --n "${N_SYNTH:-5000}"
$PY src/export_public.py
