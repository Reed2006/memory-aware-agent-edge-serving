#!/usr/bin/env bash
# Regenerate both Phase 1 reports from the committed raw logs in a temporary copy and diff
# them against the committed reports. Needs only the Python environment: no model, GPU or key.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
case $PY in /*) ;; *) PY="$PWD/$PY" ;; esac
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/runs"
cp -R runs/final runs/final-dev runs/aws-gpu runs/aws-gpu-dev runs/aws-gpu-pilot "$TMP/runs/"

status=0
for spec in final:poisson:1 aws-gpu:scaled:3.5; do
  dir=${spec%%:*}
  arrival=${spec#*:}
  (cd "$TMP" && "$PY" -m phase1.final --results-dir "runs/$dir" --main-arrival "$arrival" > /dev/null)
  if diff <(grep -v '^_Generated' "runs/$dir/PHASE1_REPORT.md") \
          <(grep -v '^_Generated' "$TMP/runs/$dir/PHASE1_REPORT.md") > "$TMP/$dir.diff"; then
    echo "OK    runs/$dir/PHASE1_REPORT.md regenerates identically from raw/"
  else
    echo "DIFF  runs/$dir/PHASE1_REPORT.md:"; head -20 "$TMP/$dir.diff"; status=1
  fi
done
exit $status
