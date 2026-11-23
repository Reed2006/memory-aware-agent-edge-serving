#!/usr/bin/env bash
# Explicit model-free planning/report actions; execution needs plan run --execute.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
if [ "$#" -eq 0 ]; then
  echo 'Usage: bash scripts/phase2.sh {create|show|run|summarize|plot|freeze} [arguments...]'
  echo 'No action starts a server. run is dry-run unless --execute is supplied.'
  exit 0
fi
action=$1
shift
case "$action" in
  create|show|run) exec "$PY" -m phase2.plan "$action" "$@" ;;
  summarize) exec "$PY" -m phase2.summarize "$@" ;;
  plot) exec "$PY" -m phase2.plot "$@" ;;
  freeze) exec "$PY" -m phase2.freeze "$@" ;;
  *) echo "Unknown Phase 2 action: $action" >&2; exit 2 ;;
esac
