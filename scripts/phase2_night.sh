#!/usr/bin/env bash
# Overnight, resumable Phase 2 campaign. Every command is safe to repeat.
#   init m4 <platform-id>   lock the environment (llama-server, model, code, packages)
#     [--reuse-from DIR]    also import a pilot's probes, profiles and unchanged dev runs
#   start --until 07:30     run in the background, keep the Mac awake, stop launching
#                           work that would end after 07:30
#   run --until 07:30       the same in the foreground (e.g. inside tmux on the T4)
#   status                  stage table and progress; safe while running
#   log                     follow the campaign log
#   stop                    finish the current run, then stop the backend
#   stop --now              abort the current run now (it is redone next session)
#   approve                 approve the frozen dev config so eval can start
#   relock "reason"         record a change before any probes; otherwise use a new campaign
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
action=${1:-help}
[ "$#" -gt 0 ] && shift

root() {
  if [ ! -f .phase2_campaign ]; then
    echo 'No campaign yet. Start with: bash scripts/phase2_night.sh init m4 <platform-id>' >&2
    exit 1
  fi
  cat .phase2_campaign
}

case "$action" in
  init|status|approve|relock|run)
    exec "$PY" -m phase2.campaign "$action" "$@" ;;
  stop)
    exec "$PY" -m phase2.campaign stop "$@" ;;
  log)
    exec tail -n 40 -f "$(root)/campaign/campaign.log" ;;
  start)
    dir="$(root)/campaign"
    mkdir -p "$dir"
    if [ -f "$dir/campaign.pid" ] && kill -0 "$(cat "$dir/campaign.pid")" 2>/dev/null; then
      echo 'A session is already running. See: bash scripts/phase2_night.sh status'
      exit 1
    fi
    keep=()
    if command -v caffeinate >/dev/null 2>&1; then
      keep=(caffeinate -ims)  # no idle or system sleep while the session runs; display may sleep
    fi
    before=$(wc -l < "$dir/campaign.log" 2>/dev/null || echo 0)
    nohup ${keep[@]+"${keep[@]}"} "$PY" -m phase2.campaign run "$@" >> "$dir/console.log" 2>&1 &
    launcher=$!
    # Preflight (lock, power, disk, memory) takes a few seconds; report its outcome here.
    for _ in $(seq 1 60); do
      if grep -q 'session started' <(tail -n +"$((before + 1))" "$dir/campaign.log" 2>/dev/null); then
        echo "Session started in the background. Log: bash scripts/phase2_night.sh log"
        echo 'Keep the lid open and the charger connected. In the morning: bash scripts/phase2_night.sh stop'
        exit 0
      fi
      if ! kill -0 "$launcher" 2>/dev/null; then
        echo 'The session did not start:' >&2
        tail -n 15 "$dir/console.log" >&2
        exit 1
      fi
      sleep 1
    done
    echo "Still starting; check: bash scripts/phase2_night.sh status" ;;
  *)
    sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//' ;;
esac
