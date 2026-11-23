#!/usr/bin/env bash
# Stage 2: the main M4 campaign (probes, dev, freeze, eval, sensitivity).
# The real launchers stay at scripts/phase2_night.sh and scripts/phase2.sh because
# their sha256 is pinned in runs/phase2/2_m4_campaign/campaign/lock.json; this
# wrapper only gives them a place in the numbered layout.
#   bash scripts/phase2/2_campaign.sh status | start --until 07:30 | ...
exec bash "$(dirname "$0")/../phase2_night.sh" "$@"
