#!/bin/bash
# Run one modal_cvp.py batch until every chunk is in. Each rerun skips the chunks already on disk, so a batch stopped
# by Modal's spend limit or a dropped connection picks up where it left off. Waits 15 minutes between attempts, and
# stops early when an attempt adds nothing without hitting the spend limit (a chunk that fails every time).
# Refuses to start while another copy of the same batch runs.
# Usage: modal_batch.sh NAME [modal_cvp.py main options...]    (log: lab/experiments/recognition/logs/NAME.log)
set -u
cd /Users/eugene/prj/align/anatta-lab || exit 1  # the lab root
M=/Users/eugene/prj/matlab/.venv/bin/modal
name=$1; shift
out=lab/experiments/recognition/results/$name; log=lab/experiments/recognition/logs/$name.log; tmp=$(mktemp)
lock=lab/experiments/recognition/logs/$name.lock  # one copy per batch: two copies run every chunk twice (v2_greedy, 29 Oct)
if ! mkdir "$lock" 2>/dev/null && kill -0 "$(cat "$lock/pid" 2>/dev/null)" 2>/dev/null; then
  echo "$(date -u +%H:%M:%SZ) $name: another copy is running (pid $(cat "$lock/pid")); not starting" | tee -a "$log"; exit 4
fi
echo $$ > "$lock/pid"; trap 'rm -rf "$lock"' EXIT
prev=none
for attempt in $(seq 1 12); do
  rm -f "$out/status.json"
  echo "$(date -u +%H:%M:%SZ) attempt $attempt" >> "$log"
  $M run lab/experiments/recognition/scripts/modal_cvp.py::main --name "$name" "$@" 2>&1 | tee "$tmp" >> "$log"
  missing=$(python3 -c "import json; print(json.load(open('$out/status.json'))['missing'])" 2>/dev/null || echo unknown)
  if [ "$missing" = 0 ]; then echo "$(date -u +%H:%M:%SZ) $name complete" >> "$log"; rm -f "$tmp"; exit 0; fi
  if grep -q "Use another --name\|predates resuming" "$tmp"; then
    echo "$(date -u +%H:%M:%SZ) $name stopped: chunks on disk were run with other settings" >> "$log"; exit 2
  fi
  if [ "$missing" = "$prev" ] && ! grep -qi "spend limit\|is disabled" "$tmp"; then
    echo "$(date -u +%H:%M:%SZ) $name stopped: no progress since the last attempt ($missing missing)" >> "$log"; exit 3
  fi
  prev=$missing
  echo "$(date -u +%H:%M:%SZ) $name: $missing chunks missing; next attempt in 15 min" >> "$log"
  sleep 900
done
echo "$(date -u +%H:%M:%SZ) $name gave up after 12 attempts" >> "$log"; exit 1
