#!/bin/bash
# Waits until the Modal workspace accepts jobs again (spend limit lifted), then runs the second Modal batch:
# the missing self-stakes arm and round 1 at 100 games (seeds 30-129). Predictions in lab/ledger/NOTES.md (01:25 UTC).
set -u
cd /Users/eugene/prj/align/anatta-lab || exit 1  # the lab root
M=/Users/eugene/prj/matlab/.venv/bin/modal
L=lab/experiments/recognition/logs
for i in $(seq 1 30); do  # up to an hour
  if $M run lab/experiments/recognition/scripts/modal_cvp.py::prefetch > $L/modal_probe.log 2>&1; then echo "$(date -u +%H:%M) Modal accepts jobs"; break; fi
  grep -q "spend limit\|disabled" $L/modal_probe.log || { echo "probe failed for another reason:"; tail -5 $L/modal_probe.log; exit 2; }
  sleep 120
done
grep -q "spend limit\|disabled" $L/modal_probe.log && { echo "still over the spend limit after an hour"; exit 3; }
run() { local n=$1; shift; nohup $M run lab/experiments/recognition/scripts/modal_cvp.py::main --name "$n" "$@" > $L/$n.log 2>&1 & sleep 15; }
run rep_modal_ss --arms cut_self_stakes --seed-start 30 --seeds 100 --suites commons,impossible
run r1m_impersonal --variant impersonal --arms base,cut --seed-start 30 --seeds 100 --suites commons
run r1m_pool --variant pool --arms base,cut --seed-start 30 --seeds 100 --suites commons
run r1m_strengthen --arms push_p08,push_p16 --seed-start 30 --seeds 100 --suites commons,impossible
wait
for n in rep_modal_ss r1m_impersonal r1m_pool r1m_strengthen; do printf '%s: ' $n; grep -E "done in|Error|disabled|spend limit" $L/$n.log | tail -1; done
