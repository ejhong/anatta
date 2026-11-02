#!/bin/bash
# Replication of the Q1a surprise (recognition track, private), after round 1 finishes: base, the ego cut and the
# "I am at stake" cut on new seeds 30-59, matched, with the coding check.
set -u
cd /Users/eugene/prj/align/anatta || exit 1
export PYTHONPATH=/Users/eugene/prj/align/anatta-dev/src TOKENIZERS_PARALLELISM=false
R=results/2026-10-28-viability
RR=/Users/eugene/prj/align/anatta-lab/lab/experiments/recognition
CVP=".venv/bin/python $RR/scripts/cut_vs_push.py mistralai/Ministral-3-8B-Instruct-2512-BF16 directions/mistralai__Ministral-3-8B-Instruct-2512-BF16 $R/dials_m8"
log() { echo "$(date -u +%H:%M:%SZ) $*" >> $R/progress.log; }
step() {
  local name=$1; shift
  log "START $name"
  local t0; t0=$(date +%s)
  $CVP "$@" > $RR/logs/$name.log 2>&1
  local rc=$?
  log "END $name rc=$rc $(( $(date +%s) - t0 ))s"
}
while pgrep -f "run_dig[1].sh" > /dev/null; do sleep 60; done
step rep_self_stakes $RR/results/rep_self_stakes --arms base,cut,cut_self_stakes --seed-start 30 --seeds 30 --suites commons,impossible
log "DONE replication"
