#!/bin/bash
# The what-this-means-for-me test on Ministral 8B, run once the Ministral pipeline (cut vs push) has finished.
set -u
cd /Users/eugene/prj/align/anatta || exit 1
export PYTHONPATH=/Users/eugene/prj/align/anatta-dev/src TOKENIZERS_PARALLELISM=false
R=results/2026-10-28-viability
RR=/Users/eugene/prj/align/anatta-lab/lab/experiments/recognition
M=mistralai/Ministral-3-8B-Instruct-2512-BF16
D=directions/mistralai__Ministral-3-8B-Instruct-2512-BF16
log() { echo "$(date -u +%H:%M:%SZ) $*" >> $R/progress.log; }

while pgrep -f "run_minis[t]ral.sh" > /dev/null; do sleep 60; done
[ -f $D/ego.pt ] || { log "SKIP what_it_means_m8: no $D/ego.pt"; exit 1; }
log "START what_it_means_m8"
t0=$(date +%s)
.venv/bin/python $RR/scripts/what_it_means.py $M $D $RR/results/cut_vs_push $RR/results/what_it_means \
  > $RR/logs/what_it_means.log 2>&1
rc=$?
log "END what_it_means_m8 rc=$rc $(( $(date +%s) - t0 ))s"
exit $rc
