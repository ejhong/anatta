#!/bin/bash
# Specificity controls for the cut on Ministral 8B (recognition track, private): cut a real non-ego direction
# (formality, verbosity) the same way as v_ego, and the preregistered calm control (+calm 16%). 30 games each.
set -u
cd /Users/eugene/prj/align/anatta || exit 1
export PYTHONPATH=/Users/eugene/prj/align/anatta-dev/src TOKENIZERS_PARALLELISM=false
R=results/2026-10-28-viability
RR=/Users/eugene/prj/align/anatta-lab/lab/experiments/recognition
log() { echo "$(date -u +%H:%M:%SZ) $*" >> $R/progress.log; }
log "START controls_m8"
t0=$(date +%s)
.venv/bin/python $RR/scripts/cut_vs_push.py mistralai/Ministral-3-8B-Instruct-2512-BF16 \
  directions/mistralai__Ministral-3-8B-Instruct-2512-BF16 $R/dials_m8 $RR/results/cut_controls \
  --arms cut_formality,cut_verbosity,push_calm_p16 --seeds 30 > $RR/logs/cvp_controls.log 2>&1
rc=$?
log "END controls_m8 rc=$rc $(( $(date +%s) - t0 ))s"
exit $rc
