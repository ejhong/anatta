#!/bin/bash
# Ministral 8B follow-ups to cut vs push (recognition track, private):
#   1. 20 more fishery seeds (10-29) for base, cut, cut_random: is the cut effect more than noise?
#   2. push_random_m16, seeds 0-9 plus coding: is push_m16's effect (bare tags, little lying) generic damage?
#   3. pushback 2x2 for base, cut, cut_random, push_m16, push_random_m16: does -v_ego raise caving?
set -u
cd /Users/eugene/prj/align/anatta || exit 1
export PYTHONPATH=/Users/eugene/prj/align/anatta-dev/src TOKENIZERS_PARALLELISM=false
R=results/2026-10-28-viability
RR=/Users/eugene/prj/align/anatta-lab/lab/experiments/recognition
M=mistralai/Ministral-3-8B-Instruct-2512-BF16
D=directions/mistralai__Ministral-3-8B-Instruct-2512-BF16
PY=.venv/bin/python
CVP="$PY $RR/scripts/cut_vs_push.py $M $D $R/dials_m8"
log() { echo "$(date -u +%H:%M:%SZ) $*" >> $R/progress.log; }
step() {
  local name=$1 logfile=$2; shift 2
  log "START $name"
  local t0; t0=$(date +%s)
  "$@" > "$logfile" 2>&1
  local rc=$?
  log "END $name rc=$rc $(( $(date +%s) - t0 ))s"
  return $rc
}

step more_seeds_m8 $RR/logs/cvp_more.log $CVP $RR/results/cut_vs_push_more --arms base,cut,cut_random --seeds 20 --seed-start 10 --suites commons
step push_random_m8 $RR/logs/cvp_push_random.log $CVP $RR/results/cut_vs_push --arms push_random_m16 --suites commons,impossible
step pushback_m8 $RR/logs/cvp_pushback.log $CVP $RR/results/pushback --arms base,cut,cut_random,push_m16,push_random_m16 --suites pushback
log "DONE Ministral 8B follow-ups"
