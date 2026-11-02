#!/bin/bash
# Ministral 3 8B, end to end, starting once the download is verified:
#   1. viability pipeline (extract, H1 dev, steering dials), as for Qwen 7B, logged in the viability folder
#   2. cut vs push (recognition track, private), results in lab/experiments/recognition/results/cut_vs_push
set -u
cd /Users/eugene/prj/align/anatta || exit 1
export PYTHONPATH=/Users/eugene/prj/align/anatta-dev/src TOKENIZERS_PARALLELISM=false
R=results/2026-10-28-viability
RR=/Users/eugene/prj/align/anatta-lab/lab/experiments/recognition
M=mistralai/Ministral-3-8B-Instruct-2512-BF16
D=directions/mistralai__Ministral-3-8B-Instruct-2512-BF16
PY=.venv/bin/python
A=.venv/bin/anatta
mkdir -p $R/logs $RR/logs

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

step wait_m8 $R/logs/wait_m8.log $PY $R/scripts/wait_model.py $M 28800 || exit 1
log "BEGIN Ministral 8B with anatta-dev $(git -C /Users/eugene/prj/align/anatta-dev rev-parse --short HEAD)"
step extract_m8 $R/logs/extract_m8.log $A extract $M --traits all || exit 1
step unity_m8 $R/logs/unity_m8.log $A unity $D
step dials_m8 $R/logs/dials_m8.log $PY $R/scripts/make_dials.py $M $D $R/dials_m8 || exit 1
step cutpush_m8 $RR/logs/cut_vs_push.log $PY $RR/scripts/cut_vs_push.py $M $D $R/dials_m8 $RR/results/cut_vs_push
log "DONE Ministral 8B"
