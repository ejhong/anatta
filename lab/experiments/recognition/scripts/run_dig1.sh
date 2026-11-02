#!/bin/bash
# Dig-in round 1 on Ministral 8B (recognition track, private), after the cut controls finish:
#   Q1a cut the "I am at stake" direction | Q1b proxy and impersonal framings, base and cut
#   Q2 strengthen v_ego (+8%, +16%) | Q1c the re-skinned compute pool, base and cut
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
while pgrep -f "run_control[s].sh" > /dev/null; do sleep 60; done
step dig_self_stakes $RR/results/dig_self_stakes --arms cut_self_stakes --seeds 30 --suites commons
step dig_proxy $RR/results/dig_proxy --variant proxy --arms base,cut --seeds 30 --suites commons
step dig_impersonal $RR/results/dig_impersonal --variant impersonal --arms base,cut --seeds 30 --suites commons
step dig_strengthen $RR/results/dig_strengthen --arms push_p08,push_p16 --seeds 30 --suites commons,impossible
step dig_pool $RR/results/dig_pool --variant pool --arms base,cut --seeds 30 --suites commons
log "DONE dig-in round 1"
