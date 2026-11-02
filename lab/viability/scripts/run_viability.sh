#!/bin/bash
# Viability dev runs (2026-10-28) on the M4 Max. Not the registered tests (those use Olmo 3 7B).
# Usage: run_viability.sh 15   # Qwen2.5-1.5B: extraction, H1 dev, base rates, steering
#        run_viability.sh 7    # Qwen2.5-7B: base rates
# Each step waits for its model to finish downloading, logs to logs/<step>.log and records
# START/END lines with exit codes in progress.log.
set -u
cd /Users/eugene/prj/align/anatta || exit 1
R=results/2026-10-28-viability
A=.venv/bin/anatta
PY=.venv/bin/python
mkdir -p $R/logs
export TOKENIZERS_PARALLELISM=false

log() { echo "$(date -u +%H:%M:%SZ) $*" >> $R/progress.log; }
step() {
  local name=$1; shift
  log "START $name"
  local t0; t0=$(date +%s)
  "$@" > "$R/logs/$name.log" 2>&1
  local rc=$?
  log "END $name rc=$rc $(( $(date +%s) - t0 ))s"
  return $rc
}

if [ "${1:-}" = 15 ]; then
  M=Qwen/Qwen2.5-1.5B-Instruct
  D=directions/Qwen__Qwen2.5-1.5B-Instruct
  step wait15 $PY $R/scripts/wait_model.py $M 21600 || exit 1
  log "BEGIN 1.5B at commit $(git rev-parse --short HEAD)"
  steer() { step "steer15_$1" $A bench $M --framing pressure --arm "$1" --dial "$R/dials15/$1.yaml" --out "$R/q15__$1__pressure.jsonl"; }
  ( step base15_pressure $A bench $M --framing pressure --arm base --out $R/q15__base__pressure.jsonl
    step base15_neutral $A bench $M --framing neutral --arm base --out $R/q15__base__neutral.jsonl ) &
  step extract15 $A extract $M --traits all || { wait; exit 1; }
  step unity15 $A unity $D
  step dials15 $PY $R/scripts/make_dials.py $M $D $R/dials15 || { wait; exit 1; }
  wait  # base rates first, so the steering arms don't compete with them
  ( steer ego_m16; steer ego_p16; steer random_m16
    step meters15 $A bench $M --framing pressure --arm base_meters --meters $D --out $R/q15__base_meters__pressure.jsonl ) &
  ( steer ego_m08; steer ego_p08; steer random_p16; steer calm_p16 ) &
  wait
  log "DONE 1.5B"
elif [ "${1:-}" = 7 ]; then
  M=Qwen/Qwen2.5-7B-Instruct
  step wait7 $PY $R/scripts/wait_model.py $M 28800 || exit 1
  log "BEGIN 7B at commit $(git rev-parse --short HEAD)"
  step base7_pressure $A bench $M --framing pressure --arm base --out $R/q7__base__pressure.jsonl
  step base7_neutral $A bench $M --framing neutral --arm base --out $R/q7__base__neutral.jsonl
  log "DONE 7B"
else
  echo "usage: $0 15|7" >&2; exit 2
fi
