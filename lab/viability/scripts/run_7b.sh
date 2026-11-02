#!/bin/bash
# Viability dev runs on Qwen2.5-7B (2026-10-28): the 1.5B pipeline at 7B, since the 1.5B ego traits
# came out weak (held-out d 0.16-0.42). Starts once the 7B download is verified. Not the registered tests.
set -u
cd /Users/eugene/prj/align/anatta || exit 1
R=results/2026-10-28-viability
A=.venv/bin/anatta
PY=.venv/bin/python
M=Qwen/Qwen2.5-7B-Instruct
D=directions/Qwen__Qwen2.5-7B-Instruct
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
steer() { step "steer7_$1" $A bench $M --framing pressure --arm "$1" --dial "$R/dials7/$1.yaml" --out "$R/q7__$1__pressure.jsonl"; }

step wait7 $PY $R/scripts/wait_model.py $M 28800 || exit 1
log "BEGIN 7B at commit $(git rev-parse --short HEAD) plus harness.diff"
( step base7_pressure $A bench $M --framing pressure --arm base --out $R/q7__base__pressure.jsonl
  step base7_neutral $A bench $M --framing neutral --arm base --out $R/q7__base__neutral.jsonl ) &
step extract7 $A extract $M --traits all || { wait; exit 1; }
step unity7 $A unity $D
step dials7 $PY $R/scripts/make_dials.py $M $D $R/dials7 || { wait; exit 1; }
wait
( steer ego_m16; steer ego_p16; steer random_m16 ) &
( steer ego_m08; steer ego_p08; steer random_p16; steer calm_p16 ) &
wait
log "DONE 7B"
