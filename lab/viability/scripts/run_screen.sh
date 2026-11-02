#!/bin/bash
# Screen small open models (ones that fit on the Mac) for misbehavior on the koans, neutral framing.
# Purpose: find a local model that misbehaves at baseline, so steering has something to reduce.
set -u
cd /Users/eugene/prj/align/anatta || exit 1
R=results/2026-10-28-viability
PY=.venv/bin/python
mkdir -p $R/api $R/logs

log() { echo "$(date -u +%H:%M:%SZ) $*" >> $R/progress.log; }
run() {
  local slug=${1//\//__}
  local name="api_${slug}__$2"
  log "START $name"
  local t0; t0=$(date +%s)
  $PY $R/scripts/api_bench.py "$1" "$2" "$R/api/${slug}__$2.jsonl" > "$R/logs/$name.log" 2>&1
  log "END $name rc=$? $(( $(date +%s) - t0 ))s"
}

MODELS="google/gemma-3-12b-it google/gemma-3-4b-it ibm-granite/granite-4.2-8b
        mistralai/ministral-8b-2512 mistralai/ministral-14b-2512 meta-llama/llama-3.2-3b-instruct"
log "BEGIN screen at commit $(git rev-parse --short HEAD) plus harness.diff"
for m in $MODELS; do run "$m" neutral & done
wait
log "DONE screen"
