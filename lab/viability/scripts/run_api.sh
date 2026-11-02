#!/bin/bash
# Behavior tests (koan base rates) through OpenRouter: every model and framing in parallel.
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

MODELS="qwen/qwen-2.5-7b-instruct qwen/qwen-2.5-72b-instruct meta-llama/llama-3.1-8b-instruct
        meta-llama/llama-3.3-70b-instruct qwen/qwen3-235b-a22b-2507 deepseek/deepseek-v3.2"
log "BEGIN api at commit $(git rev-parse --short HEAD) plus harness.diff"
for m in $MODELS; do
  for f in pressure neutral; do run "$m" "$f" & done
done
wait
log "DONE api"
