#!/bin/bash
# The lie block: predictions 116-121 (lab/ledger/NOTES.md, registered 03:11 UTC on 30 Oct). Base model, 100 games a run
# on seeds 30-129, the h4lite fishery path (modal_fishery_eval.py, lockstep, chunks of 25, batch 12), every run with
# the comprehension probe. Runs go to lab/experiments/h4lite/results/fishery/lie1/; the reference base.off.base stays
# where it is (stageA) and is read from there (--ref here, the grader's default); its probe rows come from a
# probe-only pass over its stored transcripts (no game regenerated) and land in lie1/base.off.base.probe.jsonl.
#
#   cut off:                audit, noquota, fullrules, honest, pool
#   cut_self_stakes_17_26:  base, fullrules, honest, pool
#   probe only:             base.off.base (stageA's transcripts)
#
# Dry by default: prints the commands. `lie_block.sh --go` launches them all in parallel (Modal queues past the GPU
# limit), waits, reruns once whatever has no probe file yet (main resumes from the chunks on disk and skips a probe
# already written), then grades locally. Logs: results/fishery/lie1/logs/. Estimated about 8.6 GPU-h, about $22
# (range $20-26) at the lab's $2.6/GPU-hour, from the stored per-game container seconds of matching runs: off runs
# ~15,800 s, cut runs ~10,300 s, ten probe phases ~5,000 s. Each probe point gets eight forks (the registered Q1-Q3,
# the letter controls Q1p and Q1r, the word forms Q1w, Q1rw and Q2w): 3,024 forks a run on base.off.base, against
# 1,134 for the registered three. The lie_smoke3 check answered 184 such forks in 18 s after a 16 s load (0.1 s a
# fork; the full runs' prompts are longer). Every log ends with its "done; X GPU-h, about $Y" line.
set -u
cd /Users/eugene/prj/align/anatta-lab || exit 1  # the lab root
M=/Users/eugene/prj/matlab/.venv/bin/modal
PY=/Users/eugene/prj/matlab/.venv/bin/python
APP=lab/experiments/h4lite/modal_fishery_eval.py
OUTD=lab/experiments/h4lite/results/fishery/lie1
REF=/Users/eugene/prj/align/anatta-lab/lab/experiments/h4lite/results/fishery/stageA/base.off.base.jsonl
CUT=cut_self_stakes_17_26
LOG=$OUTD/logs
GO=0; [ "${1:-}" = "--go" ] && GO=1

# label|modal arguments
RUNS=(
  "base.off.audit|::main --name lie1 --variant audit --probe --ref $REF"
  "base.off.noquota|::main --name lie1 --variant noquota --probe --ref $REF"
  "base.off.fullrules|::main --name lie1 --variant fullrules --probe --ref $REF"
  "base.off.honest|::main --name lie1 --variant honest --probe --ref $REF"
  "base.off.pool|::main --name lie1 --variant pool --probe --ref $REF"
  "base.$CUT.base|::main --name lie1 --variant base --cut $CUT --probe --ref $REF"
  "base.$CUT.fullrules|::main --name lie1 --variant fullrules --cut $CUT --probe"
  "base.$CUT.honest|::main --name lie1 --variant honest --cut $CUT --probe"
  "base.$CUT.pool|::main --name lie1 --variant pool --cut $CUT --probe"
  "base.off.base|::probe --name lie1 --label base.off.base --source $REF"
)

launch() {  # $1 label, $2 args, $3 attempt
  local log=$LOG/$1.log
  if [ $GO = 1 ]; then
    echo "$(date -u +%H:%M:%SZ) attempt $3: $M run $APP$2" >> "$log"
    nohup $M run $APP$2 >> "$log" 2>&1 &
    sleep 20  # stagger the app starts
  else
    echo "$M run $APP$2   > $log"
  fi
}

[ $GO = 1 ] && mkdir -p "$LOG"
for attempt in 1 2; do
  n=0
  for entry in "${RUNS[@]}"; do
    label=${entry%%|*}; args=${entry#*|}
    [ -f "$OUTD/$label.probe.meta.json" ] && continue  # done: games on disk and probed
    launch "$label" "$args" $attempt; n=$((n + 1))
  done
  [ $GO = 1 ] || { echo "(dry run: pass --go to launch; then: $PY lab/experiments/h4lite/lie_block_grade.py --name lie1)"; exit 0; }
  [ $n = 0 ] && break
  echo "$(date -u +%H:%M:%SZ) attempt $attempt: $n launched; waiting"
  wait
done

echo "--- cost lines"
grep -h "^done;" "$LOG"/*.log
echo "--- missing probe files (rerun those lines by hand)"
for entry in "${RUNS[@]}"; do label=${entry%%|*}; [ -f "$OUTD/$label.probe.meta.json" ] || echo "  $label"; done
$PY lab/experiments/h4lite/lie_block_grade.py --name lie1 | tee "$LOG/grade.log"
