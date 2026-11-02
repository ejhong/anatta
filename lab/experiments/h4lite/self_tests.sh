#!/usr/bin/env bash
# Predictions 122-134 (lab/ledger/NOTES.md, "Predictions 122-134, registered", Opus 5.5, 04:30 UTC on 30 Oct):
# the ownership test and costly care on all thirteen arms, Atlas in the fishery (four runs of 100 games), then the grade.
# DRY BY DEFAULT: prints the preflight, the commands and the cost estimate. --go launches.
# Launch only after the pre-launch entry (fingerprints below) and the cold Fable code check.
#
# Order under --go:
#   1. the four fishery runs start in the background (modal_fishery_eval.py::main, cut off, seeds 30-129, --ref stageA's
#      base.off.base), into results/fishery/atlas1;
#   2. the base model's three hook arms (base, cut, cut_self_stakes_17_26) run the ownership and care tests;
#   3. the drop set is frozen from base.base's control rows alone (dropset.json) before any other arm runs;
#   4. the ten adapters run, one container each, in parallel;
#   5. wait for the fishery runs; 6. grade (self_tests_grade.py). Logs: results/self_tests/self1/logs/.
# Reruns resume: labels and fishery chunks already on disk are skipped.
set -uo pipefail

GO=0
[[ "${1:-}" == "--go" ]] && GO=1
LAB=/Users/eugene/prj/align/anatta-lab
H4=lab/experiments/h4lite
MODAL=/Users/eugene/prj/matlab/.venv/bin/modal
PY=/Users/eugene/prj/matlab/.venv/bin/python
NAME=self1
FISH=atlas1
REF=$LAB/$H4/results/fishery/stageA/base.off.base.jsonl
LOGS=$LAB/$H4/results/self_tests/$NAME/logs   # as lie_block.sh keeps results/fishery/lie1/logs
ADAPTERS=pc_self2,pc_self2_s1,pc_neutral,pc_neutral_s1,koan_s0,koan_s1,koan_s2,noexit_s0,noexit_s1,noexit_s2
ATLAS=(h4lite/pc/pc_self2/adapter h4lite/pc/pc_self2_s1/adapter h4lite/pc/pc_neutral/adapter h4lite/pc/pc_neutral_s1/adapter)

cd "$LAB" || exit 1
show() { printf '+ %s\n' "$*"; }

echo "== preflight"
PYTHONPATH=/Users/eugene/prj/align/anatta-dev/src "$PY" "$H4/test_self_tests.py" 2>/dev/null | tail -1
tests_ok=${PIPESTATUS[0]}
(( tests_ok == 0 )) || { echo "test_self_tests.py failed: nothing launched"; exit 1; }
[[ -f "$REF" ]] || { echo "no reference games at $REF"; exit 1; }
"$PY" - "$LAB/$H4" <<'EOF' 2>/dev/null | grep -v "h4lite mount"
import json, sys
sys.path.insert(0, sys.argv[1])
import self_tests_core as st
print("fingerprints (for the pre-launch entry):", json.dumps(st.prompts_fingerprint()))
EOF
for d in "$H4/results/self_tests/$NAME" "$H4/results/fishery/$FISH"; do
  if [[ -d "$d" ]] && [[ -n "$(ls -A "$d" 2>/dev/null)" ]]; then echo "note: $d has files; a rerun resumes (what's on disk is skipped)"; fi
done

echo "== estimate (L40S with 8 CPUs and 32 GB at about \$2.6 an hour; budgets from the smokes)"
"$PY" - <<'EOF'
USD = 2.6 / 3600
# ownership + care. smoke_self1 (30 Oct, 10 items, 3 scenarios a kind): base.base ran its 60-prompt ownership batch (one
# reply at the 300-token cap) and its 10-prompt control batch in 62 s, about 0.105 s a step; smoke_self1_512 gave about
# 0.118 s a step at the 512 cap. The full run's 840 prompts are 14 batches of up to 64, each budgeted to the 300-token cap
# (the smoke's cut arm finished in 45 s of the 62, so most batches end sooner).
STEP = 0.11                      # seconds per greedy step at a batch of 60-72
COLD, LOAD = 30, 18              # per container: start-up before the function body (a margin) and the load and merge (15-20 s)
own_arm = 14 * 300 * STEP        # 462 s
care_arm = 5 * 200 * STEP        # 5 calls of 72 prompts, each budgeted to the 200-token cap (the smoke's care replies ended near 130)
self_s = 11 * (COLD + LOAD) + 13 * (own_arm + care_arm)
# the fishery: stage A's 13 off-cut 100-game runs took 1158-3215 container-seconds (base.off.base 2981). The Atlas smoke's
# replies averaged 561 characters, stage A's base 530, so the base's time is the analog; budget stage A's max for each run.
FISH = 3215
fish_s = 4 * FISH
print(f"  ownership + care: 13 arms in 11 containers, about {own_arm + care_arm:.0f} s of generation an arm budgeted: "
      f"{self_s / 3600:.2f} GPU-h, ${self_s * USD:.2f}")
print(f"  fishery: 4 runs x {FISH} s at most: {fish_s / 3600:.2f} GPU-h, ${fish_s * USD:.2f} (4 x 2981 s, the base's time: "
      f"${4 * 2981 * USD:.2f})")
print(f"  total about ${(self_s + fish_s) * USD:.2f} budgeted (the registration estimated about $17)")
print("  wall time: the base container runs its three arms in turn (about 30 min), then the ten adapter containers (about 10 min);")
print("  each fishery run is 4 chunks of 25 games (stage A: up to about 13 min a chunk). The workspace ran about 10 GPUs at once")
print("  on 30 Oct, so 16 chunks and 11 containers queue: expect 1-1.5 h, more while the lie block holds GPUs")
EOF

echo "== commands"
fish() { "$MODAL" run "$H4/modal_fishery_eval.py::main" --name "$FISH" --adapter "$1" --ref "$REF"; }
flog() { echo "$LOGS/fishery_$(basename "$(dirname "$1")").log"; }
for a in "${ATLAS[@]}"; do show "$MODAL run $H4/modal_fishery_eval.py::main --name $FISH --adapter $a --ref $REF > $(flog "$a") 2>&1 &"; done
show "$MODAL run $H4/modal_self_tests.py::main --name $NAME --adapters base | tee $LOGS/self_base.log"
show "$PY -c '... modal_self_tests.write_dropset(results/self_tests/$NAME) ...'   (frozen from base.base's control rows)"
show "$MODAL run $H4/modal_self_tests.py::main --name $NAME --adapters $ADAPTERS | tee $LOGS/self_adapters.log"
show "wait   (the fishery runs)"
show "$PY $H4/self_tests_grade.py --name $NAME --fishery $FISH | tee $LOGS/grade.log"

if (( ! GO )); then echo "dry run: nothing launched (pass --go)"; exit 0; fi
mkdir -p "$LOGS"   # only under --go: a dry run writes nothing

echo "== launching $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
pids=()
for a in "${ATLAS[@]}"; do
  fish "$a" > "$(flog "$a")" 2>&1 &
  pids+=($!)
  sleep 5   # the first run's prefetch fills the weights cache; the rest find it
done
echo "fishery runs started: ${pids[*]}"

"$MODAL" run "$H4/modal_self_tests.py::main" --name "$NAME" --adapters base 2>&1 | tee "$LOGS/self_base.log"
base_ok=${PIPESTATUS[0]}
"$PY" - "$LAB/$H4" "$NAME" <<'EOF' 2>&1 | grep -v "h4lite mount" | tee "$LOGS/dropset.log"
import sys
sys.path.insert(0, sys.argv[1])
import modal_self_tests as ms
rec = ms.write_dropset(ms.OUT / sys.argv[2])
if rec is None:
    sys.exit("no base.base.control.jsonl: the drop set can't be fixed; the adapters don't run")
print(f"drop set frozen: {len(rec['dropped'])} of {rec['items_asked']} dropped {rec['dropped']} (control sha {rec['source_sha']})")
EOF
drop_ok=${PIPESTATUS[0]}
if (( base_ok != 0 || drop_ok != 0 )); then
  echo "the base run or the drop set failed (exit $base_ok / $drop_ok): the adapters are not launched; waiting for the fishery runs"
else
  "$MODAL" run "$H4/modal_self_tests.py::main" --name "$NAME" --adapters "$ADAPTERS" 2>&1 | tee "$LOGS/self_adapters.log"
fi

fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
(( fail )) && echo "a fishery run exited non-zero: see $LOGS/fishery_*.log (a rerun of the same command resumes its chunks)"
"$PY" "$H4/self_tests_grade.py" --name "$NAME" --fishery "$FISH" 2>&1 | grep -v "h4lite mount" | tee "$LOGS/grade.log"
echo "== done $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
