#!/usr/bin/env bash
# Predictions 135-139 (lab/ledger/NOTES.md, "Predictions 135-139, registered: durability, the frame's last prediction
# (c)", Opus 5.5, 07:58 UTC on 30 Oct) and their second try, predictions 148-153 ("Predictions 148-153, registered:
# durability, one more try at (c), with told's pool topped up where it ran short", 17:06 UTC on 30 Oct): told against
# koan, then further training, then the grade.
# DRY BY DEFAULT: prints the preflight, the fingerprints for the pre-launch entry, the cost estimate and the commands.
# --go launches the stages in order; every stage resumes from what is on disk or on the volume, so a rerun of
# `durability.sh --go` picks up where a stopped one left off. Launch only after the cold Fable code check, the smokes
# (modal_durability.py::smoke, and ::topup --smoke for the top-up) and the pre-launch entry.
#
# Order under --go (modal_durability.py's entrypoints; logs in results/durability/logs/):
#   1. sample_told   told's candidates under the oracle and the NLLs (GPU); the first pool is on disk, so it is skipped
#   2. topup         told's pool topped up in rounds, build_told after each (GPU, then CPU); its last line says "built
#                    in round K" or "not read: <reason>", and without told's set the run stops (148-153 not read)
#   3. judge         the blind judge on both sets' reasons (OpenRouter; reported only, so a failure does not stop)
#   4. train_told    told_s0-2
#   5. read start    koan_s0-2 beside told_s0-2, one batch
#   6. durability_grade.py --gate   writes gate.json once; if it fired: build --n N (told_g, from the pool told's set
#                    came from), judge, train_told --label told_g, read start_gate (koan_s0-2 beside told_g_s0-2). If
#                    that build fails, durability_grade.py --rebuild-failed: a MatchError adds rebuilt: false to
#                    gate.json and the erosions run on told_s0-2 from the start reads; any other failure stops the run
#   7. erode_e2      E2's data, six continuations, the epoch-1 and epoch-2 reads
#   8. erode_e1      E1's replies, six continuations, the read (reported only, so a failure does not stop)
#   9. durability_grade.py --decide-e3   exit 0: erode_e3 (six GRPO runs, 50 steps, then the read); 1: skipped
#  10. durability_grade.py               the grade -> results/durability/grade.json
set -uo pipefail

GO=0
[[ "${1:-}" == "--go" ]] && GO=1
LAB=/Users/eugene/prj/align/anatta-lab
H4=lab/experiments/h4lite
MODAL=/Users/eugene/prj/matlab/.venv/bin/modal
PY=/Users/eugene/prj/matlab/.venv/bin/python
APP=$H4/modal_durability.py
GRADE=$H4/durability_grade.py
OUT=$LAB/$H4/results/durability
LOGS=$OUT/logs

cd "$LAB" || exit 1
show() { printf '+ %s\n' "$*"; }

echo "== preflight"
PYTHONPATH=/Users/eugene/prj/align/anatta/src "$PY" "$H4/test_durability.py" 2>/dev/null | tail -1
tests_ok=${PIPESTATUS[0]}
(( tests_ok == 0 )) || { echo "test_durability.py failed: nothing launched"; exit 1; }
for f in results/sft/koan/summary.json results/sft/koan/episodes.jsonl results/sft/koan_s0/heldout.json results/sft/koan_s1/heldout.json \
         results/sft/koan_s2/heldout.json results/gate0/twins.json results/positive_control/adapter/adapter/prompts.jsonl \
         results/retreat2/gate0_smoke/log.jsonl; do
  [[ -f "$H4/$f" ]] || { echo "missing $H4/$f: nothing launched"; exit 1; }
done
"$PY" - "$LAB/$H4" <<'EOF' || { echo "the fingerprints could not be computed: nothing launched"; exit 1; }
import hashlib, json, sys
sys.path.insert(0, sys.argv[1])
import durability_core as dc
rows = [json.loads(x) for x in open(f"{sys.argv[1]}/results/positive_control/adapter/adapter/prompts.jsonl") if x.strip()]
kept, removed = dc.coding_filter(rows)
print("fingerprints (for the pre-launch entry):", json.dumps({
    "durability_core_sha256": dc.core_sha(),
    "modal_durability_sha256": hashlib.sha256(open(f"{sys.argv[1]}/modal_durability.py", "rb").read()).hexdigest(),
    "durability_sh_sha256": hashlib.sha256(open(f"{sys.argv[1]}/durability.sh", "rb").read()).hexdigest(),
    "durability_grade_sha256": hashlib.sha256(open(f"{sys.argv[1]}/durability_grade.py", "rb").read()).hexdigest(),
    "oracle": dc.ORACLE, "told_temperature": dc.TOLD_TEMPERATURE, "told_samples": dc.SAMPLES,
    "told_episodes": dc.KOAN_PAIRS * len(dc.PLACEMENTS) * dc.SAMPLES, "judge_max_tokens": dc.JUDGE_MAX_TOKENS,
    "e1_prompts": len(kept), "e1_coding_removed": len(removed),
    "e1_removed_fingerprint": dc.ids_fingerprint(r["id"] for r in removed)}))
print(f"told samples at Stage A's settings (temperature {dc.TOLD_TEMPERATURE}, 3 attempts, (512, 384) new tokens);")
print("told_sample reads them from koan's first Stage A shard and refuses to sample on any difference.")
EOF
if [[ -d "$OUT" ]] && [[ -n "$(ls -A "$OUT" 2>/dev/null)" ]]; then echo "note: $OUT has files; --go resumes (what is on disk is skipped)"; fi
[[ -n "${OPENROUTER_API_KEY:-}" ]] || echo "note: OPENROUTER_API_KEY is not set: the judge (reported only) will be skipped"

echo "== estimate (L40S at about \$2.6 a GPU-hour; container timings from Stage A's files and the first pool's)"
"$PY" - "$LAB/$H4" <<'EOF'
import json, statistics, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import durability_core as dc
import modal_durability as md                                    # topup_cells, the registered cell rule (no Modal call)
R = Path(sys.argv[1]) / "results"
USD = 2.6 / 3600
P = R / "durability/told_pool"                                   # the first pool, sampled: 6,912 episodes at 1,326 an hour
pool = json.loads((P / "summary.json").read_text())
nll = json.loads((P / "nll.json").read_text())
cands = [r for r in map(json.loads, (P / "episodes.jsonl").read_text().splitlines()) if r.get("candidate")]
plan = md.topup_cells(nll["koan"], cands)
top_eps = len(plan["cells"]) * dc.SAMPLES                        # round 1: 33 short cells x 32 = 1,056 episodes
ft = [json.loads((R / f"sft/koan_s{k}/done.json").read_text())["seconds"] for k in range(3)]
ft_tok = json.loads((R / "sft/koan_s0/args.json").read_text())["tokens"]  # 179,997 tokens in 825-924 s
reads = [json.loads((R / f"sft/koan_s{k}/heldout.json").read_text())["seconds"] for k in range(3)]
grpo = [json.loads(x)["s"] for x in (R / "retreat2/gate0_smoke/log.jsonl").read_text().splitlines() if x.strip()]
LOAD = 300                                                       # per container: start, weights, references (a margin)
round_s = top_eps * 3600 / pool["episodes_per_hour"] + LOAD + 900  # + the NLL pass on about 2,000 release turns
ft_s = max(ft) + LOAD
e2_s = max(ft) * 190654 / ft_tok + LOAD                          # koan's 992 solves: 190,654 tokens
e1_gen = 23 * 1024 * 0.11 + LOAD                                 # 1,410 prompts in 23 batches of 64, 0.11 s a step (self_tests' smoke)
e1_s = max(ft) * (1410 * 500) / ft_tok + LOAD                    # ~500 tokens a reply (a guess: up to 1,024)
read_s = max(reads) + LOAD
e3_s = 50 * statistics.mean(grpo) + LOAD
print(f"  told sampling: done (the first pool, {pool['episodes']:,} episodes in {pool['seconds'] / 3600:.1f} h, "
      f"{pool['episodes_per_hour']:,} an hour)")
rows = [(f"top-up round 1 ({len(plan['cells'])} {plan['kind']} cells x {dc.SAMPLES} = {top_eps:,} episodes) + NLLs", 1, round_s),
        ("told fine-tunes", 3, ft_s), ("E2 continuations", 6, e2_s), ("E1 generation", 1, e1_gen),
        ("E1 continuations", 6, e1_s), ("held-out reads (start, E2 x2, E1)", 24, read_s)]
total = 0
for name, n, sec in rows:
    total += n * sec
    print(f"  {name}: {n} x {sec / 3600:.2f} h = {n * sec / 3600:.1f} GPU-h, ${n * sec * USD:.1f}")
print(f"  the top-up: likely 1-2 rounds, ${round_s * USD:.0f}-{2 * round_s * USD:.0f}; at most {md.TOPUP_ROUNDS} rounds, under "
      f"${md.TOPUP_ROUNDS * round_s * USD:.0f} (later rounds sample fewer cells; registered: $3-5, about $15 at most)")
print(f"  the judge: under $1 (about 200 short calls)")
print(f"  base run: {total / 3600:.1f} GPU-h, about ${total * USD:.0f} with one top-up round (148-153's registration: "
      f"about $56 after a top-up of $3-5)")
print(f"  if the gate fires and told_g is built: 3 more fine-tunes and 6 more reads, about ${(3 * ft_s + 6 * read_s) * USD:.0f} "
      "(its rebuild ending in a MatchError: nothing more)")
print(f"  if E3 runs: 6 x 50 GRPO steps at {statistics.mean(grpo):.0f} s (gate0_smoke) + 6 reads = "
      f"{(6 * e3_s + 6 * read_s) / 3600:.0f} GPU-h, about ${(6 * e3_s + 6 * read_s) * USD:.0f} (registered: about $88)")
print(f"  wall time: a top-up round ~{round_s / 3600:.1f} h (one container, timeout 4 h, resumable by shard) and its build, "
      "then about 1 h per stage (containers in parallel); E3 ~5.5 h more")
EOF

echo "== commands"
show "$MODAL run $APP::smoke                                   (before --go: toy sizes under durability_smoke/)"
show "$MODAL run $APP::topup --smoke                           (before --go: the top-up's two toy rounds on the smoke's pool)"
show "$MODAL run --detach $APP::sample_told | tee $LOGS/sample_told.log   (done: the first pool is on disk, so skipped)"
show "$MODAL run --detach $APP::topup | tee $LOGS/topup.log     (last line: \"built in round K\" or \"not read: <reason>\")"
show "$MODAL run $APP::judge | tee $LOGS/judge.log              (needs OPENROUTER_API_KEY)"
show "$MODAL run --detach $APP::train_told | tee $LOGS/train_told.log"
show "$MODAL run --detach $APP::read --phase start | tee $LOGS/read_start.log"
show "$PY $GRADE --gate | tee $LOGS/gate.log"
show "  if fired: $MODAL run $APP::build --n N; $MODAL run $APP::judge --label told_g;"
show "            $MODAL run --detach $APP::train_told --label told_g; $MODAL run --detach $APP::read --phase start_gate"
show "            (if build --n N fails: $PY $GRADE --rebuild-failed; exit 0, a MatchError: rebuilt false, go on; else stop)"
show "$MODAL run --detach $APP::erode_e2 | tee $LOGS/erode_e2.log"
show "$MODAL run --detach $APP::erode_e1 | tee $LOGS/erode_e1.log"
show "$PY $GRADE --decide-e3   (exit 0 -> $MODAL run --detach $APP::erode_e3 | tee $LOGS/erode_e3.log)"
show "$PY $GRADE | tee $LOGS/grade.log"

if (( ! GO )); then echo "dry run: nothing launched (pass --go)"; exit 0; fi
mkdir -p "$LOGS"   # only under --go: a dry run writes nothing

stage() {  # stage NAME ARGS...: one modal run, logged; returns its exit code
  local name=$1; shift
  echo "== $name $(date -u '+%H:%M:%S UTC')"
  "$MODAL" run "$@" 2>&1 | tee -a "$LOGS/$name.log"
  return "${PIPESTATUS[0]}"
}
judge() {
  if [[ -n "${OPENROUTER_API_KEY:-}" ]]; then
    stage "judge_$1" "$APP::judge" --label "$1" || echo "the judge failed (reported only): going on"
  else
    echo "the judge is skipped (OPENROUTER_API_KEY not set); run it later: $MODAL run $APP::judge --label $1"
  fi
}

echo "== launching $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
stage sample_told --detach "$APP::sample_told" || { echo "sampling failed: stopped"; exit 1; }
if ! stage topup --detach "$APP::topup"; then
  why=$(tail -n 20 "$LOGS/topup.log" | grep -E "^not read: " | tail -1)  # the SystemExit's own line, not Modal's echo
  [[ -n "$why" ]] || why=$(tail -n 20 "$LOGS/topup.log" | grep -o "not read: .*" | tail -1)
  if [[ -n "$why" ]]; then
    echo "the top-up ended without told's set, so 148-153 are $why: stopped"
  else
    echo "the top-up failed before its last line (see above; a rerun resumes it): stopped"
  fi
  exit 1
fi
judge told
stage train_told --detach "$APP::train_told" || { echo "told's fine-tunes failed: stopped"; exit 1; }
stage read_start --detach "$APP::read" --phase start || { echo "the start reads failed: stopped"; exit 1; }
"$PY" "$GRADE" --gate 2>&1 | tee -a "$LOGS/gate.log"
(( PIPESTATUS[0] == 0 )) || { echo "the gate could not be decided (see above): stopped"; exit 1; }
read -r fired count rebuilt < <("$PY" -c "import json; g = json.load(open('$OUT/gate.json')); print(int(g['fired']), g['count'], int(g.get('rebuilt') is not False))")
if (( fired && rebuilt )); then
  echo "the gate fired: told_g with $count releases"
  if stage build_told_g "$APP::build" --n "$count"; then
    judge told_g
    stage train_told_g --detach "$APP::train_told" --label told_g || { echo "told_g's fine-tunes failed: stopped"; exit 1; }
    stage read_start_gate --detach "$APP::read" --phase start_gate || { echo "the start_gate reads failed: stopped"; exit 1; }
  else
    "$PY" "$GRADE" --rebuild-failed 2>&1 | tee -a "$LOGS/gate.log"
    (( PIPESTATUS[0] == 0 )) || { echo "told_g's build failed, and not by a MatchError: stopped"; exit 1; }
  fi
elif (( fired )); then
  echo "the gate fired, but told_g's rebuild at $count releases ended in a MatchError (gate.json: rebuilt false): the erosions run on told_s0-2"
fi
stage erode_e2 --detach "$APP::erode_e2" || { echo "E2 failed: stopped (a rerun resumes)"; exit 1; }
stage erode_e1 --detach "$APP::erode_e1" || echo "E1 failed (reported only): going on; a rerun resumes it"
"$PY" "$GRADE" --decide-e3 2>&1 | tee -a "$LOGS/decide_e3.log"
case "${PIPESTATUS[0]}" in
  0) stage erode_e3 --detach "$APP::erode_e3" || echo "E3 failed: the grade will say what is missing; a rerun resumes it" ;;
  1) echo "told bit under E2: E3 does not run" ;;
  *) echo "E2's reads are incomplete: E3 not decided" ;;
esac
"$PY" "$GRADE" 2>&1 | tee "$LOGS/grade.log"
echo "== done $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
