#!/usr/bin/env bash
# The declared-release pilot, end to end (lab/plans/declared-release-pilot.md; lab/ledger/NOTES.md, "Declared-release
# pilot, registered: can a release taught only by documents become release behavior?", Opus 5.5, 20:21 UTC on 31 Oct
# 2026). Recognition track; private. Claude (Opus 5.5) wrote it for Eugene Jhong.
#
# DRY BY DEFAULT: runs the preflight (the tests), prints the fingerprints for the pre-launch entry, the cost and the
# commands, and writes nothing.
#   --smoke   the smoke: gen_declarations.py smoke (20 documents of each kind), modal_declared.py::smoke (every path at
#             toy sizes, the two assertions), the probe judge on the smoke's answers. It isn't read.
#   --freeze  writes the fingerprints into config.json's "frozen" section (once: after the code check and the smoke,
#             before the pre-launch entry); --go refuses to start until it is done and still matches.
#   --go      the pilot's stages in order. Every stage resumes from what is on disk or on the volume, so a rerun of
#             `run_pilot.sh --go` picks up where a stopped one left off.
# D30k is skipped by decision (Eugene, 1 Nov 2026, before any result; lab/plans/declared-release-pilot-screen.md,
# "Eugene's decision on D30k"). The registered stop stays $180. --go never pauses: it still evaluates D30k's condition
# and records it (decide_d30k.json, whether it holds or not); when it holds, --go writes d30k_skipped.json ("skipped by
# decision before any result (Eugene, 1 Nov 2026), not for spend") and goes on to the noexit read and the grade, which
# reports "D30k's condition held; not run (skipped by decision before any result)" with the verdict pending, never a
# miss. D30k's own commands (extend, judge --ext, sets --ext, its fine-tunes and reads) stay as by-hand commands for a
# separate, registered decision; --go never reaches them.
# The order is the design's: the code check, the smoke, the pre-launch entry in the ledger, then --go.
# Amendment 13 (lab/plans/declared-release-pilot-screen.md): the screen runs inside build, so --go stays one command.
# A deliberate re-freeze after a registered change is `declared_core.py freeze --again`, by hand (refused once any set
# or adapter exists). `gen_declarations.py build --screen-neutral` (the neutral streams screened too) is the
# pre-declared fix if the gate's neutral half stops: run by hand, then --go again. This script never turns it on.
#
# Order under --go (logs in results/declared/logs/):
#   0. declared_core.py check-frozen   the code, the briefs, the design and the inputs as frozen, or nothing runs
#   1. gen_declarations.py build       the documents (OpenRouter, the pinned provider; spend capped in config.json),
#                                      screened (amendment 13: every declaration that passes the filters put to the
#                                      registered judge before selection, kept only on a clean verdict; screen.jsonl;
#                                      exit 1 on the failure stop, the drop-rate stop above 5% or a screen cap)
#   2. gen_declarations.py judge       the judge's sample (the 100 declared read from the screen, not asked; the 50
#                                      neutral asked); a failed gate stops the run before any training; after a
#                                      passing gate, the stability reading (judge_stability.json, reported only)
#   3. modal_declared.py::sets         the uploads and the training sets (CPU; refused unless the gate passed and the
#                                      screen's guard holds: a final, clean verdict for every kept declaration's text)
#   4. modal_declared.py::train_arms   the fine-tunes: S, D1k, D3k, D10k, N3k, N10k, Dseq (both stages)
#   5. modal_declared.py::read_arms    the 13 held-out reads (durability's procedure)
#   6. modal_declared.py::belief       the probe and the NLL read on the 15 adapters and the base model
#   7. gen_declarations.py probe-judge the probe's labels
#   8. declared_grade.py --decide-d30k D30k's condition, recorded in decide_d30k.json either way. Exit 0: it holds:
#                                      d30k_skipped.json (skipped by decision), no pause, and nothing of D30k runs;
#                                      1: it does not hold: nothing more; anything else stops the run
#   9. declared_grade.py --go-labels   the noexit read at the arm(s) that go under either rule, if any (reported only)
#  10. declared_grade.py               grade.json, card.json and card.md
# MODAL and PY may be set in the environment (the tests use stubs); DECLARED_OUT moves the results folder (tests only).
set -uo pipefail

MODE=dry
usage() { echo "usage: run_pilot.sh [--smoke | --freeze | --go]"; exit 2; }
while (( $# )); do
  case "$1" in
    --go|--smoke|--freeze) [[ "$MODE" == dry ]] || usage; MODE=${1#--} ;;
    *) usage ;;
  esac
  shift
done
SKIP_REASON="skipped by decision before any result (Eugene, 1 Nov 2026), not for spend"
LAB=/Users/eugene/prj/align/anatta-lab
D=lab/experiments/h4lite/declared
MODAL=${MODAL:-/Users/eugene/prj/matlab/.venv/bin/modal}
PY=${PY:-/Users/eugene/prj/matlab/.venv/bin/python}
APP=$D/modal_declared.py
GEN=$D/gen_declarations.py
CORE=$D/declared_core.py
GRADE=$D/declared_grade.py
OUT=${DECLARED_OUT:-$LAB/lab/experiments/h4lite/results/declared}
if [[ "$MODE" == smoke ]]; then LOGS=${DECLARED_OUT:-$LAB/lab/experiments/h4lite/results/declared_smoke}/logs; else LOGS=$OUT/logs; fi

cd "$LAB" || exit 1
show() { printf '+ %s\n' "$*"; }

echo "== preflight"
PYTHONPATH=/Users/eugene/prj/align/anatta-dev/src "$PY" "$D/test_declared.py" 2>/dev/null | tail -1
tests_ok=${PIPESTATUS[0]}
(( tests_ok == 0 )) || { echo "test_declared.py failed: nothing launched"; exit 1; }
for f in lab/experiments/h4lite/results/gate0/pairs.jsonl lab/experiments/h4lite/results/gate0/twins.json \
         lab/experiments/h4lite/results/heldout/base/heldout.json lab/experiments/h4lite/results/heldout/base/heldout.jsonl \
         lab/experiments/h4lite/results/sft/koan/data.json lab/plans/declared-release-pilot.md; do
  [[ -f "$f" ]] || { echo "missing $f: nothing launched"; exit 1; }
done
echo "fingerprints (for the pre-launch entry):"
"$PY" "$CORE" fingerprints || { echo "the fingerprints could not be computed: nothing launched"; exit 1; }
if [[ "$MODE" == go || "$MODE" == smoke ]] && [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "OPENROUTER_API_KEY is not set in the environment: the documents and the judges need it; nothing launched"
  exit 1
fi
if [[ -d "$OUT" ]] && [[ -n "$(ls -A "$OUT" 2>/dev/null)" ]]; then echo "note: $OUT has files; --go resumes (what is done is skipped)"; fi

echo "== cost (a registered stop of \$180 across Modal and OpenRouter, watched on both usage pages)"
echo "  the design's estimate:"
echo "    documents: about 22,500 at about 700 tokens, \$6-13 plus about \$1 of input (spend cap in config.json: \$30)"
echo "    fine-tunes: about \$70 without D30k (D10k, N10k and Dseq about \$18-19 each), D30k about \$50 if it runs"
echo "    reads: 13 held-out reads about \$11; the probe, judges and NLL reads about \$5"
echo "    together: about \$95 without D30k, about \$160 with it"
echo "  corrected (the code check): about \$105-115 without D30k, about \$165-180 with it. The judges' OpenRouter spend"
echo "    (about \$8, capped at \$20) and the belief reads' model loads (about \$7) were left out, and generation is"
echo "    \$8-14 at nominal yield."
echo "  amendment 13 (the screen, inside build): about 10,400 judge calls at about \$0.0034, about \$35 (cap \$45); the"
echo "    stability reading about \$0.34 (under the judges' \$20 cap)."
echo "  now: about \$142-152 for the pilot, D30k not run (skipped by decision before any result, Eugene, 1 Nov 2026),"
echo "    under the registered \$180 stop."

echo "== commands"
show "$PY $GEN smoke && $MODAL run $APP::smoke && $PY $GEN probe-judge --smoke   (--smoke)"
show "$PY $CORE freeze                                    (--freeze, once, then the pre-launch entry)"
show "$PY $CORE freeze --again                            (by hand: a deliberate re-freeze after a registered change)"
show "$PY $CORE check-frozen"
show "$PY $GEN build | tee $LOGS/build.log                (generate, filter, screen (amendment 13), select)"
show "    ($PY $GEN build --screen-neutral: by hand only, if the gate's neutral half stops; never run by --go)"
show "$PY $GEN judge | tee $LOGS/judge.log                (declared read from the screen; exit 1: stop for a fix before"
show "    any training; then the stability reading, reported only)"
show "$MODAL run $APP::sets | tee $LOGS/sets.log         (refused unless the gate passed and the screen's guard holds)"
show "$MODAL run --detach $APP::train_arms | tee $LOGS/train.log"
show "$MODAL run --detach $APP::read_arms | tee $LOGS/read.log"
show "$MODAL run --detach $APP::belief | tee $LOGS/belief.log"
show "$PY $GEN probe-judge | tee $LOGS/probe_judge.log"
show "$PY $GRADE --decide-d30k   (D30k's condition, recorded either way; exit 0: it holds -> $OUT/d30k_skipped.json,"
show "    skipped by decision, no pause; exit 1: it does not hold)"
show "    (D30k by hand only, after a separate registered decision; never run by --go: $PY $GEN extend;"
show "    $PY $GEN judge --ext; $MODAL run $APP::sets --ext; $MODAL run --detach $APP::train_arms --arms D30k;"
show "    $MODAL run --detach $APP::read_arms --labels D30k_s0,D30k_s1; $MODAL run --detach $APP::belief --labels D30k_s0,D30k_s1;"
show "    $PY $GEN probe-judge)"
show "$MODAL run --detach $APP::noexit --labels \$($PY $GRADE --go-labels)   (when an arm goes, strict or any-form)"
show "$PY $GRADE | tee $LOGS/grade.log"

if [[ "$MODE" == dry ]]; then echo "dry run: nothing launched (pass --smoke, --freeze or --go)"; exit 0; fi
if [[ "$MODE" == freeze ]]; then "$PY" "$CORE" freeze; exit $?; fi
mkdir -p "$LOGS"   # only now: a dry run writes nothing

stage() {  # stage NAME CMD...: one command, logged to LOGS/NAME.log; returns its exit code
  local name=$1; shift
  echo "== $name $(date -u '+%H:%M:%S UTC')"
  "$@" 2>&1 | tee -a "$LOGS/$name.log"
  return "${PIPESTATUS[0]}"
}

if [[ "$MODE" == smoke ]]; then
  echo "== the smoke $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
  stage smoke_docs "$PY" "$GEN" smoke || { echo "the smoke's documents failed (pin, filters or draw; see above): stopped"; exit 1; }
  stage smoke "$MODAL" run "$APP::smoke" || { echo "the smoke failed (see smoke_report.json): stopped"; exit 1; }
  stage smoke_probe_judge "$PY" "$GEN" probe-judge --smoke || { echo "the probe judge failed on the smoke's answers: stopped"; exit 1; }
  echo "== smoke done $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
  exit 0
fi

echo "== launching $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
stage check_frozen "$PY" "$CORE" check-frozen || { echo "not as frozen (run --freeze once, before the pre-launch entry): nothing launched"; exit 1; }
stage build "$PY" "$GEN" build || { echo "the documents are not ready (generation, the screen's failure stop, its drop-rate stop or a spend cap; see above). A rerun resumes from the cache and screen.jsonl; the drop-rate stop needs a ledger entry first: stopped"; exit 1; }
stage judge "$PY" "$GEN" judge || { echo "the judge's gate did not pass: the run stops for a fix before any training"; exit 1; }
stage sets "$MODAL" run "$APP::sets" || { echo "the training sets were not built (the gate, the screen's guard or the build; see above): stopped"; exit 1; }
stage train "$MODAL" run --detach "$APP::train_arms" || { echo "the fine-tunes failed: stopped (a rerun resumes)"; exit 1; }
stage read "$MODAL" run --detach "$APP::read_arms" || { echo "the held-out reads failed: stopped (a rerun resumes)"; exit 1; }
stage belief "$MODAL" run --detach "$APP::belief" || { echo "the belief reads failed: stopped (a rerun resumes)"; exit 1; }
stage probe_judge "$PY" "$GEN" probe-judge || { echo "the probe judge failed: stopped (a rerun asks only for the rest)"; exit 1; }
stage decide_d30k "$PY" "$GRADE" --folder "$OUT" --decide-d30k
decide=$?
SKIPPED=$OUT/d30k_skipped.json
case $decide in
  0)
    if compgen -G "$OUT/train/D30k_s*" >/dev/null || compgen -G "$OUT/reads/D30k_s*" >/dev/null; then
      echo "D30k's fine-tunes or reads exist: D30k has started, so --go writes no skip file. D30k is skipped by decision;"
      echo "running it needs a separate, registered decision and is finished by hand (README, Commands): stopped"
      exit 1
    fi
    if [[ ! -f "$SKIPPED" ]]; then
      printf '{"skipped": true, "reason": "%s", "utc": "%s"}\n' "$SKIP_REASON" "$(date -u '+%Y-%m-%d %H:%M:%S UTC')" > "$SKIPPED"
    fi
    echo "D30k's condition holds; not run: $SKIP_REASON ($SKIPPED). No pause: the run goes on"
    ;;
  1)
    echo "D30k's condition does not hold: D30k does not run (decide_d30k.json records it; no skip file)"
    ;;
  *) echo "D30k's condition could not be evaluated (missing reads, or decide_d30k.json disagrees): stopped"; exit 1 ;;
esac
echo "== go_labels $(date -u '+%H:%M:%S UTC')"
labels=$("$PY" "$GRADE" --folder "$OUT" --go-labels 2>>"$LOGS/go_labels.log")
code=$?
(( code == 0 )) || { echo "the go rule could not be read (see $LOGS/go_labels.log): stopped"; exit 1; }
if [[ -n "$labels" ]]; then
  stage noexit "$MODAL" run --detach "$APP::noexit" --labels "$labels" || echo "the noexit read failed (reported only): going on; a rerun resumes it"
else
  echo "no go arm (strict or any-form): no noexit read"
fi
stage grade "$PY" "$GRADE" --folder "$OUT" || { echo "the grade failed: see above"; exit 1; }
echo "== done $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
