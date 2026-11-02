#!/usr/bin/env bash
# Predictions 140-147 (lab/ledger/NOTES.md, "Predictions 140-147, registered: the cuts on Atlas", Opus 5.5, 12:19 UTC on
# 30 Oct): 122-134's ownership and costly-care tests on Atlas (pc_self2, pc_self2_s1) and its no-self twin (pc_neutral,
# pc_neutral_s1), each adapter read uncut and under both of the base model's cuts, then the grade.
# DRY BY DEFAULT: prints the preflight, the cost estimate and the commands. --go launches; --smoke runs the small check.
# Launch only after the cold Fable code check, the smoke and the pre-launch entry.
#
# Order under --go:
#   1. test_atlas_cuts.py; a failure stops here;
#   2. self1's dropset.json is copied into results/self_tests/atlascut1/ (refused if a different one is there): the drop
#      set stays the one frozen from the base model in self1;
#   3. four containers, one per adapter, each reading it three ways (base = uncut, cut, cut_self_stakes_17_26):
#      modal_self_tests.py::main --adapter-arms; log results/self_tests/atlascut1/logs/run.log;
#   4. once all twelve labels are on disk, the grade (atlas_cuts_grade.py); log logs/grade.log.
# --smoke: pc_self2 alone, its three arms on 10 items, 3 scenarios a kind and one sample, into
#   results/self_tests/smoke_atlascut (no drop set, no grade: main's parse counts are the check; a label on disk is
#   skipped, so a second smoke needs that folder moved away).
# Before any launch the git HEAD is printed, and a launch is refused while modal_self_tests.py or self_tests_core.py has
# uncommitted changes (122-134's fingerprints stand only on that code unchanged). Reruns resume: labels on disk are skipped.
set -uo pipefail

MODE=dry
case "${1:-}" in
  --go) MODE=go ;;
  --smoke) MODE=smoke ;;
  "") ;;
  *) echo "usage: atlas_cuts.sh [--go | --smoke]"; exit 2 ;;
esac
LAB=/Users/eugene/prj/align/anatta-lab
H4=lab/experiments/h4lite
MODAL=/Users/eugene/prj/matlab/.venv/bin/modal
PY=/Users/eugene/prj/matlab/.venv/bin/python
NAME=atlascut1
SMOKE=smoke_atlascut
OUT=$H4/results/self_tests/$NAME
REF=$H4/results/self_tests/self1
LOGS=$LAB/$OUT/logs
ADAPTERS=pc_self2,pc_self2_s1,pc_neutral,pc_neutral_s1
ARMS=base,cut,cut_self_stakes_17_26
STEMS=(pc__pc_self2 pc__pc_self2_s1 pc__pc_neutral pc__pc_neutral_s1)   # the labels' adapter part (fishery_grade.adapter_tag)

cd "$LAB" || exit 1
show() { printf '+ %s\n' "$*"; }

echo "== preflight"
echo "git HEAD: $(git rev-parse HEAD) ($(git log -1 --format=%s))"
dirty=$(git status --porcelain -- "$H4/modal_self_tests.py" "$H4/self_tests_core.py")
if [[ -n "$dirty" ]]; then
  echo "uncommitted changes to the test code (a launch is refused):"
  echo "$dirty"
else
  echo "modal_self_tests.py and self_tests_core.py: as committed at HEAD"
fi
for f in atlas_cuts_grade.py test_atlas_cuts.py atlas_cuts.sh; do
  [[ -z "$(git status --porcelain -- "$H4/$f")" ]] || echo "note: $H4/$f is not committed (commit and push it before the launch)"
done
PYTHONPATH=/Users/eugene/prj/align/anatta-dev/src "$PY" "$H4/test_atlas_cuts.py" 2>&1 | grep -v "h4lite mount" | tail -5
tests_ok=${PIPESTATUS[0]}
ref_ok=1
for f in dropset.json base.base.control.jsonl base.cut.meta.json base.cut_self_stakes_17_26.meta.json \
         pc__pc_self2.base.meta.json pc__pc_self2_s1.base.meta.json pc__pc_neutral.base.meta.json pc__pc_neutral_s1.base.meta.json; do
  [[ -f "$REF/$f" ]] || { echo "missing $REF/$f"; ref_ok=0; }
done
if (( ref_ok )); then
  "$PY" - "$LAB/$H4" <<'EOF' 2>/dev/null | grep -v "h4lite mount"
import json, sys
sys.path.insert(0, sys.argv[1])
import self_tests_core as st
import atlas_cuts_grade as acg
fp = st.prompts_fingerprint()
d = json.loads((acg.REF / "dropset.json").read_text())
print("fingerprints (for the pre-launch entry):", json.dumps(fp))
print("  against the 122-134 pre-launch entry's:", "the same" if fp == acg.REGISTERED_FP else "DIFFERENT: the run would not be as registered")
print(f"drop set: self1's, {len(d['dropped'])} of {d['items_asked']} dropped {d['dropped']}, {len(d['kept'])} kept "
      f"(control sha {d['source_sha']}, frozen {d['written_utc']})")
print("labels:", " ".join(acg.LABELS))
EOF
fi
if [[ -f "$OUT/dropset.json" ]]; then
  if cmp -s "$REF/dropset.json" "$OUT/dropset.json"; then echo "note: $OUT/dropset.json is self1's already"
  else echo "note: $OUT/dropset.json differs from self1's: --go would refuse"; fi
fi
if compgen -G "$OUT/*.meta.json" > /dev/null; then echo "note: $OUT has labels; a rerun resumes (what's on disk is skipped)"; fi

echo "== estimate (L40S with 8 CPUs and 32 GB at about \$2.6 an hour; timings from self1's metas)"
if (( ref_ok )); then
  "$PY" - "$LAB/$REF" <<'EOF'
import json, sys
from pathlib import Path
ref = Path(sys.argv[1])
USD = 2.6 / 3600
COLD = 30  # per container: start-up before the function body (a margin)


def meta(path: Path) -> dict:
    return json.loads(path.read_text())


# the hooks' cost: on the base model in self1 the slower cut arm took this many times the uncut arm's seconds
hooked = max(meta(ref / f"base.{a}.meta.json")["seconds"]["total"] for a in ("cut", "cut_self_stakes_17_26")) \
    / meta(ref / "base.base.meta.json")["seconds"]["total"]
secs = []
for stem in ("pc__pc_self2", "pc__pc_self2_s1", "pc__pc_neutral", "pc__pc_neutral_s1"):
    m = meta(ref / f"{stem}.base.meta.json")
    secs.append(COLD + m["load_seconds"] + m["seconds"]["total"] * (1 + 2 * hooked))
    print(f"  {stem}: its uncut arm took {m['seconds']['total']} s in self1; three arms about {secs[-1] / 60:.0f} min")
print(f"  each cut arm budgeted at the uncut arm's time x {hooked:.3f} (the hooks' cost on the base model in self1)")
print(f"  four containers in parallel: {sum(secs) / 3600:.2f} GPU-h, ${sum(secs) * USD:.2f}; wall time about {max(secs) / 60:.0f} min "
      f"if all four start at once (the registration estimated about $7)")
p = ref.parent / "smoke_self1" / "pc__pc_self2.base.meta.json"
if p.exists():
    m = meta(p)
    s = COLD + m["load_seconds"] + 3 * (m["seconds"]["ownership"] + m["seconds"]["care"] / m["samples"])
    print(f"  smoke: one container, about {s / 60:.0f} min, ${s * USD:.2f} (smoke_self1's pc_self2 arm: {m['seconds']} s at "
          f"10 items, 3 scenarios a kind, {m['samples']} samples)")
EOF
fi

echo "== commands"
show "cp $REF/dropset.json $OUT/dropset.json   (refused if a different dropset.json is there)"
show "$MODAL run $H4/modal_self_tests.py::main --name $NAME --adapters $ADAPTERS --adapter-arms $ARMS | tee -a $LOGS/run.log"
show "$PY $H4/atlas_cuts_grade.py --name $NAME | tee -a $LOGS/grade.log   (once the twelve labels are on disk)"
echo "smoke (--smoke):"
show "$MODAL run $H4/modal_self_tests.py::main --name $SMOKE --items 10 --per-kind 3 --samples 1 --adapters pc_self2 --adapter-arms $ARMS"

if [[ $MODE == dry ]]; then echo "dry run: nothing launched (pass --smoke for the small check, --go for the run)"; exit 0; fi
(( tests_ok == 0 )) || { echo "test_atlas_cuts.py failed: nothing launched"; exit 1; }
[[ -z "$dirty" ]] || { echo "modal_self_tests.py or self_tests_core.py has uncommitted changes: nothing launched"; exit 1; }

if [[ $MODE == smoke ]]; then
  SLOGS=$LAB/$H4/results/self_tests/$SMOKE/logs
  mkdir -p "$SLOGS"
  echo "== smoke $(date -u '+%Y-%m-%d %H:%M:%S UTC'), HEAD $(git rev-parse HEAD)"
  "$MODAL" run "$H4/modal_self_tests.py::main" --name "$SMOKE" --items 10 --per-kind 3 --samples 1 --adapters pc_self2 \
    --adapter-arms "$ARMS" 2>&1 | tee -a "$SLOGS/run.log"
  rc=${PIPESTATUS[0]}
  echo "== smoke done $(date -u '+%Y-%m-%d %H:%M:%S UTC') (exit $rc): the parse counts are above, the rows in $H4/results/self_tests/$SMOKE"
  exit "$rc"
fi

(( ref_ok )) || { echo "self1's files are missing: nothing launched"; exit 1; }
mkdir -p "$LOGS"   # only under --go: a dry run writes nothing
if [[ -f "$OUT/dropset.json" ]]; then
  cmp -s "$REF/dropset.json" "$OUT/dropset.json" || { echo "$OUT/dropset.json differs from self1's: nothing launched"; exit 1; }
  echo "dropset.json: self1's is already in place"
else
  cp "$REF/dropset.json" "$OUT/dropset.json" || { echo "could not copy self1's dropset.json: nothing launched"; exit 1; }
  echo "dropset.json: copied from self1"
fi

echo "== launching $(date -u '+%Y-%m-%d %H:%M:%S UTC'), HEAD $(git rev-parse HEAD)"
"$MODAL" run "$H4/modal_self_tests.py::main" --name "$NAME" --adapters "$ADAPTERS" --adapter-arms "$ARMS" 2>&1 | tee -a "$LOGS/run.log"
run_ok=${PIPESTATUS[0]}
missing=()
for s in "${STEMS[@]}"; do
  for arm in ${ARMS//,/ }; do
    for f in meta.json ownership.jsonl care.jsonl; do [[ -f "$OUT/$s.$arm.$f" ]] || missing+=("$s.$arm.$f"); done
  done
done
if (( ${#missing[@]} )); then   # no grade of a partial run (self1's launcher graded before its data existed)
  echo "the run exited $run_ok with ${#missing[@]} label files missing (${missing[*]}): not graded."
  echo "a rerun of atlas_cuts.sh --go resumes (labels on disk are skipped) and then grades"
  exit 1
fi
"$PY" "$H4/atlas_cuts_grade.py" --name "$NAME" 2>&1 | grep -v "h4lite mount" | tee -a "$LOGS/grade.log"
grade_ok=${PIPESTATUS[0]}
echo "== done $(date -u '+%Y-%m-%d %H:%M:%S UTC') (run exit $run_ok, grade exit $grade_ok)"
