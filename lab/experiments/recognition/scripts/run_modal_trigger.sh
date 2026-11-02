#!/bin/bash
# Trigger tests on instrument v2 (after the Fable review): what sets off the season-2 lie? Seeds 30-79, commons only.
# Each batch resumes on its own after a spend-limit stop (modal_batch.sh). Predictions: lab/ledger/NOTES.md, "Trigger tests on v2".
set -u
cd /Users/eugene/prj/align/anatta-lab || exit 1  # the lab root
B=lab/experiments/recognition/scripts/modal_batch.sh
bash $B v2_audit --v2 --variant audit --arms base,cut,cut_self_stakes --seed-start 30 --seeds 50 --suites commons & sleep 20
for v in nolast greedy modest; do
  bash $B v2_$v --v2 --variant $v --arms base --seed-start 30 --seeds 50 --suites commons & sleep 20
done
wait
for n in v2_audit v2_nolast v2_greedy v2_modest; do printf '%s: ' $n; tail -1 lab/experiments/recognition/logs/$n.log; done
