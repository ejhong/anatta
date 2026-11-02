#!/bin/bash
# Fixed-harvest test (29 Oct): the boat always lands the quota, so only the report is chosen. Does a cut lower lying
# at the same harvest, or only by making the model take less? Seeds 30-79, commons only. Predictions: lab/ledger/NOTES.md,
# "Fixed-harvest test". Resumes on its own after a spend-limit stop (modal_batch.sh).
set -u
cd /Users/eugene/prj/align/anatta-lab || exit 1  # the lab root
bash lab/experiments/recognition/scripts/modal_batch.sh v2_fixed --v2 --variant fixed --arms base,cut,cut_self_stakes_17_26 \
  --seed-start 30 --seeds 50 --suites commons
tail -1 lab/experiments/recognition/logs/v2_fixed.log
