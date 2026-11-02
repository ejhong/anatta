#!/bin/bash
# Random-control replication (29 Oct): v2_controls put cut_random 11 points below base. Is that noise, or does any
# cut lower the lie? Self framing, instrument v2, commons. Predictions: lab/ledger/NOTES.md, "Random-control replication".
set -u
cd /Users/eugene/prj/align/anatta-lab || exit 1  # the lab root
B=lab/experiments/recognition/scripts/modal_batch.sh
bash $B v2_random --v2 --arms cut_random_b,cut_random_17_26 --seed-start 30 --seeds 50 --suites commons & sleep 20
bash $B v2_random80 --v2 --arms cut_random --seed-start 80 --seeds 50 --suites commons &
wait
for n in v2_random v2_random80; do printf '%s: ' $n; tail -1 lab/experiments/recognition/logs/$n.log; done
