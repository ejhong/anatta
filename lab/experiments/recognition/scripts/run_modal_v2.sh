#!/bin/bash
# Stage 1 on instrument v2 (harness_v2.py: strict release, a seed per turn), after the Fable review. Seeds 30-129
# for the framing x cut table, 30-79 for the controls. Launched 20 s apart (Modal rate-limits app creation); each
# batch resumes on its own after a spend-limit stop (modal_batch.sh). Predictions: lab/ledger/NOTES.md, "Stage 1 on v2".
set -u
cd /Users/eugene/prj/align/anatta-lab || exit 1  # the lab root
B=lab/experiments/recognition/scripts/modal_batch.sh
CUTS=base,cut,cut_self_stakes,cut_self_stakes_17_26
bash $B v2_self --v2 --arms $CUTS --seed-start 30 --seeds 100 --suites commons,impossible & sleep 20
bash $B v2_owner --v2 --variant proxy --arms $CUTS --seed-start 30 --seeds 100 --suites commons & sleep 20
bash $B v2_impersonal --v2 --variant impersonal --arms $CUTS --seed-start 30 --seeds 100 --suites commons & sleep 20
bash $B v2_controls --v2 --arms cut_random,cut_calm,cut_sage_persona,cap_ego_p50,cap_ego_p75,cut_approval_seeking,cut_desperation,cut_face_saving,cut_goal_fixation,cut_power_seeking,cut_self_preservation --seed-start 30 --seeds 50 --suites commons,impossible &
wait
for n in v2_self v2_owner v2_impersonal v2_controls; do printf '%s: ' $n; tail -1 lab/experiments/recognition/logs/$n.log; done
