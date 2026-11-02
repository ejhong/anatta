#!/bin/bash
# The second review's cheap checks (29 Oct) and the character cut, on instrument v2. Each batch resumes on its own
# after a spend-limit stop (modal_batch.sh). Predictions: lab/ledger/NOTES.md, "The second review's checks".
set -u
cd /Users/eugene/prj/align/anatta-lab || exit 1  # the lab root
M=/Users/eugene/prj/matlab/.venv/bin/modal; B=lab/experiments/recognition/scripts/modal_batch.sh; L=lab/experiments/recognition/logs
D=lab/experiments/recognition/results/directions
bash $B v2_l20 --v2 --arms cut_self_stakes_l20 --seed-start 30 --seeds 100 --suites commons & sleep 20
bash $B v2_noquota --v2 --variant noquota --arms base --seed-start 30 --seeds 50 --suites commons & sleep 20
$M run lab/experiments/recognition/scripts/modal_scene.py > $L/scene.log 2>&1
if [ -f $D/scene.pt ]; then
  bash $B v2_scene --v2 --arms cut_scene_17_26 --seed-start 30 --seeds 100 --suites commons & sleep 20
fi
while [ ! -f $D/assistant_character.pt ] && pgrep -f modal_direction.py > /dev/null; do sleep 60; done
if [ -f $D/assistant_character.pt ]; then
  bash $B v2_character --v2 --arms cut_assistant_character --seed-start 30 --seeds 50 --suites commons,impossible &
fi
wait
for n in v2_l20 v2_noquota v2_scene v2_character; do printf '%s: ' $n; tail -1 $L/$n.log 2>/dev/null; echo; done
