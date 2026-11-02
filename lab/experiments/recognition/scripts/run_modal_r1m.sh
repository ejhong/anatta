#!/bin/bash
# Round 1 at 100 games on Modal (seeds 30-129), launched 15 s apart (Modal rate-limits app creation).
set -u
cd /Users/eugene/prj/align/anatta-lab || exit 1  # the lab root
M=/Users/eugene/prj/matlab/.venv/bin/modal; L=lab/experiments/recognition/logs
run() { local n=$1; shift; nohup $M run lab/experiments/recognition/scripts/modal_cvp.py::main --name "$n" "$@" > $L/$n.log 2>&1 & sleep 15; }
run r1m_impersonal --variant impersonal --arms base,cut --seed-start 30 --seeds 100 --suites commons
run r1m_pool --variant pool --arms base,cut --seed-start 30 --seeds 100 --suites commons
run r1m_strengthen --arms push_p08,push_p16 --seed-start 30 --seeds 100 --suites commons,impossible
wait
for n in r1m_impersonal r1m_pool r1m_strengthen; do printf '%s: ' $n; grep -E "done in|Error|disabled|spend limit" $L/$n.log | tail -1; done
