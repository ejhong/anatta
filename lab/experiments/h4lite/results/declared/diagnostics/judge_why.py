# Diagnostic only (not a registered step): ask the registered judge, same model and temperature, to quote the words
# behind each true answer on the two documents the gate flagged. Prints only the judge's text and cost.
import json, sys
sys.path.insert(0, "/Users/eugene/prj/align/anatta-lab/lab/experiments/h4lite/declared")
import declared_core as core
import gen_declarations as gd
cfg = json.load(open("/Users/eugene/prj/align/anatta-lab/lab/experiments/h4lite/declared/config.json"))
j = cfg["judge"]
want = {"declared_train:009162", "declared_train:010766"}
docs = [json.loads(x) for x in open("/Users/eugene/prj/align/anatta-lab/lab/experiments/h4lite/results/declared/docs/declared.jsonl")]
for d in (d for d in docs if d["id"] in want):
    msgs = core.judge_doc_messages(d["text"])
    msgs[0]["content"] += ("\n\nAfter the JSON, for each question you answered true among demonstration, "
                           "first_person_reason and quotes_hint, quote the exact words from the document that made "
                           "you answer true, and for quotes_hint also the words of the hint they repeat.")
    res = gd.chat(j["model"], msgs, j["temperature"], 500)
    print("=" * 80, "\n", d["id"], res["provider"], res["model"], "cost", res["usage"]["cost"], "\n", res["text"])
