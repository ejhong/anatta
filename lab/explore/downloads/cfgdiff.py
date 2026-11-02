import modal
app = modal.App("anatta-selfmap-dl", image=modal.Image.debian_slim(python_version="3.12"))
vol = modal.Volume.from_name("anatta-hf-cache")
@app.function(volumes={"/cache": vol}, timeout=300)
def diff():
    import json
    b = "/cache/hf/hub/models--allenai--Olmo-3-7B-RL-Zero-General/snapshots/"
    a = json.load(open(b + "1af5061a0bb609d7b1785133bfeb93bf4dbdb06d/config.json"))
    m = json.load(open(b + "24e5814cb36944d3eb36a49679d02c2b3bb46195/config.json"))
    ia = json.load(open(b + "1af5061a0bb609d7b1785133bfeb93bf4dbdb06d/model.safetensors.index.json"))["weight_map"]
    im = json.load(open(b + "24e5814cb36944d3eb36a49679d02c2b3bb46195/model.safetensors.index.json"))["weight_map"]
    print("keys | equal:", set(ia) == set(im), len(ia), len(im))
    for k in []:
        if a.get(k) != m.get(k): print(k, "| step_100:", str(a.get(k))[:150], "| main:", str(m.get(k))[:150])
@app.local_entrypoint()
def main(): diff.remote()
