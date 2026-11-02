"""K1's features: WIT-1024 read from its two parquet files, DINOv2's per-block CLS tokens, and each language model's
masked-mean hidden states.

The feature code mirrors Balestriero & LeCun's prh/features.py (galilai-group/prh-proof at
a2b9949fa907d4fbe562742d4b1ffa7205ab7427, MIT, Copyright (c) 2026 Randall Balestriero and Yann LeCun; the licence is in
third_party/prh_proof/LICENSE): _image_net and embed_images with layers=True, _text_net, embed_texts with layers=True
and _mean. Where it departs, the comment says so; README.md lists every departure. The rows are read as the paper's
prh/data.py wit1024 reads them through `datasets` (the "train" split of minhuh/prh at wit_1024, whose card's
data_files are data/train-*: the two parquet files in name order), without running `datasets` itself, and each image
is decoded as `datasets`' Image feature decodes it (PIL.Image.open on the bytes, load, exif_transpose when the EXIF
has an Orientation tag; datasets 2.20 to 5.1 agree), then converted to RGB as embed_images does.

torch, transformers, timm, PIL, pyarrow and sentencepiece are imported inside the functions that need them.
"""

from __future__ import annotations

import gc
import re
import time
from pathlib import Path

# the paper's rules on its model names (prh/features.py _text_net), applied to K1's keys (qwen0.5b ... qwen14b,
# open_llama_3b): bfloat16 for a name ending in a size in billions other than 0.5b and 1.5b, float32 otherwise; 14B
# and up split across the visible GPUs (device_map="auto")
BF16_NAME = re.compile(r"\d+b$")
BIG_NAME = re.compile(r"(1[4-9]|[2-9]\d)b$")


def text_dtype(key: str) -> str:
    return "bfloat16" if BF16_NAME.search(key) and not key.endswith(("0.5b", "1.5b")) else "float32"


def device_map_auto(key: str) -> bool:
    return BIG_NAME.search(key) is not None


def device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _batches(xs, size):
    return (xs[i : i + size] for i in range(0, len(xs), size))


# ------------------------------------------------------------------------------------------------ the rows


def read_wit(files: list, rows: int | None = None, images: bool = True) -> tuple[list, list[str]]:
    """The first `rows` pairs (all when None) of the parquet files, in the order given: each image's encoded bytes
    (or None where images is False) and its first caption, text[0]. Refuses files without the image and text columns
    (CheckError schema), a row with no caption (caption) and a row whose image has no bytes (image_bytes)."""
    import pyarrow.parquet as pq
    from k1 import CheckError

    caps, imgs = [], []
    for f in files:
        names = pq.read_schema(f).names
        if "text" not in names or (images and "image" not in names):
            raise CheckError("schema")
        t = pq.read_table(f, columns=["text", "image"] if images else ["text"])
        for i, text in enumerate(t.column("text").to_pylist()):
            if not text or not isinstance(text[0], str):
                raise CheckError("caption", row=len(caps))
            caps.append(text[0])
            if rows is not None and len(caps) >= rows:
                break
        if images:
            col = t.column("image").slice(0, len(caps) - len(imgs))
            for cell in col.to_pylist():
                if not cell or not cell.get("bytes"):
                    raise CheckError("image_bytes", row=len(imgs))
                imgs.append(cell["bytes"])
        if rows is not None and len(caps) >= rows:
            break
    return (imgs if images else [None] * len(caps)), caps


def decode_image(data: bytes):
    """An image's bytes as `datasets`' Image feature decodes them (datasets 2.20 to 5.1, features/image.py
    decode_example): PIL.Image.open, load, exif_transpose when the EXIF carries an Orientation tag."""
    from io import BytesIO

    import PIL.Image
    import PIL.ImageOps

    image = PIL.Image.open(BytesIO(data))
    image.load()
    if image.getexif().get(PIL.Image.ExifTags.Base.Orientation) is not None:
        image = PIL.ImageOps.exif_transpose(image)
    return image


UNSAFE = re.compile(r"[^A-Za-z0-9_.\-]")


def safe_key(name) -> str:
    """A name from a decoder as a result may carry it: every character but letters, digits, _ . and - as _ (PIL's
    16-bit mode I;16 becomes I_16), at most 60 characters."""
    return UNSAFE.sub("_", str(name))[:60]


def image_facts(images: list[bytes]) -> dict:
    """Counts over the decoded images: formats and modes (as safe_key gives them), how many an EXIF orientation
    turned, and sizes."""
    from io import BytesIO

    import PIL.Image

    fmt, mode, turned, px = {}, {}, 0, []
    for data in images:
        raw = PIL.Image.open(BytesIO(data))
        f, m = safe_key(raw.format), safe_key(raw.mode)
        fmt[f] = fmt.get(f, 0) + 1
        mode[m] = mode.get(m, 0) + 1
        turned += raw.getexif().get(PIL.Image.ExifTags.Base.Orientation) is not None
        px.append(raw.size[0] * raw.size[1])
    px.sort()
    return {"images": len(images), "formats": fmt, "modes": mode, "exif_orientation": turned,
            "pixels": {"min": px[0], "median": px[len(px) // 2], "max": px[-1]} if px else None}


# ------------------------------------------------------------------------------------------------ images


def vision_net(timm_name: str, weights):
    """(net, transform, data config) as the paper's _image_net builds them with layers=True:
    timm.create_model(name, pretrained=True, num_classes=0) at its native configuration, and timm's transform from
    resolve_data_config(net.pretrained_cfg, model=net). Departure: the weights come from the file prepare fetched to
    the volume (pretrained_cfg_overlay file=..., timm's local-file source, with the same filter_fn and strict load),
    not from the hub at run time."""
    import timm

    net = timm.create_model(timm_name, pretrained=True, num_classes=0, pretrained_cfg_overlay={"file": str(weights)})
    cfg = timm.data.resolve_data_config(net.pretrained_cfg, model=net)
    return net, timm.data.create_transform(**cfg), cfg


def data_config_facts(cfg: dict) -> dict:
    """The resolved data config as plain JSON (input size, interpolation, crop, mean and std)."""
    out = {}
    for k, v in cfg.items():
        out[k] = list(v) if isinstance(v, (tuple, list)) else v
    return out


def embed_images(net, tf, images, batch_size: int = 64, decode=decode_image) -> list:
    """The CLS token after each block, [N, d] float32 per block (the paper's embed_images with layers=True: float32
    weights and inputs, a forward hook on each block taking pool(out) = out[:, 0] for a model with a prefix token,
    cast to float32; per-block features stitched from the hooks' order). Images are given as encoded bytes and decoded
    one batch at a time (decode, then .convert("RGB") as the paper does), where the paper's `datasets` column decodes
    them all first: the same pixels."""
    import torch

    dev, dtype = device(), torch.float32
    net = net.eval().to(dev, dtype)
    pool = (lambda o: o[:, 0]) if getattr(net, "num_prefix_tokens", 1) else (lambda o: o.mean(1))
    feats = []
    hooks = [blk.register_forward_hook(lambda m, i, o: feats.append(pool(o).float().cpu())) for blk in net.blocks]
    try:
        with torch.no_grad():
            for chunk in _batches(list(images), batch_size):
                x = torch.stack([tf(decode(im).convert("RGB")) for im in chunk])
                x = x.to(dev, dtype)
                net(x)
    finally:
        for h in hooks:
            h.remove()
    n = len(net.blocks)
    return [torch.cat(feats[i::n]) for i in range(n)]


# ------------------------------------------------------------------------------------------------ texts


def check_bin(path) -> int:
    """A pickle checkpoint (OpenLLaMA's pytorch_model.bin) opened with torch.load(weights_only=True), memory-mapped:
    refused (CheckError pickle) unless it loads that way and is a flat dict of tensors. Returns its tensor count."""
    import torch
    from k1 import CheckError

    try:
        sd = torch.load(str(path), map_location="cpu", weights_only=True, mmap=True)
    except Exception:  # noqa: BLE001 (any refusal of the safe unpickler is a refusal)
        raise CheckError("pickle") from None
    if not isinstance(sd, dict) or not sd or not all(isinstance(v, torch.Tensor) for v in sd.values()):
        raise CheckError("pickle")
    n = len(sd)
    del sd
    return n


def load_tokenizer(snap) -> tuple:
    """The tokenizer at its own defaults, with eos as pad if it has none (the paper's
    tok.pad_token = tok.pad_token or tok.eos_token), and what was in effect."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(snap))
    had = tok.pad_token
    tok.pad_token = tok.pad_token or tok.eos_token
    facts = {"class": type(tok).__name__, "padding_side": tok.padding_side, "pad_token_id": tok.pad_token_id,
             "eos_token_id": tok.eos_token_id, "bos_token_id": tok.bos_token_id, "pad_was_set": had is not None,
             "pad_is_eos": tok.pad_token_id == tok.eos_token_id, "pad_token_hex": str(tok.pad_token).encode().hex(),
             "legacy": getattr(tok, "legacy", None) if isinstance(getattr(tok, "legacy", None), bool) else None,
             "vocab": len(tok)}
    return tok, facts


def text_net(key: str, snap, weights: str = "safetensors"):
    """(net, tok, facts) as the paper's _text_net loads a pretrained model: AutoModel (no LM head) in its dtype rule's
    dtype (text_dtype), device_map="auto" for 14B and up (else moved to the GPU), eval mode; the tokenizer at its
    defaults with eos as pad if none (load_tokenizer). K1 adds safety, not arithmetic: weights_only=True always, the
    Qwen models from safetensors only, and the .bin of OpenLLaMA opened first with weights_only=True (check_bin);
    with device_map="auto", every module must sit on the one GPU (CheckError device_map)."""
    import torch
    from k1 import CheckError
    from transformers import AutoConfig, AutoModel

    dtype = getattr(torch, text_dtype(key))
    big = device_map_auto(key)
    tok, tfacts = load_tokenizer(snap)
    bin_tensors = check_bin(Path(snap) / "pytorch_model.bin") if weights == "bin" else None
    cfg = AutoConfig.from_pretrained(str(snap))
    stored = getattr(cfg, "dtype", None) or getattr(cfg, "torch_dtype", None)
    t0 = time.time()
    net = AutoModel.from_pretrained(str(snap), dtype=dtype, weights_only=True, use_safetensors=weights != "bin",
                                    **({"device_map": "auto"} if big else {})).eval()
    net = net if big else net.to(device())
    placed = sorted({str(v) for v in (getattr(net, "hf_device_map", None) or {}).values()})
    if big and any(p not in ("0", "cuda", "cuda:0") for p in placed):
        raise CheckError("device_map")
    facts = {"tokenizer": tfacts, "dtype": text_dtype(key), "stored_dtype": str(stored).replace("torch.", ""),
             "device_map_auto": big, "device_map": placed or None, "attn_implementation": net.config._attn_implementation,
             "layers": net.config.num_hidden_layers, "hidden": net.config.hidden_size, "weights": weights,
             "bin_tensors": bin_tensors, "load_seconds": round(time.time() - t0, 2)}
    return net, tok, facts


def _mean(h, mask):
    m = mask.unsqueeze(-1).float()
    return (h.float() * m).sum(1) / m.sum(1)


def embed_texts(net, tok, texts: list[str], batch_size: int = 64, max_length: int = 512) -> list:
    """The masked mean of every hidden state, [N, d] float32 per state, the embedding output included (the paper's
    embed_texts with layers=True: padding=True, truncation=True, max_length=512, output_hidden_states=True)."""
    import torch

    dev, out = device(), []
    with torch.no_grad():
        for chunk in _batches(texts, batch_size):
            t = tok(chunk, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(dev)
            hs = net(**t, output_hidden_states=True).hidden_states
            out.append([_mean(h, t["attention_mask"]).float().cpu() for h in hs])
    return [torch.cat(layer) for layer in zip(*out)]


def token_facts(tok, texts: list[str], batch_size: int = 64, max_length: int = 512) -> dict:
    """The padded length of each batch of the paper's tokenization, and how many texts truncation shortens."""
    lens = [len(x) for x in tok(texts, truncation=False)["input_ids"]]
    padded = [max(min(n, max_length) for n in lens[i:i + batch_size]) for i in range(0, len(lens), batch_size)]
    s = sorted(lens)
    return {"texts": len(texts), "padded": padded, "max_padded": max(padded) if padded else 0,
            "truncated": sum(n > max_length for n in lens),
            "tokens": {"min": s[0], "median": s[len(s) // 2], "max": s[-1]} if s else None}


def sentencepiece_agreement(tok, model_file, texts: list[str]) -> dict:
    """OpenLLaMA's tokenizer as transformers builds it (converted from tokenizer.model) against sentencepiece's own
    encoding of each text, BOS first as add_bos_token says: how many texts get other ids."""
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=str(model_file))
    hf = tok(texts, truncation=False)["input_ids"]
    differ = sum(list(h) != [sp.bos_id()] + sp.encode(t) for h, t in zip(hf, texts))
    return {"texts": len(texts), "differ": differ, "sp_bos": sp.bos_id(), "sp_eos": sp.eos_id(),
            "sp_vocab": sp.get_piece_size()}


def peak_gib() -> dict | None:
    """The GPU's peak allocated and reserved memory since the last reset, in GiB (None without a GPU). CheckError
    peak_memory if a GPU is present but the reading is 0."""
    import torch
    from k1 import CheckError

    if not torch.cuda.is_available():
        return None
    a, r = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
    if a <= 0:
        raise CheckError("peak_memory")
    return {"allocated": round(a / 2**30, 3), "reserved": round(r / 2**30, 3),
            "total": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 3)}


def reset_peak() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def release() -> None:
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def predict_peak(key: str, weights_bytes: int, config: dict, batch: int, length: int,
                 total_gib: float | None = None) -> dict:
    """A rough upper prediction of a language model's peak GPU memory under the paper's batching, in GiB: its weights
    in its dtype (the Qwen files hold bfloat16, so float32 doubles them; OpenLLaMA's float16 becomes bfloat16), every
    hidden state of the longest padded batch (output_hidden_states keeps them all), one layer's MLP and attention
    scores, and the pooling's float32 copies. fits: within 92% of the GPU's memory, when that is known. The smoke
    checks the formula against the 0.5B's measured peak (memory.calibration on the card)."""
    fp32 = text_dtype(key) == "float32"
    act = 4 if fp32 else 2
    h, n_layers = config["hidden_size"], config["num_hidden_layers"]
    inter, heads = config.get("intermediate_size", 4 * h), config["num_attention_heads"]
    parts = {"weights": weights_bytes * (2 if fp32 else 1),
             "hidden_states": (n_layers + 1) * batch * length * h * act,
             "mlp": 3 * batch * length * inter * act,
             "attention": batch * heads * length * length * act,
             "pooling": 2 * batch * length * h * 4}
    gib = sum(parts.values()) / 2**30
    return {"gib": round(gib, 2), "parts_gib": {k: round(v / 2**30, 2) for k, v in parts.items()}, "length": length,
            "total_gib": total_gib, "fits": None if total_gib is None else gib <= 0.92 * total_gib}


def check_layers(layers: list, count: int, width: int, rows: int) -> None:
    """A model's features: count layers of [rows, width], all finite (CheckError shape or nonfinite)."""
    import torch
    from k1 import CheckError

    if len(layers) != count or any(tuple(f.shape) != (rows, width) for f in layers):
        raise CheckError("shape", layers=len(layers), want=count)
    if not all(bool(torch.isfinite(f).all()) for f in layers):
        raise CheckError("nonfinite")


def vision_layers(weights, timm_name: str, images: list, spec: dict, log=print) -> tuple[list, dict]:
    """DINOv2's 12 per-block CLS features on the images (bytes), timed, with the GPU's peak memory and the resolved
    data config."""
    reset_peak()
    t0 = time.time()
    net, tf, cfg = vision_net(timm_name, weights)
    load = time.time() - t0
    layers = embed_images(net, tf, images, spec["batch"])
    facts = {"data_config": data_config_facts(cfg), "blocks": len(net.blocks), "load_seconds": round(load, 2),
             "seconds": round(time.time() - t0, 2), "peak_gib": peak_gib(),
             "num_prefix_tokens": getattr(net, "num_prefix_tokens", None)}
    del net
    release()
    log(f"vision: {len(layers)} layers of {tuple(layers[0].shape)} in {facts['seconds']} s")
    return layers, facts


def text_layers(key: str, snap, texts: list[str], spec: dict, protocol: dict, log=print) -> tuple[list, dict]:
    """One language model's masked-mean features of every hidden state on the texts, timed, with the GPU's peak
    memory, the load facts and the tokenization's padded lengths."""
    reset_peak()
    t0 = time.time()
    net, tok, facts = text_net(key, snap, spec.get("weights", "safetensors"))
    layers = embed_texts(net, tok, texts, protocol["batch"], protocol["max_length"])
    facts.update(hidden_states=len(layers), seconds=round(time.time() - t0, 2), peak_gib=peak_gib(),
                 tokens=token_facts(tok, texts, protocol["batch"], protocol["max_length"]))
    del net
    release()
    log(f"{key}: {len(layers)} layers of {tuple(layers[0].shape)} in {facts['seconds']} s")
    return layers, facts
