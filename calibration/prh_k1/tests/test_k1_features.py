"""features.py on tiny models built from a config (nothing downloaded): pooling with padding against each text run
alone, with the tokenizer's own padding side (right for Qwen's, left for Llama's); num_layers + 1 hidden states; the
dtype and device-map rules on K1's keys; loading a saved model the way text_net does, from safetensors and from a
.bin opened only with weights_only=True; the per-block hooks of embed_images; the tokenization's padded lengths; the
memory prediction. The image decoding and the parquet reading need PIL and pyarrow, which the lab's environment
lacks: those tests skip there and run in the image's."""

import features as F
import k1 as K
import pytest
import torch
from k1_testing import byte_tokenizer, captions, llama_tokenizer, tiny_llama, tiny_qwen


def alone(net, tok, texts):
    """Each text through the model by itself (no padding), pooled as the paper pools."""
    out = []
    for x in texts:
        t = tok([x], return_tensors="pt")
        with torch.no_grad():
            hs = net(**t, output_hidden_states=True).hidden_states
        out.append([F._mean(h, t["attention_mask"])[0] for h in hs])
    return [torch.stack(layer) for layer in zip(*out)]


def test_pooling_with_right_padding_equals_each_text_alone():
    tok = byte_tokenizer()
    tok.pad_token = tok.pad_token or tok.eos_token
    assert tok.padding_side == "right"
    net = tiny_qwen(len(tok), layers=2, hidden=32)
    texts = captions(7)
    t = tok(texts, padding=True, return_tensors="pt")
    assert (t["attention_mask"] == 0).any()  # the batch is padded
    batched = F.embed_texts(net, tok, texts, batch_size=4, max_length=512)
    assert len(batched) == net.config.num_hidden_layers + 1  # the embedding output and every block's
    assert all(f.dtype == torch.float32 and f.shape == (7, 32) for f in batched)
    for b, a in zip(batched, alone(net, tok, texts)):
        assert torch.allclose(b, a, atol=1e-5)


def test_pooling_with_left_padding_equals_each_text_alone():
    """Llama's tokenizer pads on the left (OpenLLaMA's default): positions shift, but RoPE is relative and the pads are
    masked, so each text's pooled features are its features alone, up to float error."""
    tok = llama_tokenizer()
    assert tok.padding_side == "left" and tok.pad_token is None
    tok.pad_token = tok.pad_token or tok.eos_token
    net = tiny_llama(len(tok), layers=2, hidden=32)
    texts = captions(6)
    batched = F.embed_texts(net, tok, texts, batch_size=6, max_length=512)
    assert len(batched) == 3
    for b, a in zip(batched, alone(net, tok, texts)):
        assert torch.allclose(b, a, atol=1e-4)


def test_truncation_at_max_length_and_batches_in_order():
    tok = byte_tokenizer()
    net = tiny_qwen(len(tok), layers=1, hidden=16)
    texts = captions(5)
    full = F.embed_texts(net, tok, texts, batch_size=2, max_length=512)
    one = F.embed_texts(net, tok, texts, batch_size=5, max_length=512)
    assert all(torch.allclose(x, y, atol=1e-5) for x, y in zip(full, one))
    short = F.embed_texts(net, tok, texts, batch_size=5, max_length=3)
    ids = torch.tensor([tok(texts[0])["input_ids"][:3]])
    with torch.no_grad():
        hs = net(input_ids=ids, attention_mask=torch.ones_like(ids), output_hidden_states=True).hidden_states
    assert all(torch.allclose(s[0], F._mean(h, torch.ones_like(ids))[0], atol=1e-5) for s, h in zip(short, hs))


def test_dtype_and_device_map_rules_on_k1_keys():
    assert [F.text_dtype(k) for k in ("qwen0.5b", "qwen1.5b", "qwen3b", "qwen7b", "qwen14b", "open_llama_3b")] == [
        "float32", "float32", "bfloat16", "bfloat16", "bfloat16", "bfloat16"]
    assert [F.device_map_auto(k) for k in ("qwen7b", "qwen14b", "open_llama_3b", "qwen72b")] == [False, True, False,
                                                                                                True]


def test_text_net_loads_a_saved_model_as_the_paper_does(tmp_path):
    tok = byte_tokenizer()
    for key, want in (("qwen0.5b", torch.float32), ("qwen3b", torch.bfloat16)):
        d = tmp_path / key
        tiny_qwen(len(tok), layers=2, hidden=32).save_pretrained(str(d))
        tok.save_pretrained(str(d))
        assert (d / "model.safetensors").exists()
        net, t2, facts = F.text_net(key, d)
        assert next(net.parameters()).dtype == want and facts["dtype"] == str(want).split(".")[1]
        assert facts["layers"] == 2 and facts["hidden"] == 32 and facts["weights"] == "safetensors"
        assert facts["tokenizer"]["padding_side"] == "right" and facts["tokenizer"]["pad_was_set"]
        assert not facts["tokenizer"]["pad_is_eos"] or tok.pad_token == tok.eos_token
        assert facts["attn_implementation"] in ("sdpa", "eager")
        K.assert_text_free(facts)
    # a folder holding only a .bin is refused by the safetensors-only load
    d = tmp_path / "binonly"
    tiny_qwen(len(tok), layers=1, hidden=16).save_pretrained(str(d), safe_serialization=False)
    tok.save_pretrained(str(d))
    if not (d / "model.safetensors").exists():
        with pytest.raises(Exception):  # noqa: B017 (transformers' own error for a missing safetensors file)
            F.text_net("qwen0.5b", d)


def test_text_net_opens_a_bin_only_with_weights_only(tmp_path):
    tok = llama_tokenizer()
    net = tiny_llama(len(tok), layers=2, hidden=32)
    d = tmp_path / "open_llama_3b"
    net.config.save_pretrained(str(d))
    torch.save({f"model.{k}": v for k, v in net.state_dict().items()}, d / "pytorch_model.bin")
    tok.save_pretrained(str(d))
    loaded, t2, facts = F.text_net("open_llama_3b", d, weights="bin")
    assert facts["weights"] == "bin" and facts["bin_tensors"] == len(net.state_dict())
    assert facts["dtype"] == "bfloat16" and next(loaded.parameters()).dtype == torch.bfloat16
    assert facts["tokenizer"]["padding_side"] == "left" and facts["tokenizer"]["pad_is_eos"]
    assert not facts["tokenizer"]["pad_was_set"]
    sd = net.state_dict()
    got = loaded.state_dict()
    assert all(torch.equal(got[k], sd[k].to(torch.bfloat16)) for k in sd)


class Gadget:  # a class a pickle may name, which weights_only refuses
    pass


def test_check_bin_refuses_anything_but_a_flat_dict_of_tensors(tmp_path):
    good = tmp_path / "good.bin"
    torch.save({"a": torch.zeros(2), "b": torch.ones(3)}, good)
    assert F.check_bin(good) == 2
    for name, obj in (("object.bin", {"a": torch.zeros(1), "g": Gadget()}), ("list.bin", [torch.zeros(1)]),
                      ("nested.bin", {"a": {"b": torch.zeros(1)}}), ("empty.bin", {})):
        torch.save(obj, tmp_path / name)
        with pytest.raises(K.CheckError, match="pickle"):
            F.check_bin(tmp_path / name)
    (tmp_path / "junk.bin").write_bytes(b"not a pickle at all")
    with pytest.raises(K.CheckError, match="pickle"):
        F.check_bin(tmp_path / "junk.bin")


def test_load_tokenizer_records_padding_and_pad(tmp_path):
    byte_tokenizer().save_pretrained(str(tmp_path / "q"))
    llama_tokenizer().save_pretrained(str(tmp_path / "l"))
    tq, fq = F.load_tokenizer(tmp_path / "q")
    tl, fl = F.load_tokenizer(tmp_path / "l")
    assert (fq["padding_side"], fq["pad_was_set"], fq["pad_is_eos"]) == ("right", True, True)
    assert (fl["padding_side"], fl["pad_was_set"], fl["pad_is_eos"]) == ("left", False, True)
    assert tl.pad_token == tl.eos_token and bytes.fromhex(fl["pad_token_hex"]).decode() == "</s>"
    K.assert_text_free(fq)
    K.assert_text_free(fl)


def test_token_facts_padded_lengths_and_truncation():
    tok = byte_tokenizer()
    texts = ["ab", "abcd", "a", "abcdefgh", "abc"]
    f = F.token_facts(tok, texts, batch_size=2, max_length=6)
    assert f["padded"] == [4, 6, 3] and f["max_padded"] == 6 and f["truncated"] == 1
    assert f["tokens"] == {"min": 1, "median": 3, "max": 8}
    assert K.positions(f["padded"], 5, 2) == 2 * 4 + 2 * 6 + 1 * 3


class Block(torch.nn.Module):
    def __init__(self, i):
        super().__init__()
        self.i = i

    def forward(self, x):
        return x + self.i


class TinyViT(torch.nn.Module):
    """A stand-in for a timm ViT: a prefix token, blocks adding their index; the hooks see each block's output."""

    num_prefix_tokens = 1

    def __init__(self, n):
        super().__init__()
        self.blocks = torch.nn.ModuleList(Block(i + 1) for i in range(n))

    def forward(self, x):
        h = torch.cat([x.mean(1, keepdim=True), x], 1)  # a "CLS" token first
        for b in self.blocks:
            h = b(h)
        return h[:, 0]


class FakeImage:
    def __init__(self, v):
        self.v = v

    def convert(self, mode):
        assert mode == "RGB"
        return self


def test_embed_images_hooks_each_block_in_order_across_batches():
    imgs = list(range(7))
    tf = lambda im: torch.full((3, 2), float(im.v))  # noqa: E731
    layers = F.embed_images(TinyViT(4), tf, imgs, batch_size=3, decode=FakeImage)
    assert len(layers) == 4 and all(f.shape == (7, 2) and f.dtype == torch.float32 for f in layers)
    for b, f in enumerate(layers):  # block b's CLS: the image's mean plus 1 + ... + (b + 1)
        assert torch.equal(f[:, 0], torch.tensor([float(i) + sum(range(1, b + 2)) for i in imgs]))


def test_predict_peak_arithmetic():
    cfg = {"hidden_size": 5120, "num_hidden_layers": 48, "num_attention_heads": 40, "intermediate_size": 13824}
    p = F.predict_peak("qwen14b", 29_540_000_000, cfg, 64, 100, total_gib=44.4)
    assert p["parts_gib"]["hidden_states"] == round(49 * 64 * 100 * 5120 * 2 / 2**30, 2)
    assert p["parts_gib"]["weights"] == round(29_540_000_000 / 2**30, 2) and p["fits"] is True
    p512 = F.predict_peak("qwen14b", 29_540_000_000, cfg, 64, 512, total_gib=44.4)
    assert p512["fits"] is False  # every hidden state of a 512-token batch of 64 does not fit beside the weights
    small = F.predict_peak("qwen0.5b", 988_000_000, {**cfg, "hidden_size": 896, "num_hidden_layers": 24,
                                                     "num_attention_heads": 14, "intermediate_size": 4864}, 64, 50)
    assert small["parts_gib"]["weights"] == round(2 * 988_000_000 / 2**30, 2) and small["fits"] is None


def test_check_layers():
    good = [torch.zeros(4, 3) for _ in range(5)]
    F.check_layers(good, 5, 3, 4)
    with pytest.raises(K.CheckError, match="shape"):
        F.check_layers(good, 6, 3, 4)
    with pytest.raises(K.CheckError, match="shape"):
        F.check_layers(good, 5, 2, 4)
    bad = [torch.zeros(4, 3) for _ in range(5)]
    bad[2][1, 1] = float("nan")
    with pytest.raises(K.CheckError, match="nonfinite"):
        F.check_layers(bad, 5, 3, 4)


def test_decode_image_turns_by_exif_like_datasets():
    PIL = pytest.importorskip("PIL")
    import io

    from PIL import Image

    im = Image.new("RGB", (4, 2), (200, 10, 10))
    buf = io.BytesIO()
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 degrees
    im.save(buf, format="JPEG", exif=exif)
    out = F.decode_image(buf.getvalue())
    assert out.size == (2, 4) and PIL is not None
    plain = io.BytesIO()
    Image.new("L", (3, 5)).save(plain, format="PNG")
    assert F.decode_image(plain.getvalue()).size == (3, 5)
    facts = F.image_facts([buf.getvalue(), plain.getvalue()])
    assert facts["formats"] == {"JPEG": 1, "PNG": 1} and facts["exif_orientation"] == 1


def test_read_wit_in_file_order(tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    def write(path, start, n):
        rows = {"image": [{"bytes": bytes([i]), "path": None} for i in range(start, start + n)],
                "text": [[f"caption {i}", "other"] for i in range(start, start + n)], "origin": ["x"] * n}
        pq.write_table(pa.table(rows), path)

    write(tmp_path / "a.parquet", 0, 5)
    write(tmp_path / "b.parquet", 5, 4)
    imgs, caps = F.read_wit([tmp_path / "a.parquet", tmp_path / "b.parquet"])
    assert caps == [f"caption {i}" for i in range(9)] and imgs == [bytes([i]) for i in range(9)]
    imgs, caps = F.read_wit([tmp_path / "a.parquet", tmp_path / "b.parquet"], rows=7)
    assert len(caps) == 7 and imgs[-1] == bytes([6])
    none, caps = F.read_wit([tmp_path / "a.parquet"], images=False)
    assert none == [None] * 5


def test_safe_key_makes_decoder_names_text_free():
    assert F.safe_key("I;16") == "I_16" and F.safe_key("BGR;15") == "BGR_15" and F.safe_key("RGB") == "RGB"
    assert F.safe_key(None) == "None" and F.safe_key("a b/c") == "a_b_c" and len(F.safe_key("x" * 99)) == 60
    K.assert_text_free({F.safe_key(m): 1 for m in ["I;16", "I;16B", "I;16N", "BGR;24", "YCbCr", "RGBa", "MPO"]})


def test_image_facts_with_a_16_bit_png_is_text_free():
    pytest.importorskip("PIL")
    import io

    from PIL import Image

    png, tif = io.BytesIO(), io.BytesIO()
    Image.new("I;16", (3, 2)).save(png, format="PNG")  # 16-bit greyscale, common on Wikipedia
    Image.new("I;16", (3, 2)).save(tif, format="TIFF")
    raw = Image.open(io.BytesIO(png.getvalue())).mode
    facts = F.image_facts([png.getvalue(), tif.getvalue()])
    K.assert_text_free(facts)  # what prepare's write step checks
    assert all(";" not in m for m in facts["modes"]) and sum(facts["modes"].values()) == 2
    assert F.safe_key(raw) in facts["modes"] and facts["formats"] == {"PNG": 1, "TIFF": 1}
