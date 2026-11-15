"""Small real Whisper networks for native resource and binding integration tests."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file
from tokenizers import Tokenizer, models, pre_tokenizers

from listenflow.fastest_whisper._cached_model import CachedWhisper
from listenflow.fastest_whisper._decoding import DecodingSettings


def prepare_source_checkpoint(tmp_path_factory, *, text_positions=64):
    root = tmp_path_factory.mktemp("native-runtime")
    config = {
        "num_mel_bins": 80,
        "max_source_positions": 1500,
        "d_model": 16,
        "encoder_attention_heads": 1,
        "encoder_ffn_dim": 32,
        "encoder_layers": 1,
        "decoder_attention_heads": 1,
        "decoder_ffn_dim": 32,
        "decoder_layers": 1,
        "max_target_positions": text_positions,
        "vocab_size": 16,
        "_runtime_fixture": str(root),
    }
    (root / "config.json").write_text(json.dumps(config, sort_keys=True))
    names = [
        "[UNK]",
        "a",
        "b",
        "<|endoftext|>",
        "<|startoftranscript|>",
        "<|en|>",
        "<|transcribe|>",
        "<|notimestamps|>",
        "<|startofprev|>",
        "<|startoflm|>",
        "<|nospeech|>",
        "<|translate|>",
        "<|0.00|>",
        "<|0.02|>",
        "<|0.04|>",
        "c",
    ]
    tokenizer = Tokenizer(
        models.WordLevel(dict(zip(names, range(16), strict=True)), unk_token="[UNK]")
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(root / "tokenizer.json"))
    rng = np.random.default_rng(61003)
    weights = {}

    def linear(name, out=16, inp=16, bias=True):
        weights[name + ".weight"] = rng.normal(0, 0.05, (out, inp)).astype(np.float32)
        if bias:
            weights[name + ".bias"] = rng.normal(0, 0.01, out).astype(np.float32)

    def norm(name):
        weights[name + ".weight"] = np.ones(16, np.float32)
        weights[name + ".bias"] = np.zeros(16, np.float32)

    for side in ("encoder", "decoder"):
        prefix = "model." + side
        layer = prefix + ".layers.0"
        for attention in ("self_attn", "encoder_attn") if side == "decoder" else ("self_attn",):
            for projection in ("q", "k", "v", "out"):
                linear(layer + f".{attention}.{projection}_proj", bias=projection != "k")
            norm(layer + f".{attention}_layer_norm")
        linear(layer + ".fc1", 32, 16)
        linear(layer + ".fc2", 16, 32)
        norm(layer + ".final_layer_norm")
        norm(prefix + ".layer_norm")
    for name, inp in (("conv1", 80), ("conv2", 16)):
        weights[f"model.encoder.{name}.weight"] = rng.normal(0, 0.05, (16, inp, 3)).astype(
            np.float32
        )
        weights[f"model.encoder.{name}.bias"] = np.zeros(16, np.float32)
    for name, shape in (
        ("encoder.embed_positions", (1500, 16)),
        ("decoder.embed_positions", (text_positions, 16)),
        ("decoder.embed_tokens", (16, 16)),
    ):
        weights[f"model.{name}.weight"] = rng.normal(0, 0.05, shape).astype(np.float32)
    save_file(weights, root / "model.safetensors")
    return root


def native_checkpoint_lifetime(source_checkpoint):
    if os.environ.get("LFFW_USE_GPU") != "1":
        pytest.skip("set LFFW_USE_GPU=1 after reserving a GPU for native integration")
    yield source_checkpoint
    records = source_checkpoint / "native-engines.json"
    if records.exists():
        owned = json.loads(records.read_text())
        report = Path(__file__).resolve().parents[2] / ".nf/tmp/native-runtime-cache-cleanup.json"
        report.parent.mkdir(parents=True, exist_ok=True)
        outcomes = []
        for path, digest in owned.items():
            engine = Path(path)
            if engine.is_file():
                actual = hashlib.sha256(engine.read_bytes()).hexdigest()
                if actual != digest:
                    outcomes.append({"path": path, "removed": False, "reason": "contents changed"})
                    continue
                engine.unlink()
            outcomes.append({"path": path, "removed": True})
        previous = json.loads(report.read_text()) if report.exists() else []
        report.write_text(json.dumps(previous + outcomes, indent=2))


def record_engines(root, model):
    """Record the fixture's exact engine paths in the cache selected by pytest."""
    config = json.loads((root / "config.json").read_text())
    assert config["_runtime_fixture"] == str(root)
    record = root / "native-engines.json"
    values = json.loads(record.read_text()) if record.exists() else {}
    for path in model._native_model.info["engine_paths"].values():
        values[path] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    record.write_text(json.dumps(values, indent=2))


def native_model_lifetime(native_checkpoint):
    # A fresh model keeps close/failure ownership checks independent, while the
    # uniquely identified engines are reused in pytest's temporary cache.
    with CachedWhisper(
        native_checkpoint,
        precision="fp32",
        settings=DecodingSettings(temperatures=(0.0,), best_of=1),
    ) as model:
        record_engines(native_checkpoint, model)
        yield model


def encoded(model):
    return model.encode(np.zeros((80, 3000), np.float32))
