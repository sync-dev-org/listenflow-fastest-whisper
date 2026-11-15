"""Adapt synthetic GPU operators to the model-specific native test boundary.

Only tests use this dispatcher. Real CPU decoding, tokenizer, window control and
error mapping stay in the product; native resource tests run separately in C++
and through the GPU binding.
"""

from types import SimpleNamespace

import numpy as np


class Mel:
    def __init__(self, values):
        self.values = values

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self.values, dtype=dtype)

    def close(self):
        pass


class Audio:
    def __init__(self, backend, tensors):
        self.backend, self.tensors = backend, tensors

    def decoder(self, kind, batch=1):
        return Chain(self, batch)

    def close(self):
        pass


class Chain:
    def __init__(self, audio, batch):
        self.audio, self.batch, self.tokens = audio, batch, []

    def execute(self, parent, token, suppression):
        inputs = {
            "self_keys": np.array([0, *parent], np.float32).reshape(1, 1, 1, -1, 1),
            "tokens": np.array([[token]], np.int32),
            "positions": np.array([[len(parent)]], np.int32),
            "suppression": suppression,
            **{name: self.audio.tensors[name] for name in ("cross_keys", "cross_values")},
        }
        return self.audio.backend.decoder.run(
            inputs, device_outputs=("present_keys", "present_values"), reuse_cache=True
        )

    def prefix_step(self, parent, token):
        values = [] if parent is None else parent.tokens
        vocab = self.audio.backend.model.config.vocab_size
        result = self.execute(values, token, np.zeros((1, 1, vocab), np.float32))
        prefix = SimpleNamespace(tokens=[*values, token], close=lambda: None)
        return prefix, result["logits"].reshape(-1, vocab)[-1].copy()

    def greedy_step(self, token, first, score):
        model = self.audio.backend.model
        suppression = np.zeros((1, 1, model.config.vocab_size), np.float32)
        suppression[..., list(model.suppressed)] = -np.inf
        if first:
            suppression[..., list(model.begin_suppressed)] = -np.inf
        result = self.execute(self.tokens, token, suppression)
        self.tokens.append(token)
        raw = result["logits"].reshape(-1, model.config.vocab_size)[-1]
        filtered = raw + suppression.reshape(-1)
        selected = int(np.argmax(filtered))
        probability = float("nan")
        if score and (speech := model.tokenizer.token_to_id("<|nospeech|>")) is not None:
            probability = float(np.exp(raw[speech] - np.logaddexp.reduce(raw)))
        return (
            selected,
            float(filtered[selected] - np.logaddexp.reduce(filtered)) if score else float("nan"),
            probability,
        )

    def close(self):
        pass


class Model:
    def __init__(self, model, mel, encoder, decoder):
        self.model, self.features, self.encoder, self.decoder = model, mel, encoder, decoder
        self.closed = False

    def mel(self, samples):
        padded = np.pad(samples, (0, 480000 - samples.size))
        centered = np.pad(padded, (200, 200), mode="reflect")
        result = self.features.run({"audio": centered}, device_outputs=())
        return Mel(result["mel"])

    def log_mel_block(self, centered):
        return self.features.run({"audio": centered}, device_outputs=())["log_mel"]

    def encode_features(self, features):
        result = self.encoder.run(
            {"mel": np.asarray(features, np.float32)[None]}, device_outputs=()
        )
        return Audio(self, result)

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.decoder.close()
        self.encoder.close()
        self.features.close()


def initialize_state(model, mel, encoder, decoder):
    model._native_model = Model(model, mel, encoder, decoder)
    model._prefix_states, model._root_prefix = {}, ()
    model._prefix_decoder = model._greedy_decoder = None
    model._mel_tensor = model._encoded_audio = None
    model._greedy_length = 0
    model._sampled_decoder = model._sampled_batch = None
