"""Build Whisper encoder and decoder graphs with TensorRT's network API."""

import numpy as np
import tensorrt as trt
from tokenizers import Tokenizer

from ._checkpoint import Checkpoint
from ._plugins import _fc1_plugin


class _Graph:
    def __init__(self, network: trt.INetworkDefinition, checkpoint: Checkpoint, precision: str):
        self.net = network
        self.checkpoint = checkpoint
        self.config = checkpoint.config
        self.precision = precision
        self._weights: list[np.ndarray] = []

    def constant(self, value: np.ndarray, *, fp32: bool = False) -> trt.ITensor:
        dtype = np.float32 if fp32 or self.precision == "fp32" else np.float16
        array = np.ascontiguousarray(value, dtype=dtype)
        self._weights.append(array)
        return self.net.add_constant(array.shape, array).get_output(0)

    def half(self, value: trt.ITensor) -> trt.ITensor:
        if self.precision == "fp16":
            return self.net.add_cast(value, trt.float16).get_output(0)
        return value

    def single(self, value: trt.ITensor) -> trt.ITensor:
        if self.precision == "fp16":
            return self.net.add_cast(value, trt.float32).get_output(0)
        return value

    def weight(self, key: str) -> np.ndarray:
        return self.checkpoint.tensor(key)

    def add(self, left: trt.ITensor, right: trt.ITensor) -> trt.ITensor:
        return self.net.add_elementwise(left, right, trt.ElementWiseOperation.SUM).get_output(0)

    def multiply(self, left: trt.ITensor, right: trt.ITensor) -> trt.ITensor:
        return self.net.add_elementwise(left, right, trt.ElementWiseOperation.PROD).get_output(0)

    def linear(self, x: trt.ITensor, key: str, *, bias: bool = True) -> trt.ITensor:
        weight = self.constant(self.weight(f"{key}.weight").T[None, :, :])
        y = self.net.add_matrix_multiply(
            x, trt.MatrixOperation.NONE, weight, trt.MatrixOperation.NONE
        ).get_output(0)
        if bias:
            y = self.add(y, self.constant(self.weight(f"{key}.bias")[None, None, :]))
        return y

    def layer_norm(self, x: trt.ITensor, key: str) -> trt.ITensor:
        scale = self.constant(self.weight(f"{key}.weight")[None, None, :], fp32=True)
        bias = self.constant(self.weight(f"{key}.bias")[None, None, :], fp32=True)
        layer = self.net.add_normalization_v2(self.single(x), scale, bias, 1 << 2)
        layer.epsilon = 1e-5
        return self.half(layer.get_output(0))

    def gelu(self, x: trt.ITensor) -> trt.ITensor:
        # 0.5 * x * (1 + erf(x / sqrt(2))), matching torch's exact GELU.
        scalar_shape = (1,) * len(x.shape)
        inv_sqrt2 = self.constant(np.full(scalar_shape, 2**-0.5, dtype=np.float32))
        one = self.constant(np.ones(scalar_shape, dtype=np.float32))
        half = self.constant(np.full(scalar_shape, 0.5, dtype=np.float32))
        scaled = self.multiply(x, inv_sqrt2)
        erf = self.net.add_unary(scaled, trt.UnaryOperation.ERF).get_output(0)
        return self.multiply(self.multiply(x, half), self.add(one, erf))

    def decoder_fc1(self, x: trt.ITensor, key: str) -> trt.ITensor:
        if self.precision == "fp32":
            return self.gelu(self.linear(x, key))
        weight = self.constant(self.weight(f"{key}.weight"))
        bias = self.constant(self.weight(f"{key}.bias"))
        layer = self.net.add_plugin_v3([x, weight, bias], [], _fc1_plugin())
        if layer is None:
            raise RuntimeError("TensorRT rejected the required fp16 fc1 layer")
        layer.name = f"{key}.gelu"
        return layer.get_output(0)

    def split_heads(self, x: trt.ITensor, n_head: int) -> trt.ITensor:
        head_dim = self.config.audio_state // n_head
        shuffle = self.net.add_shuffle(x)
        shuffle.reshape_dims = (0, 0, n_head, head_dim)
        shuffle.second_transpose = (0, 2, 1, 3)
        return shuffle.get_output(0)

    def merge_heads(self, x: trt.ITensor) -> trt.ITensor:
        shuffle = self.net.add_shuffle(x)
        shuffle.first_transpose = (0, 2, 1, 3)
        shuffle.reshape_dims = (0, 0, self.config.audio_state)
        return shuffle.get_output(0)

    def attention(
        self,
        x: trt.ITensor,
        context: trt.ITensor,
        key: str,
        n_head: int,
        mask: trt.ITensor | None = None,
    ) -> trt.ITensor:
        keys, values = self.project_kv(context, key, n_head)
        return self.attend(x, key, n_head, keys, values, mask)

    def project_kv(
        self, context: trt.ITensor, key: str, n_head: int
    ) -> tuple[trt.ITensor, trt.ITensor]:
        keys = self.split_heads(self.linear(context, f"{key}.k_proj", bias=False), n_head)
        values = self.split_heads(self.linear(context, f"{key}.v_proj"), n_head)
        return keys, values

    def attend(
        self,
        x: trt.ITensor,
        key: str,
        n_head: int,
        keys: trt.ITensor,
        values: trt.ITensor,
        mask: trt.ITensor | None = None,
    ) -> trt.ITensor:
        query = self.split_heads(self.linear(x, f"{key}.q_proj"), n_head)
        scale = (self.config.audio_state // n_head) ** -0.25
        factor = self.constant(np.full((1, 1, 1, 1), scale, dtype=np.float32))
        query = self.multiply(query, factor)
        keys = self.multiply(keys, factor)
        scores = self.net.add_matrix_multiply(
            query, trt.MatrixOperation.NONE, keys, trt.MatrixOperation.TRANSPOSE
        ).get_output(0)
        scores = self.single(scores)
        if mask is not None:
            scores = self.add(scores, mask)
        softmax = self.net.add_softmax(scores)
        softmax.axes = 1 << 3
        attended = self.net.add_matrix_multiply(
            self.half(softmax.get_output(0)),
            trt.MatrixOperation.NONE,
            values,
            trt.MatrixOperation.NONE,
        ).get_output(0)
        return self.linear(self.merge_heads(attended), f"{key}.out_proj")

    def concatenate(self, tensors: list[trt.ITensor], axis: int) -> trt.ITensor:
        layer = self.net.add_concatenation(tensors)
        layer.axis = axis
        return layer.get_output(0)

    def stack_layers(self, tensors: list[trt.ITensor], name: str) -> None:
        expanded = []
        for tensor in tensors:
            shuffle = self.net.add_shuffle(tensor)
            shuffle.reshape_dims = (1, *tensor.shape)
            expanded.append(shuffle.get_output(0))
        output = self.concatenate(expanded, 0)
        output.name = name
        self.net.mark_output(output)

    def cache_layer(self, tensor: trt.ITensor, index: int) -> trt.ITensor:
        array = np.asarray(index, dtype=np.int32)
        self._weights.append(array)
        indices = self.net.add_constant((), array).get_output(0)
        return self.net.add_gather(tensor, indices, 0).get_output(0)

    def _select_token(self, logits, suppression):
        scores = self.add(self.single(logits), suppression)
        maximum = self.net.add_reduce(scores, trt.ReduceOperation.MAX, 1 << 2, True).get_output(0)
        equal = self.net.add_elementwise(
            scores, maximum, trt.ElementWiseOperation.EQUAL
        ).get_output(0)
        indices = np.arange(self.config.vocab_size, dtype=np.int32)[None, None]
        sentinel = np.full((1, 1, 1), self.config.vocab_size, dtype=np.int32)
        self._weights.extend((indices, sentinel))
        indices_tensor = self.net.add_constant(indices.shape, indices).get_output(0)
        sentinel_tensor = self.net.add_constant(sentinel.shape, sentinel).get_output(0)
        eligible = self.net.add_select(equal, indices_tensor, sentinel_tensor).get_output(0)
        # NumPy/torch argmax choose the first ID on ties. Do not depend on TopK's tie order.
        return self.net.add_reduce(eligible, trt.ReduceOperation.MIN, 1 << 2, True).get_output(0)

    def _greedy_result(self, logits, suppression, *, token, no_speech):
        token = self.net.add_cast(token, trt.float32).get_output(0)
        scores = self.add(self.single(logits), suppression)
        distribution = self.net.add_softmax(scores)
        distribution.axes = 1 << 2
        maximum = self.net.add_reduce(
            distribution.get_output(0), trt.ReduceOperation.MAX, 1 << 2, True
        ).get_output(0)
        logprob = self.net.add_unary(maximum, trt.UnaryOperation.LOG).get_output(0)
        if no_speech is None:
            silence = self.constant(np.full((1, 1, 1), np.nan, dtype=np.float32), fp32=True)
        else:
            distribution = self.net.add_softmax(self.single(logits))
            distribution.axes = 1 << 2
            index = np.array([no_speech], dtype=np.int32)
            self._weights.append(index)
            indices = self.net.add_constant(index.shape, index).get_output(0)
            silence = self.net.add_gather(distribution.get_output(0), indices, 2).get_output(0)
        return self.concatenate([token, logprob, silence], 2)

    def block(
        self,
        x: trt.ITensor,
        key: str,
        n_head: int,
        *,
        encoder: trt.ITensor | None = None,
        mask: trt.ITensor | None = None,
    ) -> trt.ITensor:
        normalized = self.layer_norm(x, f"{key}.self_attn_layer_norm")
        x = self.add(x, self.attention(normalized, normalized, f"{key}.self_attn", n_head, mask))
        if encoder is not None:
            normalized = self.layer_norm(x, f"{key}.encoder_attn_layer_norm")
            x = self.add(x, self.attention(normalized, encoder, f"{key}.encoder_attn", n_head))
        normalized = self.layer_norm(x, f"{key}.final_layer_norm")
        hidden = self.gelu(self.linear(normalized, f"{key}.fc1"))
        return self.add(x, self.linear(hidden, f"{key}.fc2"))

    def encoder(self) -> None:
        c = self.config
        x = self.net.add_input("mel", trt.float32, (1, c.n_mels, c.audio_positions * 2))
        x = self.half(x)
        expanded = self.net.add_shuffle(x)
        expanded.reshape_dims = (0, 0, 0, 1)
        x = expanded.get_output(0)
        for layer_id, stride in ((1, 1), (2, 2)):
            key = f"model.encoder.conv{layer_id}"
            dtype = np.float16 if self.precision == "fp16" else np.float32
            weight = np.asarray(self.weight(f"{key}.weight")[..., None], dtype=dtype)
            bias = np.asarray(self.weight(f"{key}.bias"), dtype=dtype)
            self._weights.extend((weight, bias))
            conv = self.net.add_convolution_nd(
                x,
                c.audio_state,
                (3, 1),
                weight,
                bias,
            )
            conv.padding_nd = (1, 0)
            conv.stride_nd = (stride, 1)
            x = self.gelu(conv.get_output(0))
        shuffle = self.net.add_shuffle(x)
        shuffle.first_transpose = (0, 2, 1, 3)
        shuffle.reshape_dims = (0, 0, 0)
        x = shuffle.get_output(0)
        position = self.constant(self.weight("model.encoder.embed_positions.weight")[None, :, :])
        x = self.add(x, position)
        for index in range(c.audio_layers):
            x = self.block(x, f"model.encoder.layers.{index}", c.audio_heads)
        x = self.layer_norm(x, "model.encoder.layer_norm")
        x.name = "encoder"
        self.net.mark_output(x)
        keys, values = [], []
        for index in range(c.text_layers):
            k, v = self.project_kv(x, f"model.decoder.layers.{index}.encoder_attn", c.text_heads)
            keys.append(k)
            values.append(v)
        self.stack_layers(keys, "cross_keys")
        self.stack_layers(values, "cross_values")

    def decoder(self) -> None:
        c = self.config
        tokens = self.net.add_input("tokens", trt.int32, (1, 1))
        positions = self.net.add_input("positions", trt.int32, (1, 1))
        mask = self.net.add_input("mask", trt.float32, (1, 1, 1, -1))
        dtype = trt.float16 if self.precision == "fp16" else trt.float32
        head_dim = c.audio_state // c.text_heads
        self_shape = (c.text_layers, 1, c.text_heads, -1, head_dim)
        cross_shape = (c.text_layers, 1, c.text_heads, c.audio_positions, head_dim)
        self_keys = self.net.add_input("self_keys", dtype, self_shape)
        self_values = self.net.add_input("self_values", dtype, self_shape)
        cross_keys = self.net.add_input("cross_keys", dtype, cross_shape)
        cross_values = self.net.add_input("cross_values", dtype, cross_shape)
        suppression = self.net.add_input("suppression", trt.float32, (1, 1, c.vocab_size))
        token_table = self.constant(self.weight("model.decoder.embed_tokens.weight"))
        position_table = self.constant(self.weight("model.decoder.embed_positions.weight"))
        x = self.add(
            self.net.add_gather(token_table, tokens, 0).get_output(0),
            self.net.add_gather(position_table, positions, 0).get_output(0),
        )
        present_keys, present_values = [], []
        for index in range(c.text_layers):
            key = f"model.decoder.layers.{index}"
            normalized = self.layer_norm(x, f"{key}.self_attn_layer_norm")
            k, v = self.project_kv(normalized, f"{key}.self_attn", c.text_heads)
            k = self.concatenate([self.cache_layer(self_keys, index), k], 2)
            v = self.concatenate([self.cache_layer(self_values, index), v], 2)
            present_keys.append(k)
            present_values.append(v)
            x = self.add(x, self.attend(normalized, f"{key}.self_attn", c.text_heads, k, v, mask))
            normalized = self.layer_norm(x, f"{key}.encoder_attn_layer_norm")
            x = self.add(
                x,
                self.attend(
                    normalized,
                    f"{key}.encoder_attn",
                    c.text_heads,
                    self.cache_layer(cross_keys, index),
                    self.cache_layer(cross_values, index),
                ),
            )
            normalized = self.layer_norm(x, f"{key}.final_layer_norm")
            hidden = self.decoder_fc1(normalized, f"{key}.fc1")
            x = self.add(x, self.linear(hidden, f"{key}.fc2"))
        x = self.layer_norm(x, "model.decoder.layer_norm")
        projected_table = self.net.add_shuffle(token_table)
        projected_table.reshape_dims = (1, c.vocab_size, c.audio_state)
        logits = self.net.add_matrix_multiply(
            x,
            trt.MatrixOperation.NONE,
            projected_table.get_output(0),
            trt.MatrixOperation.TRANSPOSE,
        ).get_output(0)
        logits.name = "logits"
        self.net.mark_output(logits)
        token = self._select_token(logits, suppression)
        token.name = "token"
        self.net.mark_output(token)
        tokenizer = Tokenizer.from_file(str(self.checkpoint.directory / "tokenizer.json"))
        result = self._greedy_result(
            logits, suppression, token=token, no_speech=tokenizer.token_to_id("<|nospeech|>")
        )
        result.name = "greedy"
        self.net.mark_output(result)
        self.stack_layers(present_keys, "present_keys")
        self.stack_layers(present_values, "present_values")
