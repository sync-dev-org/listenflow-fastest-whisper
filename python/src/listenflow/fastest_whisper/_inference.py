"""Run independently built TensorRT encoder and decoder engines."""

from __future__ import annotations

import asyncio
import ctypes
import platform
import threading
import zlib
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from time import monotonic, sleep
from typing import Self

import numpy as np
import tensorrt as trt
from cuda.bindings import runtime as cudart
from tensoracle import EngineHandle
from tokenizers import Tokenizer

from ._audio import FFT_SIZE, HOP_SIZE, N_SAMPLES
from ._checkpoint import Checkpoint, ModelConfig
from ._decoding import (
    DecodeAttempt,
    DecodingSettings,
    decode_with_fallback,
    decoding_settings,
    search_candidates,
)
from ._engines import _open_mel_engine, open_whisper_engine
from ._longform import (
    DecodedWindow,
    LongTranscription,
    SegmentTrace,
    WindowTrace,
    apply_timestamp_rules,
    decoder_prefix,
    transcribe_mel,
)
from ._precision import PRECISIONS

_DEFAULT_DECODING_SETTINGS = DecodingSettings()
_CACHE_BOUNDARIES = (1, 8, 16, 24, 32, 48, 64, 96, 128, 160, 192, 224, 256, 320, 384, 448)
_POLL_INTERVAL_S = 50e-6
_POLL_TIMER_SLACK_NS = 1
_syscall = ctypes.CDLL("libc.so.6", use_errno=True).syscall
_syscall.argtypes = [ctypes.c_long, ctypes.c_int, *([ctypes.c_ulong] * 4)]
_syscall.restype = ctypes.c_long
# Linux x86_64 SYS_prctl; libc's int-returning prctl would truncate timer slack.
_prctl = partial(_syscall, 157)


def _timer_slack(value: int | None = None) -> int:
    machine = platform.machine()
    if machine != "x86_64":
        raise RuntimeError(
            f"timer slack is unsupported on architecture {machine!r}; requires x86_64"
        )
    option = 30 if value is None else 29  # PR_GET_TIMERSLACK / PR_SET_TIMERSLACK
    result = _prctl(option, 0 if value is None else value, 0, 0, 0)
    if result == -1:
        operation = "PR_GET_TIMERSLACK" if value is None else "PR_SET_TIMERSLACK"
        raise OSError(ctypes.get_errno(), f"prctl({operation}) failed")
    return ctypes.c_ulong(result).value


def _cuda(result: tuple, operation: str):
    if result[0] != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"{operation} failed: {result[0]}")
    return result[1] if len(result) > 1 else None


class _DeviceTensor:
    """Own one completed, contiguous CUDA allocation until its explicit release."""

    def __init__(
        self,
        pointer: int,
        shape: tuple[int, ...],
        dtype,
        device: int,
        *,
        release: Callable[[], None] | None = None,
    ):
        self.pointer = pointer
        self.shape = shape
        self.dtype = np.dtype(dtype)
        self.device = device
        self._release = release

    def numpy(self) -> np.ndarray:
        if self.pointer is None:
            raise RuntimeError("device tensor is closed")
        _cuda(cudart.cudaSetDevice(self.device), "cudaSetDevice")
        array = np.empty(self.shape, dtype=self.dtype)
        _cuda(
            cudart.cudaMemcpy(
                array.ctypes.data,
                self.pointer,
                array.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
            ),
            "cudaMemcpy device-to-host",
        )
        return array

    def close(self) -> None:
        pointer, self.pointer = self.pointer, None
        if pointer is not None:
            if self._release is not None:
                release, self._release = self._release, None
                release()
                return
            _cuda(cudart.cudaSetDevice(self.device), "cudaSetDevice")
            _cuda(cudart.cudaFree(pointer), "cudaFree")

    def __del__(self) -> None:
        with suppress(Exception):
            self.close()


def _release_tensors(tensors: dict[str, np.ndarray | _DeviceTensor]) -> None:
    for tensor in tensors.values():
        if isinstance(tensor, _DeviceTensor):
            tensor.close()


class _EncodedAudio:
    """One audio window, retaining encoder output and projected K/V on the device."""

    def __init__(self, tensors: dict[str, np.ndarray | _DeviceTensor]):
        self.tensors = tensors

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        tensor = self.tensors["encoder"]
        array = tensor.numpy() if isinstance(tensor, _DeviceTensor) else tensor
        return (
            np.asarray(array[0], dtype=dtype).copy() if copy else np.asarray(array[0], dtype=dtype)
        )

    def close(self) -> None:
        _release_tensors(self.tensors)


class _GreedyCache(_DeviceTensor):
    """Borrow padded input storage for the next step of one mutable greedy chain."""

    def __init__(self, tensor, owner, length, bucket):
        super().__init__(
            tensor.pointer,
            (*tensor.shape[:-2], bucket, tensor.shape[-1]),
            tensor.dtype,
            tensor.device,
            release=lambda: None,
        )
        self.owner = owner
        self.length = length


@dataclass
class _PrefixState:
    tensors: dict[str, np.ndarray | _DeviceTensor]
    logits: np.ndarray

    def close(self) -> None:
        _release_tensors(self.tensors)


@dataclass
class _Buffer:
    pointer: int
    nbytes: int
    lease: _DeviceTensor | None = None


@dataclass
class _TensorIO:
    name: str
    dtype: type
    shape: tuple[int, ...]
    nbytes: int
    buffer: _Buffer | None = None
    host: np.ndarray | None = None


class _CudaGraphs:
    """Retain one immutable execution context for each padded decoder length interval."""

    def __init__(self, handle: EngineHandle):
        self.handle = handle
        self.entries = {}

    @staticmethod
    def _release(entry):
        context, executable, _ = entry
        try:
            _cuda(cudart.cudaGraphExecDestroy(executable), "cudaGraphExecDestroy")
        finally:
            context.close()

    def close(self):
        entries, self.entries = self.entries, {}
        for entry in entries.values():
            self._release(entry)
        self.handle = None

    def enqueue(self, execution, addresses, *, stream):
        shapes = {name: execution.get_tensor_shape(name) for name in execution._ranges}
        key = tuple(shapes.items())
        signature = tuple(addresses.items())
        if key in self.entries and self.entries[key][2] != signature:
            self._release(self.entries.pop(key))
        if key not in self.entries:
            context = self.handle.create_execution()
            try:
                context.set_input_shapes(shapes)
                context.enqueue(addresses, stream=stream)
                _cuda(cudart.cudaStreamSynchronize(stream), "cudaStreamSynchronize")
                _cuda(
                    cudart.cudaStreamBeginCapture(
                        stream,
                        cudart.cudaStreamCaptureMode.cudaStreamCaptureModeThreadLocal,
                    ),
                    "cudaStreamBeginCapture",
                )
                try:
                    context.enqueue(addresses, stream=stream)
                except BaseException:
                    result = cudart.cudaStreamEndCapture(stream)
                    if result[0] == cudart.cudaError_t.cudaSuccess:
                        _cuda(cudart.cudaGraphDestroy(result[1]), "cudaGraphDestroy")
                    raise
                graph = _cuda(cudart.cudaStreamEndCapture(stream), "cudaStreamEndCapture")
                try:
                    executable = _cuda(
                        cudart.cudaGraphInstantiate(graph, 0), "cudaGraphInstantiate"
                    )
                finally:
                    _cuda(cudart.cudaGraphDestroy(graph), "cudaGraphDestroy")
            except BaseException:
                context.close()
                raise
            self.entries[key] = context, executable, signature
        _, executable, _ = self.entries[key]
        _cuda(cudart.cudaGraphLaunch(executable, stream), "cudaGraphLaunch")


class _HostIO:
    """Exchange host arrays or resident tensors with tensoracle on one owned stream."""

    def __init__(self, handle: EngineHandle, *, device: int = 0):
        self.device = device
        self.execution = None
        self.stream = None
        self._completion = None
        self._buffers: dict[str, list[_Buffer]] = {}
        self._host_buffers: dict[str, tuple[int, np.ndarray]] = {}
        self._capacities: dict[str, int] = {}
        self._graphs = None
        self._layouts = {}
        self._input_shapes = None
        try:
            _cuda(cudart.cudaSetDevice(device), "cudaSetDevice")
            self.execution = handle.create_execution()
            engine = handle.engine
            names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
            self.inputs = tuple(
                name for name in names if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            )
            self.outputs = tuple(name for name in names if name not in self.inputs)
            if "logits" in self.outputs:
                self._graphs = _CudaGraphs(handle)
            self.ranged_inputs = tuple(
                name
                for name in self.inputs
                if -1 in engine.get_tensor_shape(name)
                and tuple(engine.get_tensor_profile_shape(name, 0)[0])
                != tuple(engine.get_tensor_profile_shape(name, 0)[2])
            )
            self.execution.set_input_shapes(
                {
                    name: tuple(engine.get_tensor_profile_shape(name, 0)[2])
                    for name in self.ranged_inputs
                }
            )
            self._capacities = {
                name: int(np.prod(self.execution.get_tensor_shape(name)))
                * np.dtype(trt.nptype(self.execution.get_tensor_dtype(name))).itemsize
                for name in names
            }
            self.stream = _cuda(cudart.cudaStreamCreate(), "cudaStreamCreate")
            self._completion = _cuda(
                cudart.cudaEventCreateWithFlags(cudart.cudaEventDisableTiming),
                "cudaEventCreateWithFlags",
            )
        except BaseException:
            self.close()
            # Retained initialization errors must not keep native owners alive.
            handle = engine = None
            raise

    def close(self) -> None:
        stream = getattr(self, "stream", None)
        self.stream = None
        completion, self._completion = getattr(self, "_completion", None), None
        execution, self.execution = getattr(self, "execution", None), None
        graphs, self._graphs = getattr(self, "_graphs", None), None
        self._layouts = {}
        self._input_shapes = None
        try:
            if stream is not None:
                _cuda(cudart.cudaSetDevice(self.device), "cudaSetDevice")
                try:
                    _cuda(cudart.cudaStreamSynchronize(stream), "cudaStreamSynchronize")
                finally:
                    try:
                        if graphs is not None:
                            graphs.close()
                    finally:
                        try:
                            if completion is not None:
                                _cuda(cudart.cudaEventDestroy(completion), "cudaEventDestroy")
                        finally:
                            _cuda(cudart.cudaStreamDestroy(stream), "cudaStreamDestroy")
        finally:
            try:
                buffers, self._buffers = self._buffers, {}
                for entries in buffers.values():
                    for buffer in entries:
                        if buffer.lease is not None:
                            buffer.lease.close()
                        _cuda(cudart.cudaFree(buffer.pointer), "cudaFree")
            finally:
                try:
                    host_buffers, self._host_buffers = self._host_buffers, {}
                    for pointer, _ in host_buffers.values():
                        _cuda(cudart.cudaFreeHost(pointer), "cudaFreeHost")
                finally:
                    if execution is not None:
                        execution.close()

    def _wait(self) -> None:
        _cuda(cudart.cudaEventRecord(self._completion, self.stream), "cudaEventRecord")
        previous_slack = _timer_slack()
        try:
            _timer_slack(_POLL_TIMER_SLACK_NS)
            while True:
                result = cudart.cudaEventQuery(self._completion)
                if result[0] != cudart.cudaError_t.cudaErrorNotReady:
                    _cuda(result, "cudaEventQuery")
                    break
                sleep(_POLL_INTERVAL_S)
        except BaseException as error:
            try:
                _timer_slack(previous_slack)
            except OSError as restore_error:
                error.add_note(f"Timer slack restoration also failed: {restore_error}")
            raise
        _timer_slack(previous_slack)

    def _enqueue(self, addresses, *, stream):
        if self._graphs is None:
            self.execution.enqueue(addresses, stream=stream)
        else:
            self._graphs.enqueue(self.execution, addresses, stream=stream)

    def _layout(self, prepared):
        key = tuple(prepared[name].shape for name in self.ranged_inputs)
        layout = self._layouts.get(key)
        if layout is None:
            shapes = dict(zip(self.ranged_inputs, key, strict=True))
            self._input_shapes = None
            self.execution.set_input_shapes(shapes)
            tensors = []
            for name in (*self.inputs, *self.outputs):
                dtype = trt.nptype(self.execution.get_tensor_dtype(name))
                shape = self.execution.get_tensor_shape(name)
                nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
                tensors.append(_TensorIO(name, dtype, shape, nbytes))
            layout = shapes, tuple(tensors)
            self._layouts[key] = layout
        elif self._input_shapes is not layout[0]:
            self._input_shapes = None
            self.execution.set_input_shapes(layout[0])
        self._input_shapes = layout[0]
        return layout[1]

    def _buffer(self, name: str, nbytes: int) -> _Buffer:
        entries = self._buffers.setdefault(name, [])
        for buffer in entries:
            if buffer.lease is None and buffer.nbytes >= nbytes:
                return buffer
        capacity = max(1, nbytes, self._capacities[name])
        pointer = _cuda(cudart.cudaMalloc(capacity), "cudaMalloc")
        buffer = _Buffer(int(pointer), capacity)
        entries.append(buffer)
        return buffer

    def _host_array(self, tensor: _TensorIO) -> np.ndarray:
        if tensor.host is None:
            allocation = self._host_buffers.get(tensor.name)
            if allocation is None:
                capacity = max(1, tensor.nbytes, self._capacities[tensor.name])
                pointer = int(_cuda(cudart.cudaMallocHost(capacity), "cudaMallocHost"))
                raw = np.ctypeslib.as_array((ctypes.c_ubyte * capacity).from_address(pointer))
                allocation = self._host_buffers[tensor.name] = pointer, raw
            tensor.host = allocation[1][: tensor.nbytes].view(tensor.dtype).reshape(tensor.shape)
        return tensor.host

    def _lease(self, buffer: _Buffer, shape: tuple[int, ...], dtype) -> _DeviceTensor:
        def release():
            buffer.lease = None

        buffer.lease = _DeviceTensor(buffer.pointer, shape, dtype, self.device, release=release)
        return buffer.lease

    def _pad_cache(self, inputs, *, reuse=False):
        """Pad immutable histories or reuse one greedy chain at stable input addresses."""
        shape = inputs["self_keys"].shape
        length, width = shape[-2:]
        if reuse and isinstance(inputs["self_keys"], _GreedyCache):
            length = inputs["self_keys"].length
        for name in ("self_keys", "self_values", "mask"):
            array = inputs[name]
            low, high = self.execution._ranges[name]
            if len(array.shape) != len(low) or any(
                not minimum <= value <= maximum
                for value, minimum, maximum in zip(array.shape, low, high, strict=True)
            ):
                raise ValueError(f"{name} shape is outside its input profile")
            if array.dtype != trt.nptype(self.execution.get_tensor_dtype(name)):
                raise ValueError(f"{name} has an incompatible dtype")
        if inputs["self_values"].shape != shape or inputs["mask"].shape[-1] != length + 1:
            raise ValueError("decoder keys, values and mask must share one cache length")
        bucket = next(value for value in _CACHE_BOUNDARIES if value >= length)
        rows = int(np.prod(shape[:-2]))
        rowbytes = width * inputs["self_keys"].dtype.itemsize
        padded, borrowed = dict(inputs), []
        for name in ("self_keys", "self_values"):
            source = inputs[name]
            if isinstance(source, _DeviceTensor):
                if source.pointer is None:
                    raise RuntimeError(f"{name} device tensor is closed")
                if source.device != self.device:
                    raise ValueError(f"{name} must be on CUDA device {self.device}")
            buffer = self._buffer(name, rows * bucket * rowbytes)
            if reuse and isinstance(source, _GreedyCache):
                if source.owner is not self or source.pointer != buffer.pointer:
                    raise ValueError(f"{name} belongs to another greedy cache")
                if source.length != length or source.shape[-2] != bucket:
                    raise ValueError("decoder keys and values must share one greedy cache length")
            else:
                _cuda(
                    cudart.cudaMemsetAsync(
                        buffer.pointer, 0, rows * bucket * rowbytes, self.stream
                    ),
                    "cudaMemsetAsync padded cache",
                )
                resident = isinstance(source, _DeviceTensor)
                _cuda(
                    cudart.cudaMemcpy2DAsync(
                        buffer.pointer,
                        bucket * rowbytes,
                        source.pointer if resident else source.ctypes.data,
                        length * rowbytes,
                        length * rowbytes,
                        rows,
                        cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice
                        if resident
                        else cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                        self.stream,
                    ),
                    "cudaMemcpy2DAsync padded cache",
                )
            tensor = _DeviceTensor(
                buffer.pointer,
                (*shape[:-2], bucket, width),
                source.dtype,
                self.device,
                release=lambda: None,
            )
            padded[name] = tensor
            borrowed.append(tensor)
        mask = np.full((*inputs["mask"].shape[:-1], bucket + 1), -np.inf, inputs["mask"].dtype)
        mask[..., :length] = inputs["mask"][..., :length]
        mask[..., -1] = inputs["mask"][..., -1]
        padded["mask"] = mask
        return padded, borrowed, length

    def _advance_cache(self, source, target, length):
        """Append one position, moving the full history only to enter a larger bucket."""
        bucket, width = target.shape[-2:]
        if length == _CACHE_BOUNDARIES[-1]:
            return source  # The final context position is already contiguous.
        rows = int(np.prod(target.shape[:-2]))
        rowbytes = width * source.dtype.itemsize
        next_bucket = next(value for value in _CACHE_BOUNDARIES if value >= length + 1)
        if next_bucket != bucket:
            _cuda(
                cudart.cudaMemsetAsync(
                    target.pointer, 0, rows * next_bucket * rowbytes, self.stream
                ),
                "cudaMemsetAsync greedy cache",
            )
            offset, source_offset, copied = 0, 0, (length + 1) * rowbytes
        else:
            offset, source_offset, copied = length * rowbytes, bucket * rowbytes, rowbytes
        _cuda(
            cudart.cudaMemcpy2DAsync(
                target.pointer + offset,
                next_bucket * rowbytes,
                source.pointer + source_offset,
                (bucket + 1) * rowbytes,
                copied,
                rows,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
                self.stream,
            ),
            "cudaMemcpy2DAsync greedy cache",
        )
        result = _GreedyCache(target, self, length + 1, next_bucket)
        source.close()
        return result

    def _compact_cache(self, name, source, length):
        """Move the appended position past the real history, independently in each row."""
        shape = source.shape
        rows = int(np.prod(shape[:-2]))
        rowbytes = shape[-1] * source.dtype.itemsize
        pool = "_compact_" + name
        self._capacities[pool] = self._capacities[name]
        buffer = self._buffer(pool, rows * (length + 1) * rowbytes)
        for offset, source_offset, width in (
            (0, 0, length * rowbytes),
            (length * rowbytes, (shape[-2] - 1) * rowbytes, rowbytes),
        ):
            _cuda(
                cudart.cudaMemcpy2DAsync(
                    buffer.pointer + offset,
                    (length + 1) * rowbytes,
                    source.pointer + source_offset,
                    shape[-2] * rowbytes,
                    width,
                    rows,
                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
                    self.stream,
                ),
                "cudaMemcpy2DAsync compact cache",
            )
        tensor = self._lease(buffer, (*shape[:-2], length + 1, shape[-1]), source.dtype)
        source.close()
        return tensor

    def __del__(self) -> None:
        with suppress(Exception):
            self.close()

    def run(
        self,
        inputs: dict[str, np.ndarray | _DeviceTensor],
        *,
        device_outputs: tuple[str, ...] = (),
        reuse_cache: bool = False,
    ) -> dict[str, np.ndarray | _DeviceTensor]:
        if self.execution is None:
            raise RuntimeError("host inference is closed")
        _cuda(cudart.cudaSetDevice(self.device), "cudaSetDevice")
        buffers: dict[str, int] = {}
        prepared = {
            name: array if isinstance(array, _DeviceTensor) else np.ascontiguousarray(array)
            for name, array in inputs.items()
        }
        outputs: dict[str, np.ndarray | _DeviceTensor] = {}
        borrowed = []
        cache_length = None
        succeeded = False
        try:
            if self._graphs is not None:
                prepared, borrowed, cache_length = self._pad_cache(prepared, reuse=reuse_cache)
            for tensor in self._layout(prepared):
                name, expected_dtype, shape = tensor.name, tensor.dtype, tensor.shape
                if name in self.inputs:
                    array = prepared[name]
                    if array.dtype != expected_dtype:
                        raise ValueError(f"{name} must have dtype {expected_dtype}")
                    if array.shape != shape:
                        raise ValueError(f"{name} must have shape {shape}")
                    if isinstance(array, _DeviceTensor):
                        if array.pointer is None:
                            raise RuntimeError(f"{name} device tensor is closed")
                        if array.device != self.device:
                            raise ValueError(f"{name} must be on CUDA device {self.device}")
                        buffers[name] = array.pointer
                        continue
                elif name not in device_outputs:
                    array = self._host_array(tensor)
                    outputs[name] = array
                buffer = tensor.buffer
                if buffer is None or buffer.lease is not None:
                    buffer = tensor.buffer = self._buffer(name, tensor.nbytes)
                pointer = buffer.pointer
                buffers[name] = pointer
                if name in device_outputs:
                    outputs[name] = self._lease(buffer, shape, expected_dtype)
                if name in self.inputs:
                    _cuda(
                        cudart.cudaMemcpyAsync(
                            pointer,
                            array.ctypes.data,
                            array.nbytes,
                            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                            self.stream,
                        ),
                        "cudaMemcpyAsync host-to-device",
                    )
            self._enqueue(buffers, stream=int(self.stream))
            if cache_length is not None:
                for name in ("present_keys", "present_values"):
                    if isinstance(outputs.get(name), _DeviceTensor):
                        outputs[name] = (
                            self._advance_cache(
                                outputs[name],
                                prepared[name.replace("present", "self")],
                                cache_length,
                            )
                            if reuse_cache
                            else self._compact_cache(name, outputs[name], cache_length)
                        )
            for name, array in outputs.items():
                if isinstance(array, _DeviceTensor):
                    continue
                _cuda(
                    cudart.cudaMemcpyAsync(
                        array.ctypes.data,
                        buffers[name],
                        array.nbytes,
                        cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                        self.stream,
                    ),
                    "cudaMemcpyAsync device-to-host",
                )
            self._wait()
            # Staging storage belongs to this exchange; returned values belong
            # to the caller and must survive the next exchange and close().
            outputs = {
                name: array.copy() if isinstance(array, np.ndarray) else array
                for name, array in outputs.items()
            }
            if cache_length is not None:
                for name in ("present_keys", "present_values"):
                    if isinstance(outputs.get(name), np.ndarray):
                        array = outputs[name]
                        outputs[name] = np.concatenate(
                            (array[..., :cache_length, :], array[..., -1:, :]), axis=-2
                        )
            succeeded = True
            return outputs
        finally:
            _release_tensors({str(index): tensor for index, tensor in enumerate(borrowed)})
            if not succeeded:
                try:
                    _cuda(cudart.cudaStreamSynchronize(self.stream), "cudaStreamSynchronize")
                finally:
                    _release_tensors(outputs)


@dataclass(frozen=True)
class Transcription:
    """The decoded text and token IDs, including the final end-of-text token."""

    text: str
    tokens: list[int]


def _suppressed_tokens(tokenizer: Tokenizer) -> set[int]:
    # Match the official multilingual tokenizer's -1 non-speech policy.
    symbols = list('"#()*+/:;<=>@[\\]^_`{|}~「」『』')
    symbols += [
        "<<",
        ">>",
        "<<<",
        ">>>",
        "--",
        "---",
        "-(",
        "-[",
        "('",
        '("',
        "((",
        "))",
        "(((",
        ")))",
        "[[",
        "]]",
        "{{",
        "}}",
        "♪♪",
        "♪♪♪",
    ]
    miscellaneous = set("♩♪♫♬♭♮♯")
    result = {
        tokenizer.encode(" -", add_special_tokens=False).ids[0],
        tokenizer.encode(" '", add_special_tokens=False).ids[0],
    }
    for symbol in symbols + sorted(miscellaneous):
        for text in (symbol, " " + symbol):
            ids = tokenizer.encode(text, add_special_tokens=False).ids
            if len(ids) == 1 or symbol in miscellaneous:
                result.add(ids[0])
    for symbol in (
        "<|transcribe|>",
        "<|translate|>",
        "<|startoftranscript|>",
        "<|startofprev|>",
        "<|startoflm|>",
        "<|nospeech|>",
    ):
        token_id = tokenizer.token_to_id(symbol)
        if token_id is not None:
            result.add(token_id)
    return result


class TensorRTWhisper:
    """Own the two TensorRT engines during one comparison or transcription."""

    def __init__(self, checkpoint_dir: Path, *, precision: str = "fp32", device: int = 0):
        if precision not in PRECISIONS:
            raise ValueError(f"unsupported precision: {precision}")
        self.checkpoint_dir = Path(checkpoint_dir)
        self.precision = precision
        self.device = device
        self._encoder = self._decoder = self._mel = None
        self.encoder_engine = self.decoder_engine = None
        self._encoder_handle = self._decoder_handle = None
        self._mel_handle = None
        self._mel_tensor = None
        self._encoded_audio: _EncodedAudio | None = None
        self._prefix_states: dict[tuple[int, ...], _PrefixState] = {}
        self._root_prefix: tuple[int, ...] = ()
        self._greedy_cache = {}
        self._greedy_masks = {}
        self._cancel_requested: threading.Event | None = None
        try:
            _cuda(cudart.cudaSetDevice(device), "cudaSetDevice")
            self.tokenizer = Tokenizer.from_file(str(self.checkpoint_dir / "tokenizer.json"))
            self.logger = trt.Logger(trt.Logger.WARNING)
            self.build_seconds: dict[str, float] = {}
            with Checkpoint(self.checkpoint_dir) as checkpoint:
                self.config: ModelConfig = checkpoint.config
                self._encoder_handle = self._open_engine(checkpoint, decoder=False)
                self.encoder_engine = self._encoder_handle.engine
                self._decoder_handle = self._open_engine(checkpoint, decoder=True)
                self.decoder_engine = self._decoder_handle.engine
                self._mel_handle = self._open_mel(checkpoint)
            self._encoder = _HostIO(self._encoder_handle, device=device)
            self._decoder = _HostIO(self._decoder_handle, device=device)
            self._mel = _HostIO(self._mel_handle, device=device)
            required = (
                "<|startoftranscript|>",
                "<|en|>",
                "<|transcribe|>",
                "<|notimestamps|>",
                "<|endoftext|>",
            )
            ids = [self.tokenizer.token_to_id(token) for token in required]
            if any(token_id is None for token_id in ids):
                raise ValueError("checkpoint tokenizer lacks a required transcription token")
            self.initial_tokens = ids[:4]
            self.eot = ids[4]
            self.suppressed = _suppressed_tokens(self.tokenizer)
            self.begin_suppressed = set(self.tokenizer.encode(" ", add_special_tokens=False).ids)
            self.begin_suppressed.add(self.eot)
        except BaseException:
            self.close()
            raise

    def _open_engine(self, checkpoint: Checkpoint, *, decoder: bool):
        name = "decoder" if decoder else "encoder"
        build_started = monotonic()
        handle = open_whisper_engine(
            checkpoint,
            decoder=decoder,
            logger=self.logger,
            precision=self.precision,
            force_rebuild=True,
        )
        self.build_seconds[name] = monotonic() - build_started
        return handle

    def _open_mel(self, checkpoint: Checkpoint):
        started = monotonic()
        handle = _open_mel_engine(
            checkpoint, precision=self.precision, force_rebuild=True, logger=self.logger
        )
        self.build_seconds["mel"] = monotonic() - started
        return handle

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        """Return execution contexts and streams before releasing engines and runtimes."""
        decoder, self._decoder = getattr(self, "_decoder", None), None
        encoder, self._encoder = getattr(self, "_encoder", None), None
        mel, self._mel = getattr(self, "_mel", None), None
        try:
            self._clear_prefix_states()
            self._clear_greedy_cache()
            masks, self._greedy_masks = getattr(self, "_greedy_masks", {}), {}
            _release_tensors(masks)
            encoded, self._encoded_audio = getattr(self, "_encoded_audio", None), None
            if encoded is not None:
                encoded.close()
            tensor, self._mel_tensor = getattr(self, "_mel_tensor", None), None
            if isinstance(tensor, _DeviceTensor):
                tensor.close()
        finally:
            try:
                if decoder is not None:
                    decoder.close()
            finally:
                try:
                    if encoder is not None:
                        encoder.close()
                finally:
                    try:
                        if mel is not None:
                            mel.close()
                    finally:
                        self.decoder_engine = self.encoder_engine = None
                        self._decoder_handle = self._encoder_handle = self._mel_handle = None

    def _mel_spectrogram(self, audio: np.ndarray):
        self._check_cancelled()
        if self._mel is None:
            raise RuntimeError("Whisper model is closed")
        samples = np.asarray(audio, dtype=np.float32)
        if samples.ndim != 1 or samples.size > N_SAMPLES or not np.isfinite(samples).all():
            raise ValueError("audio must be a finite mono waveform of at most 30 seconds")
        previous, self._mel_tensor = getattr(self, "_mel_tensor", None), None
        if isinstance(previous, _DeviceTensor):
            previous.close()
        padded = np.pad(samples, (0, N_SAMPLES - samples.size))
        centered = np.pad(padded, (FFT_SIZE // 2, FFT_SIZE // 2), mode="reflect")
        outputs = self._mel.run({"audio": centered}, device_outputs=("mel", "log_mel"))
        tensor = outputs.pop("mel")
        _release_tensors(outputs)
        if isinstance(tensor, _DeviceTensor):
            tensor.shape = (1, *tensor.shape)
        self._mel_tensor = tensor
        return tensor

    def _mel_spectrogram_long(self, audio: np.ndarray) -> np.ndarray:
        self._check_cancelled()
        if self._mel is None:
            raise RuntimeError("Whisper model is closed")
        samples = np.asarray(audio, dtype=np.float32)
        if samples.ndim != 1 or not np.isfinite(samples).all():
            raise ValueError("audio must be a finite mono waveform")
        padded = np.pad(samples, (0, N_SAMPLES))
        centered = np.pad(padded, (FFT_SIZE // 2, FFT_SIZE // 2), mode="reflect")
        frames = padded.size // HOP_SIZE
        chunks = []
        for first in range(0, frames, 3000):
            self._check_cancelled()
            block = centered[first * HOP_SIZE : first * HOP_SIZE + N_SAMPLES + FFT_SIZE]
            block = np.pad(block, (0, N_SAMPLES + FFT_SIZE - block.size))
            outputs = self._mel.run({"audio": block}, device_outputs=("mel",))
            chunks.append(outputs["log_mel"][:, : min(3000, frames - first)])
            _release_tensors(outputs)
        log_mel = np.concatenate(chunks, axis=1)
        # Long-form window selection consumes host arrays. Use one threshold for
        # the complete recording rather than independently clipping each block.
        log_mel = np.maximum(log_mel, log_mel.max() - np.float32(8))
        return (log_mel + np.float32(4)) * np.float32(0.25)

    def __del__(self) -> None:
        with suppress(Exception):
            self.close()

    def _check_cancelled(self) -> None:
        cancelled = getattr(self, "_cancel_requested", None)
        if cancelled is not None and cancelled.is_set():
            raise asyncio.CancelledError

    def _clear_prefix_states(self) -> None:
        states, self._prefix_states = getattr(self, "_prefix_states", {}), {}
        self._root_prefix = ()
        for state in states.values():
            state.close()

    def encode(self, mel: np.ndarray) -> _EncodedAudio:
        self._check_cancelled()
        if self._encoder is None:
            raise RuntimeError("Whisper model is closed")
        self._clear_prefix_states()
        encoded, self._encoded_audio = getattr(self, "_encoded_audio", None), None
        if encoded is not None:
            encoded.close()
        outputs = self._encoder.run(
            {
                "mel": mel
                if isinstance(mel, _DeviceTensor)
                else np.asarray(mel, dtype=np.float32)[None]
            },
            device_outputs=("encoder", "cross_keys", "cross_values"),
        )
        self._encoded_audio = _EncodedAudio(outputs)
        return self._encoded_audio

    def logits(self, prefix: list[int], encoder: _EncodedAudio) -> np.ndarray:
        """Return prefix position scores, executing only positions absent from its cache.

        The graph returns one row per invocation. Host rows remain available for
        oracle comparisons and the start-of-transcript no-speech probability.
        Device cache states retain the root and live beam expansion depths only.
        """
        self._check_cancelled()
        if self._decoder is None:
            raise RuntimeError("Whisper model is closed")
        length = len(prefix)
        if not (1 <= length <= self.config.text_positions):
            raise ValueError("decoder prefix exceeds checkpoint context length")
        if encoder is not getattr(self, "_encoded_audio", None):
            raise ValueError("decoder requires the current encoded audio window")
        path = tuple(prefix)
        root = self._root_prefix
        if root and path[: len(root)] != root:
            self._clear_prefix_states()
        states = self._prefix_states
        parent_path = next(
            (path[:offset] for offset in range(length - 1, 0, -1) if path[:offset] in states),
            None,
        )
        # Keep the root for retries and the two depths used by beam expansion.
        for previous in list(states):
            if previous not in (self._root_prefix, parent_path) and len(previous) not in (
                length - 1,
                length,
            ):
                states.pop(previous).close()
        if path in states:
            return states[path].logits
        parent = None if parent_path is None else states[parent_path]
        offset = 0 if parent is None else len(parent.logits)
        rows = [] if parent is None else list(parent.logits)
        c = self.config
        dtype = np.dtype(PRECISIONS[self.precision].dtype)
        empty_shape = (c.text_layers, 1, c.text_heads, 1, c.audio_state // c.text_heads)
        cache = (
            {
                "self_keys": np.zeros(empty_shape, dtype=dtype),
                "self_values": np.zeros(empty_shape, dtype=dtype),
            }
            if parent is None
            else parent.tensors
        )
        owns_cache = False
        try:
            for position in range(offset, length):
                self._check_cancelled()
                mask = np.zeros((1, 1, 1, position + 2), dtype=np.float32)
                mask.flat[0] = -np.inf  # The input profile requires one dummy cache position.
                outputs = self._decoder.run(
                    {
                        "tokens": np.array([[prefix[position]]], dtype=np.int32),
                        "positions": np.array([[position]], dtype=np.int32),
                        "mask": mask,
                        **cache,
                        "cross_keys": encoder.tensors["cross_keys"],
                        "cross_values": encoder.tensors["cross_values"],
                        "suppression": np.zeros((1, 1, c.vocab_size), dtype=np.float32),
                    },
                    device_outputs=("present_keys", "present_values", "greedy"),
                )
                if owns_cache:
                    _release_tensors(cache)
                cache = {
                    "self_keys": outputs["present_keys"],
                    "self_values": outputs["present_values"],
                }
                owns_cache = True
                rows.append(outputs["logits"][0, 0])
                _release_tensors({"greedy": outputs.get("greedy")})
            state = _PrefixState(cache, np.stack(rows))
            states[path] = state
            owns_cache = False
            if not self._root_prefix:
                self._root_prefix = path
            return state.logits
        finally:
            if owns_cache:
                _release_tensors(cache)

    def prompt_tokens(self, initial_prompt: str | None) -> list[int]:
        if initial_prompt is None:
            return []
        return self.tokenizer.encode(" " + initial_prompt.strip(), add_special_tokens=False).ids

    def _clear_greedy_cache(self):
        cache, self._greedy_cache = getattr(self, "_greedy_cache", {}), {}
        _release_tensors(cache)
        self._greedy_length = 0

    def _greedy_step(self, prefix, encoder, *, first, score=False, speech_position=0):
        c = self.config
        if encoder is not self._encoded_audio:
            raise ValueError("decoder requires the current encoded audio window")
        if not 1 <= len(prefix) <= c.text_positions:
            raise ValueError("decoder prefix exceeds checkpoint context length")
        masks = getattr(self, "_greedy_masks", None)
        if not masks:
            mask = np.zeros((1, 1, c.vocab_size), dtype=np.float32)
            mask[..., list(self.suppressed)] = -np.inf
            begin = mask.copy()
            begin[..., list(self.begin_suppressed)] = -np.inf
            masks = self._greedy_masks = {False: mask, True: begin}
        no_speech_prob = float("nan")
        for position in range(self._greedy_length, len(prefix)):
            self._check_cancelled()
            dtype = np.dtype(PRECISIONS[self.precision].dtype)
            empty_shape = (c.text_layers, 1, c.text_heads, 1, c.audio_state // c.text_heads)
            cache = self._greedy_cache or {
                name: np.zeros(empty_shape, dtype=dtype) for name in ("self_keys", "self_values")
            }
            mask = np.zeros((1, 1, 1, position + 2), dtype=np.float32)
            mask.flat[0] = -np.inf
            outputs = self._decoder.run(
                {
                    "tokens": np.array([[prefix[position]]], dtype=np.int32),
                    "positions": np.array([[position]], dtype=np.int32),
                    "mask": mask,
                    **cache,
                    "cross_keys": encoder.tensors["cross_keys"],
                    "cross_values": encoder.tensors["cross_values"],
                    "suppression": masks[first],
                },
                device_outputs=(
                    "logits",
                    "present_keys",
                    "present_values",
                    "token" if score else "greedy",
                ),
                reuse_cache=True,
            )
            masks[first] = outputs.pop("suppression", masks[first])
            _release_tensors(cache)
            self._greedy_cache = {
                "self_keys": outputs["present_keys"],
                "self_values": outputs["present_values"],
            }
            _release_tensors({"logits": outputs["logits"]})
            self._greedy_length = position + 1
            if score:
                token, logprob, probability = outputs["greedy"].reshape(-1)
                if position == speech_position:
                    no_speech_prob = float(probability)
                _release_tensors({"token": outputs["token"]})
            else:
                token, logprob = outputs["token"].reshape(-1)[0], float("nan")
                _release_tensors({"greedy": outputs.get("greedy")})
        return int(token), float(logprob), no_speech_prob

    def _decode_attempt(
        self,
        encoder: _EncodedAudio,
        prompt: list[int],
        *,
        with_timestamps: bool,
        temperature: float,
        beam_size: int | None,
        best_of: int | None,
        sample_token: Callable[[np.ndarray], int] | None = None,
    ) -> DecodeAttempt:
        """Read one window with the chosen candidate selection rule."""
        if not with_timestamps and temperature == 0 and beam_size is None:
            return self._greedy_attempt(encoder, prompt)
        previous = self.tokenizer.token_to_id("<|startofprev|>")
        no_speech = self.tokenizer.token_to_id("<|nospeech|>")
        timestamp_begin = self.tokenizer.token_to_id("<|0.00|>") if with_timestamps else None
        no_timestamps = self.tokenizer.token_to_id("<|notimestamps|>") if with_timestamps else None
        if prompt and previous is None:
            raise ValueError("checkpoint tokenizer lacks the previous-text token")
        if with_timestamps and (timestamp_begin is None or no_timestamps is None):
            raise ValueError("checkpoint tokenizer lacks long-form tokens")
        initial_tokens = self.initial_tokens[:3] if with_timestamps else self.initial_tokens
        prefix = decoder_prefix(
            prompt,
            initial_tokens=initial_tokens,
            previous_token=previous,
            text_positions=self.config.text_positions,
        )
        first_logits = self.logits(prefix, encoder)
        if no_speech is None:
            no_speech_prob = float("nan")
        else:
            speech_logits = first_logits[len(prefix) - len(initial_tokens)].astype(np.float32)
            no_speech_prob = float(
                np.exp(speech_logits[no_speech] - np.logaddexp.reduce(speech_logits))
            )

        def filtered(current: list[int]) -> np.ndarray:
            self._check_cancelled()
            sampled = current[len(prefix) :]
            raw = first_logits[-1] if not sampled else self.logits(current, encoder)[-1]
            next_logits = raw.astype(np.float32, copy=True)
            next_logits[list(self.suppressed)] = -np.inf
            if not sampled:
                next_logits[list(self.begin_suppressed)] = -np.inf
            if with_timestamps:
                next_logits[no_timestamps] = -np.inf
                apply_timestamp_rules(
                    next_logits, sampled, timestamp_begin=timestamp_begin, eot=self.eot
                )
            return next_logits

        chosen = search_candidates(
            prefix,
            filtered,
            eot=self.eot,
            max_tokens=self.config.text_positions // 2,
            context_length=self.config.text_positions,
            beam_size=beam_size,
            temperature=temperature,
            best_of=best_of,
            sample_token=sample_token,
        )
        text_tokens = [token for token in chosen.tokens if token < self.eot]
        text = self.tokenizer.decode(text_tokens, skip_special_tokens=False).strip()
        text_bytes = text.encode("utf-8")
        compression_ratio = len(text_bytes) / len(zlib.compress(text_bytes))
        return DecodeAttempt(
            chosen.tokens,
            text,
            chosen.sum_logprob / (len(chosen.tokens) + 1),
            no_speech_prob,
            compression_ratio,
            temperature,
        )

    def decode_short(
        self, encoder: _EncodedAudio, prompt: list[int], settings: DecodingSettings
    ) -> Transcription:
        if settings.greedy_only:
            return self.greedy_decode(encoder, prompt)
        result = decode_with_fallback(
            lambda temperature, beam_size, best_of: self._decode_attempt(
                encoder,
                prompt,
                with_timestamps=False,
                temperature=temperature,
                beam_size=beam_size,
                best_of=best_of,
            ),
            temperatures=settings.temperatures,
            beam_size=settings.beam_size,
            best_of=settings.best_of,
            compression_ratio_threshold=settings.compression_ratio_threshold,
            logprob_threshold=settings.logprob_threshold,
            no_speech_threshold=settings.no_speech_threshold,
        )
        return Transcription(result.text, result.tokens + [self.eot])

    def decode_long(
        self, encoder: _EncodedAudio, prompt: list[int], settings: DecodingSettings
    ) -> DecodedWindow:
        if settings.greedy_only:
            return self.timestamp_decode(encoder, prompt)
        result = decode_with_fallback(
            lambda temperature, beam_size, best_of: self._decode_attempt(
                encoder,
                prompt,
                with_timestamps=True,
                temperature=temperature,
                beam_size=beam_size,
                best_of=best_of,
            ),
            temperatures=settings.temperatures,
            beam_size=settings.beam_size,
            best_of=settings.best_of,
            compression_ratio_threshold=settings.compression_ratio_threshold,
            logprob_threshold=settings.logprob_threshold,
            no_speech_threshold=settings.no_speech_threshold,
        )
        return DecodedWindow(
            result.tokens, result.avg_logprob, result.no_speech_prob, result.temperature
        )

    def greedy_decode(
        self, encoder: _EncodedAudio, prompt: list[int] | None = None
    ) -> Transcription:
        result = self._greedy_attempt(encoder, prompt or [], score=False)
        return Transcription(result.text, result.tokens + [self.eot])

    def _greedy_attempt(self, encoder, prompt, *, score=True) -> DecodeAttempt:
        previous = self.tokenizer.token_to_id("<|startofprev|>") if prompt else None
        if prompt and previous is None:
            raise ValueError("checkpoint tokenizer lacks the previous-text token")
        prefix = decoder_prefix(
            prompt or [],
            initial_tokens=self.initial_tokens,
            previous_token=previous,
            text_positions=self.config.text_positions,
        )
        result: list[int] = []
        sum_logprob = np.float32(0.0)
        no_speech_prob = float("nan")
        speech_position = len(prefix) - len(self.initial_tokens)
        device_selection = "token" in getattr(getattr(self, "_decoder", None), "outputs", ())
        if device_selection:
            self._clear_prefix_states()
            self._clear_greedy_cache()
        try:
            for step in range(self.config.text_positions // 2):
                self._check_cancelled()
                if device_selection:
                    token, logprob, probability = self._greedy_step(
                        prefix,
                        encoder,
                        first=step == 0,
                        score=score,
                        speech_position=speech_position,
                    )
                    if step == 0:
                        no_speech_prob = probability
                else:
                    raw = self.logits(prefix, encoder)
                    if step == 0 and score:
                        no_speech = self.tokenizer.token_to_id("<|nospeech|>")
                        if no_speech is not None:
                            speech_logits = raw[speech_position].astype(np.float32)
                            no_speech_prob = float(
                                np.exp(
                                    speech_logits[no_speech] - np.logaddexp.reduce(speech_logits)
                                )
                            )
                    next_logits = raw[-1].astype(np.float32, copy=True)
                    next_logits[list(self.suppressed)] = -np.inf
                    if step == 0:
                        next_logits[list(self.begin_suppressed)] = -np.inf
                    token = int(np.argmax(next_logits))
                    logprob = (
                        float(next_logits[token] - np.logaddexp.reduce(next_logits))
                        if score
                        else float("nan")
                    )
                sum_logprob = np.float32(sum_logprob + logprob)
                result.append(token)
                if token == self.eot:
                    break
                prefix.append(token)
                if len(prefix) > self.config.text_positions:
                    result.append(self.eot)
                    break
            else:
                result.append(self.eot)
        finally:
            if device_selection:
                self._clear_greedy_cache()
        text = self.tokenizer.decode(result[:-1], skip_special_tokens=False).strip()
        text_bytes = text.encode("utf-8")
        return DecodeAttempt(
            result[:-1],
            text,
            float(sum_logprob) / len(result),
            no_speech_prob,
            len(text_bytes) / len(zlib.compress(text_bytes)),
            0.0,
        )

    def timestamp_decode(self, encoder: _EncodedAudio, prompt: list[int]) -> DecodedWindow:
        """Decode one window at temperature zero with timestamp tokens enabled."""
        timestamp_begin = self.tokenizer.token_to_id("<|0.00|>")
        previous = self.tokenizer.token_to_id("<|startofprev|>")
        no_speech = self.tokenizer.token_to_id("<|nospeech|>")
        no_timestamps = self.tokenizer.token_to_id("<|notimestamps|>")
        if any(value is None for value in (timestamp_begin, previous, no_speech, no_timestamps)):
            raise ValueError("checkpoint tokenizer lacks long-form tokens")
        prefix = decoder_prefix(
            prompt,
            initial_tokens=self.initial_tokens[:3],
            previous_token=previous,
            text_positions=self.config.text_positions,
        )
        sot_index = len(prefix) - len(self.initial_tokens[:3])
        first_logits = self.logits(prefix, encoder)
        speech_logits = first_logits[sot_index].astype(np.float32)
        no_speech_prob = float(
            np.exp(speech_logits[no_speech] - np.logaddexp.reduce(speech_logits))
        )
        result: list[int] = []
        sum_logprobs = 0.0
        for step in range(self.config.text_positions // 2):
            self._check_cancelled()
            if len(prefix) > self.config.text_positions:
                break
            raw = first_logits[-1] if step == 0 else self.logits(prefix, encoder)[-1]
            next_logits = raw.astype(np.float32, copy=True)
            next_logits[list(self.suppressed)] = -np.inf
            next_logits[no_timestamps] = -np.inf
            if step == 0:
                next_logits[list(self.begin_suppressed)] = -np.inf
            apply_timestamp_rules(
                next_logits, result, timestamp_begin=timestamp_begin, eot=self.eot
            )
            token = int(np.argmax(next_logits))
            sum_logprobs += float(next_logits[token] - np.logaddexp.reduce(next_logits))
            if token == self.eot:
                break
            result.append(token)
            prefix.append(token)
        return DecodedWindow(result, sum_logprobs / (len(result) + 1), no_speech_prob)


def transcribe_fp32(
    checkpoint_dir: Path,
    audio: np.ndarray,
    *,
    initial_prompt: str | None = None,
    beam_size: int = DecodingSettings.beam_size,
    temperature: float | Sequence[float] = DecodingSettings.temperatures,
    best_of: int = DecodingSettings.best_of,
    compression_ratio_threshold: float | None = DecodingSettings.compression_ratio_threshold,
    logprob_threshold: float | None = DecodingSettings.logprob_threshold,
    no_speech_threshold: float | None = DecodingSettings.no_speech_threshold,
) -> Transcription:
    """Transcribe one <=30 s 16 kHz mono waveform for fp32 correctness checks.

    Engine lifetime and reuse are left to the later listenflow engine-contract issue.
    """
    settings = decoding_settings(
        beam_size=beam_size,
        temperature=temperature,
        best_of=best_of,
        compression_ratio_threshold=compression_ratio_threshold,
        logprob_threshold=logprob_threshold,
        no_speech_threshold=no_speech_threshold,
    )
    return _transcribe(
        checkpoint_dir, audio, "fp32", initial_prompt=initial_prompt, settings=settings
    )


def transcribe_fp16(
    checkpoint_dir: Path,
    audio: np.ndarray,
    *,
    initial_prompt: str | None = None,
    beam_size: int = DecodingSettings.beam_size,
    temperature: float | Sequence[float] = DecodingSettings.temperatures,
    best_of: int = DecodingSettings.best_of,
    compression_ratio_threshold: float | None = DecodingSettings.compression_ratio_threshold,
    logprob_threshold: float | None = DecodingSettings.logprob_threshold,
    no_speech_threshold: float | None = DecodingSettings.no_speech_threshold,
) -> Transcription:
    """Transcribe one <=30 s 16 kHz mono waveform for fp16 correctness checks."""
    settings = decoding_settings(
        beam_size=beam_size,
        temperature=temperature,
        best_of=best_of,
        compression_ratio_threshold=compression_ratio_threshold,
        logprob_threshold=logprob_threshold,
        no_speech_threshold=no_speech_threshold,
    )
    return _transcribe(
        checkpoint_dir, audio, "fp16", initial_prompt=initial_prompt, settings=settings
    )


def transcribe_long_fp32(
    checkpoint_dir: Path,
    audio: np.ndarray,
    *,
    initial_prompt: str | None = None,
    beam_size: int = DecodingSettings.beam_size,
    temperature: float | Sequence[float] = DecodingSettings.temperatures,
    best_of: int = DecodingSettings.best_of,
    compression_ratio_threshold: float | None = DecodingSettings.compression_ratio_threshold,
    logprob_threshold: float | None = DecodingSettings.logprob_threshold,
    no_speech_threshold: float | None = DecodingSettings.no_speech_threshold,
) -> LongTranscription:
    """Transcribe a complete 16 kHz mono recording with timed segments."""
    settings = decoding_settings(
        beam_size=beam_size,
        temperature=temperature,
        best_of=best_of,
        compression_ratio_threshold=compression_ratio_threshold,
        logprob_threshold=logprob_threshold,
        no_speech_threshold=no_speech_threshold,
    )
    return _transcribe_long_fp32(
        checkpoint_dir, audio, initial_prompt=initial_prompt, settings=settings
    )


def _transcribe_long_fp32(
    checkpoint_dir: Path,
    audio: np.ndarray,
    *,
    initial_prompt: str | None = None,
    settings: DecodingSettings = _DEFAULT_DECODING_SETTINGS,
    trace: Callable[[WindowTrace], None] | None = None,
    segment_trace: Callable[[SegmentTrace], None] | None = None,
) -> LongTranscription:
    """Run long-form decoding, optionally recording its windows for evaluation."""
    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim != 1 or not np.isfinite(samples).all():
        raise ValueError("audio must be a finite mono waveform")
    with TensorRTWhisper(checkpoint_dir, precision="fp32") as model:
        mel = model._mel_spectrogram_long(samples)
        timestamp_begin = model.tokenizer.token_to_id("<|0.00|>")
        if timestamp_begin is None:
            raise ValueError("checkpoint tokenizer lacks timestamp tokens")
        return transcribe_mel(
            mel,
            content_frames=mel.shape[-1] - 3_000,
            decode=lambda window, prompt: model.decode_long(model.encode(window), prompt, settings),
            trace=trace,
            segment_trace=segment_trace,
            text=lambda tokens: model.tokenizer.decode(tokens, skip_special_tokens=False),
            timestamp_begin=timestamp_begin,
            eot=model.eot,
            audio_positions=model.config.audio_positions,
            initial_prompt_tokens=model.prompt_tokens(initial_prompt),
            no_speech_threshold=settings.no_speech_threshold,
            logprob_threshold=settings.logprob_threshold,
        )


def _transcribe(
    checkpoint_dir: Path,
    audio: np.ndarray,
    precision: str,
    *,
    initial_prompt: str | None = None,
    settings: DecodingSettings = _DEFAULT_DECODING_SETTINGS,
) -> Transcription:
    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim != 1 or samples.size > 480_000 or not np.isfinite(samples).all():
        raise ValueError("audio must be a finite mono waveform of at most 30 seconds")
    with TensorRTWhisper(checkpoint_dir, precision=precision) as model:
        mel = model._mel_spectrogram(samples)
        return model.decode_short(model.encode(mel), model.prompt_tokens(initial_prompt), settings)
