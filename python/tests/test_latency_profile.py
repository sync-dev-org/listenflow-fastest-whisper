"""Behavior checks for the fp16 latency breakdown without a CUDA GPU."""

import csv
import ctypes
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scripts import profile_latency


def _row(name, seconds, subset="test-clean"):
    return {
        "subset": subset,
        "id": name,
        "audio": f"/{name}.flac",
        "reference": "HELLO",
        "frames": str(int(seconds * 16_000)),
    }


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
def test_profile_selects_first_utterances_in_each_band_and_preserves_manifest(tmp_path):
    """The profile selects each band's first inputs in manifest order without changing it."""
    rows = [
        _row("other", 3, "test-other"),
        _row("long", 20),
        _row("short", 4.99),
        _row("five", 5),
        _row("ten", 10),
        _row("thirty", 30),
        _row("too-long", 30.01),
        _row("short-extra", 1),
    ]
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    original = manifest.read_bytes()

    selected = profile_latency.select_utterances(manifest, per_band=1)

    assert [row["id"] for row in selected] == ["long", "short", "five", "ten"]
    assert manifest.read_bytes() == original


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
def test_profile_rejects_insufficient_inputs_and_duplicate_selected_ids(tmp_path):
    """A run fails before using the GPU if a band is incomplete or IDs are ambiguous."""
    manifest = tmp_path / "manifest.csv"
    rows = [_row("same", seconds) for seconds in (1, 5, 10, 20)]
    with manifest.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="duplicate"):
        profile_latency.select_utterances(manifest, per_band=1)
    rows = [_row(str(seconds), seconds) for seconds in (1, 5, 10)]
    with manifest.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="20-30"):
        profile_latency.select_utterances(manifest, per_band=1)


def _measurement(name, frames, mel, encoder, steps, other=0):
    gpu = sum(step["gpu_seconds"] for step in steps)
    host = sum(step["host_seconds"] for step in steps)
    return {
        "id": name,
        "frames": frames,
        "total_seconds": mel + encoder + gpu + host + other,
        "mel_seconds": mel,
        "encoder_seconds": encoder,
        "decoder_gpu_seconds": gpu,
        "decoder_host_seconds": host,
        "other_seconds": other,
        "decoder_steps": steps,
        "generated_tokens": len(steps),
    }


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
def test_profile_reports_component_medians_total_shares_and_pooled_step_medians():
    """Bands report real component totals and pool decoder steps across their utterances."""
    rows = [
        _measurement("one", 16_000, 1, 2, [{"gpu_seconds": 2, "host_seconds": 1}], 1),
        _measurement(
            "two",
            32_000,
            3,
            4,
            [{"gpu_seconds": 4, "host_seconds": 2}, {"gpu_seconds": 8, "host_seconds": 3}],
            1,
        ),
        _measurement("boundary", 5 * 16_000, 1, 1, [{"gpu_seconds": 1, "host_seconds": 1}]),
    ]

    report = profile_latency.summarize(rows)
    short = report["0-5"]

    assert short["count"] == 2
    assert short["total_median_seconds"] == 16
    assert short["components"]["mel"]["median_seconds"] == 2
    assert short["components"]["decoder_gpu"]["median_seconds"] == 7
    assert short["components"]["decoder_gpu"]["share_of_total"] == pytest.approx(14 / 32)
    assert sum(part["share_of_total"] for part in short["components"].values()) == 1
    assert short["decoder_step_gpu_median_seconds"] == 4
    assert short["decoder_step_host_median_seconds"] == 2
    assert short["decoder_steps_median"] == 1.5
    assert report["5-10"]["count"] == 1
    assert report["10-20"]["total_median_seconds"] is None
    markdown = profile_latency.markdown_table(report)
    assert "0-5" in markdown and "43.8%" in markdown


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
def test_profile_rejects_missing_gpu_steps_and_invalid_durations():
    """The profile refuses measurements whose GPU timing or input duration is missing."""
    row = _measurement("one", 16_000, 1, 1, [])
    with pytest.raises(ValueError, match="decoder"):
        profile_latency.summarize([row])
    row = _measurement("one", 480_001, 1, 1, [{"gpu_seconds": 1, "host_seconds": 1}])
    with pytest.raises(ValueError, match="frames"):
        profile_latency.summarize([row])


@pytest.fixture
def cpu_native(monkeypatch):
    """Keep real Whisper and tensoracle objects, replacing only CUDA/TensorRT operations."""
    import tensorrt as trt
    from cuda.bindings import runtime as cudart
    from tensoracle import EngineHandle
    from tokenizers import Tokenizer, models

    from listenflow.fastest_whisper._inference import TensorRTWhisper, _HostIO

    clock = SimpleNamespace(seconds=0.0)
    allocations, events, trace = {}, {}, []
    transfers = []
    graph_state = SimpleNamespace(capture=None, executables={}, captures=0, next_id=1)
    success = cudart.cudaError_t.cudaSuccess

    def allocate(size):
        block = ctypes.create_string_buffer(size)
        pointer = ctypes.addressof(block)
        allocations[pointer] = block
        clock.seconds += 0.0005
        return success, pointer

    def free(pointer):
        del allocations[pointer]
        clock.seconds += 0.0005
        return (success,)

    def copy(destination, source, size, kind, stream):
        transfers.append((kind, size))
        ctypes.memmove(destination, source, size)
        clock.seconds += 0.001
        return (success,)

    def copy2d(destination, dpitch, source, spitch, width, height, kind, stream):
        for row in range(height):
            ctypes.memmove(destination + row * dpitch, source + row * spitch, width)
        return (success,)

    monkeypatch.setattr(cudart, "cudaMemcpy2DAsync", copy2d)
    monkeypatch.setattr(
        cudart,
        "cudaMemsetAsync",
        lambda pointer, value, size, stream: (ctypes.memset(pointer, value, size), (success,))[1],
    )

    def create_event():
        event = len(events) + 100
        events[event] = None
        return success, event

    def record(event, stream):
        events[event] = clock.seconds
        trace.append(("event", stream))
        return (success,)

    def elapsed(start, stop):
        assert events[start] is not None and events[stop] is not None
        return success, (events[stop] - events[start]) * 1000

    def destroy(event):
        del events[event]
        return (success,)

    for name in (
        "cudaSetDevice",
        "cudaStreamSynchronize",
        "cudaStreamDestroy",
        "cudaDeviceSynchronize",
    ):
        monkeypatch.setattr(cudart, name, lambda *args: (success,))
    monkeypatch.setattr(cudart, "cudaStreamCreate", lambda: (success, 42))
    for name, callback in (
        ("cudaMalloc", allocate),
        ("cudaFree", free),
        ("cudaMemcpyAsync", copy),
        ("cudaEventCreate", create_event),
        ("cudaEventRecord", record),
        ("cudaEventElapsedTime", elapsed),
        ("cudaEventDestroy", destroy),
    ):
        monkeypatch.setattr(cudart, name, callback)

    def begin_capture(stream, mode):
        assert graph_state.capture is None
        graph_state.capture = []
        graph_state.captures += 1
        return (success,)

    def end_capture(stream):
        graph = tuple(graph_state.capture)
        graph_state.capture = None
        return success, graph

    def instantiate(graph, flags):
        identifier = graph_state.next_id
        graph_state.next_id += 1
        graph_state.executables[identifier] = graph
        return success, identifier

    def launch(identifier, stream):
        for operation in graph_state.executables[identifier]:
            operation(stream)
        return (success,)

    def destroy_graph(identifier):
        del graph_state.executables[identifier]
        return (success,)

    monkeypatch.setattr(cudart, "cudaStreamBeginCapture", begin_capture)
    monkeypatch.setattr(cudart, "cudaStreamEndCapture", end_capture)
    monkeypatch.setattr(cudart, "cudaGraphInstantiate", instantiate)
    monkeypatch.setattr(cudart, "cudaGraphLaunch", launch)
    monkeypatch.setattr(cudart, "cudaGraphDestroy", lambda graph: (success,))
    monkeypatch.setattr(cudart, "cudaGraphExecDestroy", destroy_graph)

    class Engine:
        def __init__(self, decoder):
            self.decoder = decoder is True
            self.mel = decoder == "mel"
            self.names = (
                ("audio", "log_mel", "mel")
                if self.mel
                else (
                    "tokens",
                    "positions",
                    "mask",
                    "self_keys",
                    "self_values",
                    "cross_keys",
                    "cross_values",
                    "suppression",
                    "logits",
                    "token",
                    "greedy",
                    "present_keys",
                    "present_values",
                )
                if self.decoder
                else ("mel", "encoder", "cross_keys", "cross_values")
            )
            self.num_io_tensors = len(self.names)

        def get_tensor_name(self, index):
            return self.names[index]

        def get_tensor_location(self, name):
            return trt.TensorLocation.DEVICE

        def get_tensor_mode(self, name):
            outputs = (
                ("log_mel", "mel")
                if self.mel
                else ("logits", "token", "greedy", "present_keys", "present_values")
                if self.decoder
                else ("encoder", "cross_keys", "cross_values")
            )
            return trt.TensorIOMode.OUTPUT if name in outputs else trt.TensorIOMode.INPUT

        def is_shape_inference_io(self, name):
            return False

        def get_tensor_dtype(self, name):
            if name in ("tokens", "positions", "token"):
                return trt.int32
            if self.mel:
                return trt.float32
            return trt.float32 if name in ("mel", "mask", "suppression", "greedy") else trt.float16

        def get_tensor_shape(self, name):
            return {
                "audio": (480_400,),
                "log_mel": (128, 3000),
                "mel": (128, 3000) if self.mel else (1, 128, 3000),
                "encoder": (1, 1, 2),
                "tokens": (1, 1),
                "positions": (1, 1),
                "mask": (1, 1, 1, -1),
                "self_keys": (1, 1, 1, -1, 2),
                "self_values": (1, 1, 1, -1, 2),
                "cross_keys": (1, 1, 1, 1, 2),
                "cross_values": (1, 1, 1, 1, 2),
                "present_keys": (1, 1, 1, -1, 2),
                "present_values": (1, 1, 1, -1, 2),
                "logits": (1, 1, 12),
                "token": (1, 1, 1),
                "greedy": (1, 1, 3),
                "suppression": (1, 1, 12),
            }[name]

        def get_tensor_profile_shape(self, name, index):
            if name == "mask":
                return (1, 1, 1, 2), (1, 1, 1, 5), (1, 1, 1, 9)
            return (1, 1, 1, 1, 2), (1, 1, 1, 4, 2), (1, 1, 1, 8, 2)

        def create_execution_context(self):
            return Context(self)

    class Context:
        def __init__(self, engine):
            self.engine, self.shapes, self.addresses = engine, {}, {}

        def set_input_shape(self, name, shape):
            self.shapes[name] = shape
            return True

        def infer_shapes(self):
            return []

        def get_tensor_shape(self, name):
            if name in ("present_keys", "present_values"):
                shape = list(self.shapes["self_keys"])
                shape[3] += 1
                return tuple(shape)
            return self.shapes.get(name, self.engine.get_tensor_shape(name))

        def set_tensor_address(self, name, pointer):
            self.addresses[name] = pointer
            return True

        def execute_async_v3(self, stream):
            if graph_state.capture is not None:
                graph_state.capture.append(self.perform)
                return True
            return self.perform(stream)

        def perform(self, stream):
            trace.append(("enqueue", stream))
            clock.seconds += 0.002
            for name in self.engine.names:
                if self.engine.get_tensor_mode(name) != trt.TensorIOMode.OUTPUT:
                    continue
                output = np.zeros(
                    self.get_tensor_shape(name),
                    dtype=trt.nptype(self.engine.get_tensor_dtype(name)),
                )
                if name == "logits":
                    position = ctypes.c_int.from_address(self.addresses["positions"]).value
                    output[..., 8 if position < 4 else model.eot] = 10
                if name in ("token", "greedy"):
                    position = ctypes.c_int.from_address(self.addresses["positions"]).value
                    scores = np.zeros(12, dtype=np.float32)
                    scores[8 if position < 4 else model.eot] = 10
                    suppression = np.ctypeslib.as_array(
                        (ctypes.c_float * 12).from_address(self.addresses["suppression"])
                    )
                    token = np.argmax(scores + suppression)
                    if name == "token":
                        output[...] = token
                    else:
                        output[...] = [token, model._test_logprob, model._test_silence]
                ctypes.memmove(self.addresses[name], output.ctypes.data, output.nbytes)
            return True

    model = TensorRTWhisper.__new__(TensorRTWhisper)
    model.device = 0
    model.precision = "fp16"
    model.config = SimpleNamespace(
        n_mels=128,
        text_positions=8,
        text_layers=1,
        text_heads=1,
        audio_state=2,
        vocab_size=12,
    )
    model.initial_tokens, model.eot = [1, 2, 3, 4], 5
    model.suppressed, model.begin_suppressed = {7}, {5}
    model.tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "hello": 8}, unk_token="[UNK]"))
    model._encoder = _HostIO(EngineHandle(Engine(False), object(), Path("encoder.engine")))
    model._decoder = _HostIO(EngineHandle(Engine(True), object(), Path("decoder.engine")))
    model._test_transfers = transfers
    model._test_logprob = -0.01
    model._test_silence = 0.1
    model._test_graph_state = graph_state
    model._test_mel_factory = lambda: _HostIO(
        EngineHandle(Engine("mel"), object(), Path("mel.engine"))
    )
    yield model, cudart, lambda: clock.seconds, events, trace
    model.close()
    assert not allocations
    assert not graph_state.executables


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
def test_profile_times_real_greedy_steps_and_brackets_enqueue_on_its_stream(cpu_native):
    """Timing preserves decoded text and counts EOT while separating GPU from host work."""
    from tensoracle import EngineExecution

    model, cudart, clock, events, trace = cpu_native
    original = EngineExecution.enqueue
    model.greedy_decode(model.encode(np.zeros((128, 3000), dtype=np.float32)))
    trace.clear()
    with profile_latency.DecoderProfiler(model, cudart, clock=clock) as profiler:
        row = profiler.measure(_row("one", 1), np.zeros(16_000, dtype=np.float32))
        second = profiler.measure(_row("two", 1), np.zeros(16_000, dtype=np.float32))
        assert events

    assert row["output"] == "hello"
    assert row["tokens"] == [8, 5]
    assert row["generated_tokens"] == len(row["decoder_steps"]) == 2
    assert row["decoder_enqueues"] == 5
    assert row["decoder_gpu_seconds"] == pytest.approx(0.010)
    assert [step["gpu_seconds"] for step in row["decoder_steps"]] == pytest.approx([0.008, 0.002])
    assert all(step["host_seconds"] > 0 for step in row["decoder_steps"])
    assert second["decoder_gpu_seconds"] == pytest.approx(row["decoder_gpu_seconds"])
    assert (
        trace
        == [
            ("enqueue", 42),
            *[("event", 42), ("enqueue", 42), ("event", 42)] * 5,
        ]
        * 2
    )
    assert EngineExecution.enqueue is original
    assert "logits" not in model.__dict__
    assert not events
    profile_latency.summarize([row, second])


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
def test_profile_restores_inference_and_destroys_events_after_failed_enqueue(
    cpu_native, monkeypatch
):
    """An enqueue error leaves no profiler hooks or CUDA events behind."""
    from tensoracle import EngineExecution

    model, cudart, clock, events, _ = cpu_native
    original = EngineExecution.enqueue
    context_type = type(model._decoder.execution._context)
    execute = context_type.execute_async_v3
    monkeypatch.setattr(
        context_type,
        "execute_async_v3",
        lambda context, stream: False if context.engine.decoder else execute(context, stream),
    )
    with (
        pytest.raises(RuntimeError, match="enqueue failed"),
        profile_latency.DecoderProfiler(model, cudart, clock=clock) as profiler,
    ):
        profiler.measure(_row("one", 1), np.zeros(16_000, dtype=np.float32))
    assert EngineExecution.enqueue is original
    assert "logits" not in model.__dict__
    assert not events


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
def test_profile_measures_the_gpu_audio_features_used_by_inference(cpu_native):
    """The profile includes native mel execution when the model supplies its GPU audio path."""
    model, cudart, clock, _, trace = cpu_native
    model._mel = model._test_mel_factory()
    model.greedy_decode(model.encode(model._mel_spectrogram(np.zeros(16_000, dtype=np.float32))))
    trace.clear()
    with profile_latency.DecoderProfiler(model, cudart, clock=clock) as profiler:
        row = profiler.measure(_row("gpu-mel", 1), np.zeros(16_000, dtype=np.float32))
    assert row["tokens"] == [8, 5]
    assert sum(operation == "enqueue" for operation, _ in trace) == 7
    assert row["decoder_enqueues"] == 5
    assert row["mel_seconds"] > 0


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
def test_greedy_reads_only_the_selected_token_from_each_decoder_execution(cpu_native):
    """Greedy decoding preserves tokens while each GPU-to-host transfer contains one token ID."""
    model, cudart, _, _, _ = cpu_native
    encoder = model.encode(np.zeros((128, 3000), dtype=np.float32))
    model._test_transfers.clear()
    result = model.greedy_decode(encoder)
    copied = [
        size
        for kind, size in model._test_transfers
        if kind == cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
    ]
    assert result.tokens == [8, 5]
    assert copied == [4] * 5


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
def test_default_first_attempt_keeps_greedy_tokens_and_only_copies_scalar_results(cpu_native):
    """An accepted first attempt matches greedy and transfers no vocabulary-sized logits."""
    from listenflow.fastest_whisper._decoding import DecodingSettings

    model, cudart, _, _, _ = cpu_native
    encoder = model.encode(np.zeros((128, 3000), dtype=np.float32))
    expected = model.greedy_decode(encoder)
    model._test_transfers.clear()
    actual = model.decode_short(encoder, [], DecodingSettings())

    assert actual == expected
    copied = [
        size
        for kind, size in model._test_transfers
        if kind == cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
    ]
    assert copied == [12] * 5


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
@pytest.mark.parametrize("failure", ["logprob", "compression", "silence"])
def test_default_retries_failed_quality_checks_and_stops_on_silence(
    cpu_native, failure, monkeypatch
):
    """Poor scores or repeated text trigger hotter attempts with five candidates; silence ends retries."""
    from tokenizers import Tokenizer, models

    from listenflow.fastest_whisper._decoding import DecodingSettings
    from listenflow.fastest_whisper._inference import TensorRTWhisper

    model, _, _, _, _ = cpu_native
    generator = np.random.default_rng(2026)
    monkeypatch.setattr(np.random, "default_rng", lambda: generator)
    if failure == "compression":
        model.eot = 10
        model.tokenizer = Tokenizer(
            models.WordLevel({"[UNK]": 0, "hello" * 100: 8}, unk_token="[UNK]")
        )
    else:
        model._test_logprob = -3.0
        model._test_silence = 0.7 if failure == "silence" else 0.1
    attempts = []

    def observe(frame, event, _argument):
        if event == "call" and frame.f_code is TensorRTWhisper._decode_attempt.__code__:
            arguments = frame.f_locals
            attempts.append(
                (arguments["temperature"], arguments["beam_size"], arguments["best_of"])
            )

    previous_profile = sys.getprofile()
    sys.setprofile(observe)
    try:
        result = model.decode_short(
            model.encode(np.zeros((128, 3000), dtype=np.float32)), [], DecodingSettings()
        )
    finally:
        sys.setprofile(previous_profile)

    assert result.tokens[-1] == model.eot
    expected = [(0.0, None, None)]
    if failure == "logprob":
        expected += [(0.2, None, 5)]
    elif failure == "compression":
        expected += [(temperature, None, 5) for temperature in (0.2, 0.4, 0.6, 0.8, 1.0)]
    assert attempts == expected


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
def test_repeated_audio_reuses_captured_decoder_work_and_close_returns_graphs(cpu_native):
    """Repeated decoding replays captured work without recapture and closure releases its graphs."""
    model, _, _, _, _ = cpu_native
    mel = np.zeros((128, 3000), dtype=np.float32)
    first = model.greedy_decode(model.encode(mel))
    captures = model._test_graph_state.captures
    second = model.greedy_decode(model.encode(mel))
    assert first.tokens == second.tokens == [8, 5]
    assert captures > 0
    assert model._test_graph_state.captures == captures
    model.close()
    assert not model._test_graph_state.executables


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
def test_profile_writes_machine_readable_report_and_rejects_incomplete_run(tmp_path):
    """A report retains selection and measurement conditions and requires every selected ID."""
    selected = [_row(str(seconds), seconds) for seconds in (1, 5, 10, 20)]
    rows = [
        _measurement(row["id"], int(row["frames"]), 1, 2, [{"gpu_seconds": 2, "host_seconds": 1}])
        for row in selected
    ]
    metadata = {"gpu_exclusive": False, "gpu_access": "排他なし", "precision": "fp16"}
    with pytest.raises(ValueError, match="selected"):
        profile_latency.write_report(tmp_path, selected, rows[:-1], metadata)
    assert not (tmp_path / "report.json").exists()

    profile_latency.write_report(tmp_path, selected, rows, metadata)

    result = json.loads((tmp_path / "report.json").read_text())
    assert result["metadata"] == metadata
    assert result["selected_utterance_ids"] == [row["id"] for row in selected]
    assert result["bands"]["20-30"]["count"] == 1
    markdown = (tmp_path / "report.md").read_text()
    assert "排他なし" in markdown and "20-30" in markdown
    with pytest.raises(FileExistsError):
        profile_latency.write_report(tmp_path, selected, rows, metadata)


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
def test_profile_reports_interpolated_p95_for_components_and_decoder_steps():
    """The slow tail is reported separately for utterances and pooled decoder steps."""
    rows = [
        _measurement("one", 16_000, 1, 2, [{"gpu_seconds": 2, "host_seconds": 1}], 1),
        _measurement(
            "two",
            32_000,
            3,
            4,
            [{"gpu_seconds": 4, "host_seconds": 2}, {"gpu_seconds": 8, "host_seconds": 3}],
            1,
        ),
    ]
    short = profile_latency.summarize(rows)["0-5"]
    assert short["total_p95_seconds"] == pytest.approx(24.1)
    assert short["components"]["mel"]["p95_seconds"] == pytest.approx(2.9)
    assert short["decoder_step_gpu_p95_seconds"] == pytest.approx(7.6)
    assert short["decoder_step_host_p95_seconds"] == pytest.approx(2.9)
    assert profile_latency.summarize(rows)["5-10"]["total_p95_seconds"] is None


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
def test_profile_uses_product_transcription_and_records_executed_cache_buckets(cpu_native):
    """Product transcription retains its text, tokens and actual padded cache work."""
    from listenflow.fastest_whisper._cached_model import CachedWhisper

    model, cudart, clock, _, _ = cpu_native
    # Reuse the real native model and all its state as the cached product subclass.
    model.__class__ = CachedWhisper
    model._mel = model._test_mel_factory()
    audio = np.zeros(16_000, dtype=np.float32)
    expected = model.transcribe(audio, beam_size=1, temperature=0, best_of=1)
    with profile_latency.DecoderProfiler(model, cudart, clock=clock) as profiler:
        row = profiler.measure(_row("product", 1), audio)
    assert row["output"] == " ".join(segment.text for segment in expected)
    assert row["tokens"] == [8, 5]
    assert row["entrypoint"] == "CachedWhisper.transcribe"
    assert [item["bucket"] for item in row["decoder_buckets"]] == [1, 8, 8, 8, 8]
    assert sum(item["gpu_seconds"] for item in row["decoder_buckets"]) == pytest.approx(
        row["decoder_gpu_seconds"]
    )


@pytest.mark.req("SP-LFFASTESTWHISPER-001")
def test_timeline_preserves_product_output_and_closes_capture_after_failure(
    cpu_native, monkeypatch
):
    """CUDA tracing keeps real transcription and closes its capture if native execution fails."""
    from scripts import profile_timeline

    from listenflow.fastest_whisper._cached_model import CachedWhisper

    model, cudart, _, _, _ = cpu_native
    model.__class__ = CachedWhisper
    model._mel = model._test_mel_factory()
    sample = (_row("trace", 1), np.zeros(16_000, dtype=np.float32))
    captures = []
    success = cudart.cudaError_t.cudaSuccess
    monkeypatch.setattr(
        cudart, "cudaProfilerStart", lambda: (captures.append("start"), (success,))[1]
    )
    monkeypatch.setattr(
        cudart, "cudaProfilerStop", lambda: (captures.append("stop"), (success,))[1]
    )
    rows = profile_timeline.capture_transcriptions(model, [sample], cudart)
    assert rows[0]["id"] == "trace"
    assert rows[0]["output"] == "hello"
    assert captures == ["start", "stop"]
    monkeypatch.setattr(
        cudart, "cudaMemcpyAsync", lambda *args: (cudart.cudaError_t.cudaErrorUnknown,)
    )
    with pytest.raises(RuntimeError, match="cudaMemcpyAsync"):
        profile_timeline.capture_transcriptions(model, [sample], cudart)
    assert captures == ["start", "stop", "start", "stop"]


@pytest.mark.req("SP-LFFASTESTWHISPER-011")
def test_layer_profiling_releases_its_context_without_detaching_a_null_profiler(
    cpu_native, monkeypatch
):
    """Layer diagnostics leave product contexts intact when TensorRT rejects a null profiler."""
    from scripts import profile_layers

    model, _, _, _, _ = cpu_native
    io = model._encoder
    original_execution = io.execution
    context_type = type(io.execution._context)

    def set_profiler(context, profiler):
        if profiler is None:
            raise ValueError("TensorRT forbids null profiler")
        context._test_profiler = profiler

    monkeypatch.setattr(
        context_type,
        "profiler",
        property(lambda context: getattr(context, "_test_profiler", None), set_profiler),
        raising=False,
    )
    monkeypatch.setattr(context_type, "enqueue_emits_profile", True, raising=False)

    def report(context):
        context.profiler.report_layer_time("(Unnamed Layer* 1) [Matrix Multiply]", 2.0)
        return True

    monkeypatch.setattr(context_type, "report_to_profiler", report, raising=False)
    engine_type = type(io.execution._handle.engine)
    monkeypatch.setattr(
        engine_type,
        "create_engine_inspector",
        lambda engine: SimpleNamespace(get_engine_information=lambda format: '{"Layers": []}'),
        raising=False,
    )
    report = profile_layers.profile_io(
        io, {"mel": np.zeros((1, 128, 3000), dtype=np.float32)}, iterations=2
    )
    assert report["total_mean_ms"] == 2
    assert io.execution is original_execution
    assert original_execution._context.profiler is None
