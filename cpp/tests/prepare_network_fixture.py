"""Prepare native network inputs without CUDA, or obtain a 1.x GPU reference.

Run with the existing Python project's dependencies. The C++ build never invokes
Python. Reference mode requires a tensoracle 1.x checkout and a GPU reservation.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("prepare", "reference", "compare", "statistics")
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--precision", choices=("fp32", "fp16", "fp8"), default="fp32")
    parser.add_argument(
        "--graph", choices=("mel", "encoder", "decoder", "sampled"), default="mel"
    )
    parser.add_argument("--batch", type=int, default=5)
    parser.add_argument("--history", type=int, default=1)
    parser.add_argument("--plugin", type=Path)
    parser.add_argument("--native-result", type=Path)
    parser.add_argument(
        "--encoder-reference",
        type=Path,
        help="Use cross-KV from fixed audio instead of random cross-KV",
    )
    args = parser.parse_args()
    if args.mode == "compare" and args.precision != "fp32" and args.graph != "mel":
        parser.error("fp16/fp8 tensor values are diagnostic; use statistics mode")
    from listenflow.fastest_whisper import _fp8
    from listenflow.fastest_whisper._audio import _FILTERS
    from listenflow.fastest_whisper._checkpoint import Checkpoint
    from tokenizers import Tokenizer

    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    checkpoint = Checkpoint(args.checkpoint.resolve())
    c = checkpoint.config
    suffix = f"{args.graph}-{args.precision}-b{args.batch}-h{args.history}"
    input_path = root / f"{suffix}-inputs.safetensors"
    if args.mode == "prepare":
        # Whitespace is a valid JSON change and gives this fixture an isolated
        # identity in the DEFAULT tensoracle cache without changing cache paths.
        config = json.loads((checkpoint.directory / "config.json").read_text())
        config["_native_test_fixture"] = str(root)
        (root / "config.json").write_text(json.dumps(config, sort_keys=True))
        with np.load(_FILTERS) as filters:
            save_file(
                {f"mel_{c.n_mels}": filters[f"mel_{c.n_mels}"]},
                root / "mel.safetensors",
            )
        tokenizer = Tokenizer.from_file(str(checkpoint.directory / "tokenizer.json"))
        files = (
            [checkpoint.directory / "model.safetensors"]
            if checkpoint.weight_map is None
            else [
                checkpoint.directory / f
                for f in sorted(set(checkpoint.weight_map.values()))
            ]
        )
        scales = json.loads(_fp8._SCALES.read_text())
        # This text is a TEST protocol, not a model/config parser in the library.
        paths = [
            root / "config.json",
            checkpoint.directory / "tokenizer.json",
            _fp8._SCALES,
            _FILTERS,
            root / "mel.safetensors",
            args.plugin or "",
        ]
        quote = lambda value: json.dumps(str(value))
        lines = [" ".join(str(getattr(c, name)) for name in c.__dataclass_fields__)]
        lines += [quote(p) for p in paths]
        lines += [
            f"{tokenizer.token_to_id('<|nospeech|>') if tokenizer.token_to_id('<|nospeech|>') is not None else -1} {tokenizer.token_to_id('<|0.00|>') if tokenizer.token_to_id('<|0.00|>') is not None else -1}"
        ]
        lines += [str(len(files)), *(quote(p) for p in files), str(len(scales))]
        lines += [f"{quote(k)} {float(np.float32(v)):.9g}" for k, v in scales.items()]
        (root / "source.txt").write_text("\n".join(lines) + "\n")
        dtype = np.float32 if args.precision == "fp32" else np.float16
        b = args.batch if args.graph == "sampled" else 1
        if args.graph == "mel":
            import wave

            fixture = (
                Path(__file__).resolve().parents[2]
                / "python/tests/fixtures/small_boat_16khz_mono_s16le.wav"
            )
            with wave.open(str(fixture)) as wav:
                audio = (
                    np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(
                        np.float32
                    )
                    / 32768
                )
            inputs = {
                "audio": np.pad(
                    np.pad(audio, (0, 480000 - len(audio))), (200, 200), mode="reflect"
                )
            }
        elif args.graph == "encoder":
            oracle = (
                Path(__file__).resolve().parents[2]
                / "python/tests/fixtures/oracle/small_boat_16khz_mono_s16le.npz"
            )
            with np.load(oracle) as data:
                inputs = {"mel": data["mel"][None]}
        else:
            # The same explicit token/KV input is used for both implementations.
            # Random KV values isolate graph parity from encoder rounding.
            rng = np.random.default_rng(6112)
            head_dim = c.audio_state // c.text_heads
            self_shape = (c.text_layers, b, c.text_heads, args.history, head_dim)
            cross_shape = (c.text_layers, 1, c.text_heads, c.audio_positions, head_dim)
            inputs = {
                "tokens": np.full(
                    (b, 1), tokenizer.token_to_id("<|startoftranscript|>"), np.int32
                ),
                "positions": np.full((b, 1), args.history - 1, np.int32),
                "mask": np.zeros((1, 1, 1, args.history + 1), np.float32),
                "self_keys": rng.normal(0, 0.1, self_shape).astype(dtype),
                "self_values": rng.normal(0, 0.1, self_shape).astype(dtype),
                "cross_keys": rng.normal(0, 0.1, cross_shape).astype(dtype),
                "cross_values": rng.normal(0, 0.1, cross_shape).astype(dtype),
                "suppression": np.zeros((b, 1, c.vocab_size), np.float32),
            }
            # Initial decoding uses one dummy cache position, masked exactly as
            # _inference.py logits() does. A saved encoder reference ties decoder
            # comparisons to the same fixed audio used in the encoder test.
            inputs["mask"][..., 0] = -np.inf
            if args.history == 1:
                inputs["self_keys"].fill(0)
                inputs["self_values"].fill(0)
            if args.encoder_reference is not None:
                encoded = load_file(args.encoder_reference)
                for name in ("cross_keys", "cross_values"):
                    if encoded[name].shape != cross_shape:
                        raise ValueError(f"encoder reference shape differs for {name}")
                    inputs[name] = encoded[name].astype(dtype, copy=False)
            if args.graph == "sampled":
                inputs |= {
                    "temperature": np.full((1, 1, 1), 0.4, np.float32),
                    "uniform": rng.random((b, 1, 1), dtype=np.float32),
                    "timestamps": np.ones((1, 1, 1), bool),
                }
        save_file({k: np.ascontiguousarray(v) for k, v in inputs.items()}, input_path)
        print(root / "source.txt", input_path)
    elif args.mode == "reference":
        import time

        import tensorrt as trt
        from listenflow.fastest_whisper._engines import (
            _open_mel_engine,
            open_whisper_engine,
        )
        from listenflow.fastest_whisper._inference import _HostIO

        reference_checkpoint = root / "reference-checkpoint"
        reference_checkpoint.mkdir(exist_ok=True)
        (reference_checkpoint / "config.json").write_bytes(
            (root / "config.json").read_bytes()
        )
        for entry in checkpoint.directory.iterdir():
            if entry.name.startswith("model") or entry.name == "tokenizer.json":
                target = reference_checkpoint / entry.name
                if not target.exists():
                    target.symlink_to(entry)
        checkpoint = Checkpoint(reference_checkpoint)
        logger = trt.Logger(trt.Logger.ERROR)
        start = (
            time.perf_counter() if os.environ.get("LFFW_NETWORK_MEASURE_TIME") else None
        )
        if args.graph == "mel":
            handle = _open_mel_engine(
                checkpoint, precision=args.precision, logger=logger
            )
        else:
            handle = open_whisper_engine(
                checkpoint,
                decoder=args.graph != "encoder",
                precision=args.precision,
                logger=logger,
                sample_batch=args.batch if args.graph == "sampled" else None,
            )
        if start is not None:
            print("reference_open_seconds", time.perf_counter() - start, flush=True)
        (root / f"{suffix}-reference-engine-path.txt").write_text(str(handle.path))
        if os.environ.get("LFFW_NETWORK_MEASURE_MEMORY"):
            print(
                "shared_scratch_bytes", handle.engine.device_memory_size_v2, flush=True
            )
        io = _HostIO(handle)
        try:
            outputs = io.run(load_file(input_path))
            save_file(
                {k: np.ascontiguousarray(v) for k, v in outputs.items()},
                root / f"{suffix}-reference.safetensors",
            )
        finally:
            io.close()
    else:
        reference = load_file(root / f"{suffix}-reference.safetensors")
        fp32 = (
            load_file(
                root
                / f"{args.graph}-fp32-b{args.batch}-h{args.history}-reference.safetensors"
            )
            if args.mode == "statistics"
            else None
        )
        report = {
            "relative_denominator_epsilon": float(np.finfo(np.float32).eps),
            "tensors": {},
        }
        metadata = json.loads((args.native_result / "outputs.json").read_text())
        for name, expected in reference.items():
            info = metadata[name]
            actual = np.fromfile(
                args.native_result / f"{name}.bin", dtype=info["dtype"]
            ).reshape(info["shape"])
            if name in ("mel", "log_mel"):
                atol, rtol = 3e-5, 3e-5
            elif name in ("encoder", "cross_keys", "cross_values"):
                atol, rtol = 2e-2, 5e-3
            else:
                atol, rtol = 2e-2, 2e-2
            if args.mode == "statistics":
                is_integer = np.issubdtype(expected.dtype, np.integer)

                def stats(
                    left,
                    right,
                    abs_bound=atol,
                    rel_bound=rtol,
                    integer=is_integer,
                ):
                    left, right = left.astype(np.float32), right.astype(np.float32)
                    delta = abs(left - right)
                    finite = np.isfinite(delta)
                    eps = np.finfo(np.float32).eps
                    if integer:
                        abs_bound, rel_bound = 0.0, 0.0
                    bounds = abs_bound + rel_bound * abs(right)
                    return {
                        "max_abs": float(np.max(delta[finite]))
                        if finite.any()
                        else None,
                        "mean_abs": float(np.mean(delta[finite], dtype=np.float64))
                        if finite.any()
                        else None,
                        "mean_relative": float(
                            np.mean(
                                (delta / (abs(right) + eps))[finite], dtype=np.float64
                            )
                        )
                        if finite.any()
                        else None,
                        "outside_diagnostic_fp32_bounds": int(
                            np.count_nonzero(finite & (delta > bounds))
                        ),
                        "diagnostic_atol": abs_bound,
                        "diagnostic_rtol": rel_bound,
                        "unequal_nonfinite": int(
                            np.count_nonzero(
                                ~finite
                                & ~(np.isnan(left) & np.isnan(right))
                                & (left != right)
                            )
                        ),
                    }

                report["tensors"][name] = {
                    "v1_vs_fp32": stats(expected, fp32[name]),
                    "native_vs_fp32": stats(actual, fp32[name]),
                    "native_vs_v1": stats(actual, expected),
                }
                if name in ("greedy", "sampled"):
                    report["tensors"][name]["token_mismatches"] = {
                        "v1_vs_fp32": int(
                            np.count_nonzero(expected[..., 0] != fp32[name][..., 0])
                        ),
                        "native_vs_fp32": int(
                            np.count_nonzero(actual[..., 0] != fp32[name][..., 0])
                        ),
                        "native_vs_v1": int(
                            np.count_nonzero(actual[..., 0] != expected[..., 0])
                        ),
                    }
                continue
            if name in ("greedy", "sampled"):
                # Token IDs must match exactly even though the fused result
                # stores them as float32. Never apply relative error to IDs.
                np.testing.assert_array_equal(actual[..., 0], expected[..., 0])
            if np.issubdtype(expected.dtype, np.integer):
                np.testing.assert_array_equal(actual, expected)
            else:
                np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)
            print(
                name,
                actual.shape,
                "max_abs",
                float(
                    np.nanmax(
                        np.abs(actual.astype(np.float32) - expected.astype(np.float32))
                    )
                ),
            )
        if args.mode == "statistics":
            output = root / f"{suffix}-statistics.json"
            output.write_text(json.dumps(report, indent=2))
            print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
