"""Caller-owned CUDA C++ inputs; inference only loads the generated cubins."""

from pathlib import Path

from tensoracle.producer import KernelSpec


def kernel_specs() -> tuple[KernelSpec, ...]:
    source = Path(__file__).with_suffix(".cu").read_text(encoding="utf-8")
    return (
        KernelSpec(
            name="bias_fp16",
            source=source,
            signature=(
                ("x_ptr", "*fp16"),
                ("bias_ptr", "*fp16"),
                ("out_ptr", "*fp16"),
                ("count", "i32"),
            ),
            constexprs={"BLOCK": 128},
            entry_symbol="bias_fp16",
        ),
        KernelSpec(
            name="qkv_fp16",
            source=source,
            signature=(
                ("x_ptr", "*fp16"),
                ("weight_ptr", "*fp16"),
                ("q_bias_ptr", "*fp16"),
                ("v_bias_ptr", "*fp16"),
                ("out_ptr", "*fp16"),
            ),
            constexprs={"WIDTH": 1280, "ROWS": 4, "DEPTH": 2048},
            entry_symbol="qkv_fp16",
        ),
        KernelSpec(
            name="fc1_fp16",
            source=source,
            signature=(
                ("x_ptr", "*fp16"),
                ("weight_ptr", "*fp16"),
                ("bias_ptr", "*fp16"),
                ("out_ptr", "*fp16"),
            ),
            constexprs={"WIDTH": 1280, "OUTPUT": 5120, "ROWS": 4, "DEPTH": 2048},
            entry_symbol="fc1_fp16",
        ),
    )
