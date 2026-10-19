"""Caller-owned Triton inputs; inference loads cubins without importing this module."""

import triton
import triton.language as tl
from tensoracle.producer import KernelSpec
from triton.language.extra.cuda import libdevice


@triton.jit
def bias_fp16(x_ptr, bias_ptr, out_ptr, count, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(x_ptr + offset, offset < count, 0).to(tl.float32)
    bias = tl.load(bias_ptr).to(tl.float32)
    tl.store(out_ptr + offset, value + bias, offset < count)


@triton.jit
def qkv_fp16(
    x_ptr,
    weight_ptr,
    q_bias_ptr,
    v_bias_ptr,
    out_ptr,
    WIDTH: tl.constexpr,
    ROWS: tl.constexpr,
    DEPTH: tl.constexpr,
):
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    column = tl.arange(0, DEPTH)
    hidden = tl.load(x_ptr + column, column < WIDTH, 0).to(tl.float32)
    weight = tl.load(
        weight_ptr + row[:, None] * WIDTH + column[None, :],
        (row[:, None] < 3 * WIDTH) & (column[None, :] < WIDTH),
        0,
    ).to(tl.float32)
    result = tl.sum(weight * hidden[None, :], axis=1)
    q_bias = tl.load(q_bias_ptr + row, row < WIDTH, 0).to(tl.float32)
    v_bias = tl.load(v_bias_ptr + row - 2 * WIDTH, (row >= 2 * WIDTH) & (row < 3 * WIDTH), 0).to(
        tl.float32
    )
    tl.store(out_ptr + row, result + q_bias + v_bias, row < 3 * WIDTH)


@triton.jit
def fc1_fp16(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    WIDTH: tl.constexpr,
    OUTPUT: tl.constexpr,
    ROWS: tl.constexpr,
    DEPTH: tl.constexpr,
):
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    column = tl.arange(0, DEPTH)
    x = tl.load(x_ptr + column, column < WIDTH, 0).to(tl.float32)
    w = tl.load(
        weight_ptr + row[:, None] * WIDTH + column[None, :],
        (row[:, None] < OUTPUT) & (column[None, :] < WIDTH),
        0,
    ).to(tl.float32)
    # _Graph.linear and every GELU edge produce half tensors. In particular,
    # bias must be added after rounding the dot, even when TRT fuses the layer.
    dot = tl.sum(w * x[None, :], axis=1).to(tl.float16).to(tl.float32)
    bias = tl.load(bias_ptr + row, row < OUTPUT, 0).to(tl.float32)
    value = (dot + bias).to(tl.float16).to(tl.float32)
    inv_sqrt2 = tl.full((), 0.7071067811865476, tl.float16).to(tl.float32)
    scaled = (value * inv_sqrt2).to(tl.float16).to(tl.float32)
    erf = libdevice.erf(scaled).to(tl.float16).to(tl.float32)
    gate = (1.0 + erf).to(tl.float16).to(tl.float32)
    half = (value * 0.5).to(tl.float16).to(tl.float32)
    tl.store(out_ptr + row, half * gate, row < OUTPUT)


def kernel_specs() -> tuple[KernelSpec, ...]:
    return (
        KernelSpec(
            "bias_fp16",
            bias_fp16,
            (("x_ptr", "*fp16"), ("bias_ptr", "*fp16"), ("out_ptr", "*fp16"), ("count", "i32")),
            {"BLOCK": 128},
        ),
        KernelSpec(
            "qkv_fp16",
            qkv_fp16,
            (
                ("x_ptr", "*fp16"),
                ("weight_ptr", "*fp16"),
                ("q_bias_ptr", "*fp16"),
                ("v_bias_ptr", "*fp16"),
                ("out_ptr", "*fp16"),
            ),
            {"WIDTH": 1280, "ROWS": 4, "DEPTH": 2048},
        ),
        KernelSpec(
            "fc1_fp16",
            fc1_fp16,
            (
                ("x_ptr", "*fp16"),
                ("weight_ptr", "*fp16"),
                ("bias_ptr", "*fp16"),
                ("out_ptr", "*fp16"),
            ),
            {"WIDTH": 1280, "OUTPUT": 5120, "ROWS": 4, "DEPTH": 2048},
        ),
    )
