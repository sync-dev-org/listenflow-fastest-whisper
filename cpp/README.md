# Build the fc1 and development cubin plugins

This directory builds caller-owned Triton kernels into TensorRT plugin libraries.
`liblffw_fc1_plugin.so` registers only `LffwFp16Fc1` version `1`. The fp16
Whisper decoder loads it from its Python package in an installed wheel, or from
`build/cubin-plugin/` during source development. With
`BUILD_TESTING=ON`, `liblffw_cubin_plugin.so` additionally registers the bias and
QKV development fixtures. Both targets compile the same fc1 creator and kernel
source. All creators use namespace `listenflow.fastest_whisper`.

The bias plugin uses version `1`; the QKV plugin uses version `2`.
`LffwFp16Bias` adds one serialized
fp16 scalar bias to a dynamic one-dimensional fp16 tensor. The
input and output use TensorRT's LINEAR format. Supported lengths are 1 through
`INT32_MAX`; the integration tests cover 1 through 4097, including partial blocks.
The Whisper networks do not use the bias or QKV fixtures. The Python wheel
distributes only the production fc1 library.

The bias plugin has exactly two fields: `bundle_sha256` (CHAR, 64 bytes) and `bias`
(FLOAT16, one element). It uses tensoracle's `PluginBase` and `PluginIO` for the
TensorRT V3 lifecycle, `DeviceBytes` for bias storage, and `Kernel` for the selected
GPU's cubin. The bundle digest binds the serialized fields to the embedded kernel.

## Single-token QKV projection

`LffwFp16Qkv` calculates the decoder self-attention's three projections in one
kernel launch. Version `2` takes four LINEAR fp16 inputs:

| Input | Shape | Layout |
|---|---|---|
| features | `(1, 1, 1280)` | One decoder token |
| weight | `(3840, 1280)` | Contiguous Q, K, V rows, each `[output, input]` |
| Q bias | `(1280,)` | Q output bias |
| V bias | `(1280,)` | V output bias |

Its only output is a LINEAR fp16 tensor of shape `(1, 1, 3840)`, containing Q,
then K, then V along the last axis. Other shapes and dtypes are rejected. There
is no K bias, and the plugin needs no workspace. Its only field is
`bundle_sha256` (CHAR, 64 bytes), the ASCII bundle digest. Missing, incompatible,
and additional fields, including version 1's weight fields, are rejected.

The graph supplies weight and biases using `network.add_constant`. Packing Q,
K, and V into contiguous rows happens during graph construction; inference needs
no layout conversion. TensorRT owns and serializes these constants in the engine.
The plugin borrows their device pointers for each enqueue and holds no weight or
bias bytes on the host or GPU. Execution contexts of the same engine use the same
constant device addresses. Separate engines retain their own constants; no global
resource registry or weight-content hash is used. Shape and execution state stay
local to each TensorRT execution context.

Each plugin retains its own `Kernel` objects. TensorRT may recreate the plugin
through its creator when making an execution context, so module loading can grow
with the context count. Each module is released by its plugin in the owning CUDA
context after execution has finished. The constants live with TensorRT's engine
and execution contexts; callers must finish GPU work before closing those objects.
Runtime pointers are never serialized.

Version 1 stored weights in fields and is incompatible with version 2. The
development library registers only QKV version 2. A legacy plan fails through
tensoracle's existing engine deserialization or execution-context creation error
path, instead of being executed with another input layout. TensorRT 11.4 can defer
plugin creation until the first execution context, so reading plan metadata alone
does not establish compatibility. Rebuild an old engine explicitly for version 2.

On a cache hit, tensoracle still uses `weights=(source_file,)` to identify build
conditions, so that file must remain available for the identity check. It does not
call the graph or materialize the constants again. A fresh-process test verifies
reuse and identical outputs. Inference does not import Triton, torch, or the
producer.

The kernel pads the reduction to 2048, computes four output rows per block, and
accumulates products and Q/V bias in fp32 before one fp16 output conversion.
Keeping one contiguous output avoids multiple-output lifecycle overrides in the
shared plugin helper. The decoder can recover its three `(1, 1, 1280)` tensors
using slices starting at 0, 1280 and 2560. TensorRT constants keep weights out of the
caller-supplied execution I/O while making them available to the plugin.

The correctness tests use deterministic random `(1280, 1280)` weights for each
projection and independent NumPy fp32 matrix multiplication followed by fp16
conversion. They allow `rtol=2^-10` (the fp16 relative spacing bound, at most two
ULPs depending on the significand) and
`atol=2^-12` near cancellation. For these test inputs, a fp32 reduction error bound
`gamma_12 * sum(abs(x*w), abs(bias))` is below `1.86e-5`; the NumPy fp32 reference
differs from a float64 calculation by at most `1.64e-6`. Their sum is below the
absolute allowance. Here `gamma_n = n*u/(1-n*u)` and `u=2^-24`; twelve steps cover
the eleven-level padded tree and the nonzero bias addition. Zero and basis-vector
inputs must match exactly, checking unbiased K, Q/V biases, weight layout and
projection order independently of the relative tolerance.

## Single-token fc1 and erf GELU

`LffwFp16Fc1` version `1` combines the decoder's MLP expansion and exact GELU.
It accepts three LINEAR fp16 inputs: one token `(1, 1, 1280)`, a contiguous
weight matrix `(5120, 1280)` in `[output, input]` order, and bias `(5120,)`.
The LINEAR fp16 output has shape `(1, 1, 5120)`. Its only serialized field is
`bundle_sha256` (CHAR, 64 bytes), read from the library's exported bundle identity.
TensorRT owns and serializes the weight and bias constants. The plugin borrows
their device pointers and allocates no weight storage or workspace.

The kernel computes four output rows per block, with a padded 2048-wide fp32
reduction. It rounds the dot product to fp16 before adding bias, and rounds each
edge of `0.5*x*(1+erf(x/sqrt(2)))` to fp16. These operations and rounding match
the adopted prototype. Tests use actual large-v3 weights and an independent
NumPy float64/math.erf reference with the same fp16 edge conversions, allowing
`rtol=2^-10` and `atol=2^-11` for reduction and erf error near rounding boundaries.
Directed cancellation inputs must match exactly and distinguish rounding before
bias from one final conversion. The positive tests call the product graph's
`decoder_fc1`; incompatible field, shape and dtype cases exercise the creator.

The product replaces all 32 fc1 + GELU layers only in the fp16 decoder.
The fp32 decoder, both encoders and log-mel graph keep their standard layers.
The selected device must have an sm86, sm89 or sm120 cubin; no other device or
precision is selected on failure. A missing native library stops fp16 loading.
No configuration option disables the plugin.

`open_engine(..., plugins=(library,))` includes the library's SHA-256 in the
engine's build conditions. The decoder graph version remains `5-fp16` because
that existing condition already separates the plugin graph from older engines.
There is no second library digest in the graph version or a runtime manifest.
Saved-engine recovery also probes an execution context, since TensorRT can
defer detection of incompatible plugins until context creation. An unusable
saved engine is rebuilt once automatically. A failed rebuild is returned to
the caller. Product builder settings and timing-cache behavior are unchanged.

## Prerequisites

Use a standalone checkout on Linux x86_64, including WSL2. Install Python
`>=3.12,<3.14`, uv, a C++17 compiler, CMake 3.20 or later, Ninja, the CUDA toolkit,
and binutils (`nm`, `readelf`). CMake and Ninja are build tools installed on the
host; they are not Python project dependencies. These commands use CUDA at
`/usr/local/cuda`; pass `--cuda-root` for another toolkit location.

Install the Python tooling from the repository root, keeping the environment in
`python/.venv`:

```bash
(cd python && uv sync --group cubin --locked)
```

The lock supplies tensoracle 1.1.0, Triton 3.8.0, ptxas 12.9.86 and cuobjdump
13.3.73. Cubin production uses tensoracle's pinned NVIDIA wheel tools, rather
than the toolkit's ptxas, and does not use a GPU. A fresh producer process must
not import torch. The build finds tensoracle's C++ helper inside the installed
Python package; it does not require a tensoracle source checkout.

## Prepare and verify the build SDK

The fixed inputs below follow [tensoracle's SDK procedure](https://github.com/sync-dev-org/tensoracle/blob/v1.1.0/docs/tensorrt-sdk.md).
SDK 11.2.1.2 has published digests for the exact NVIDIA wheel, its extracted
library, and all 13 headers. The application's runtime remains TensorRT
11.4.0.106 from `uv.lock`; the GPU integration tests check this SDK/runtime
combination for this plugin. Matching the major SONAME alone does not verify a
matching SDK. An alternate SDK requires independently verified inputs and the
same build and runtime checks.

Choose a new cache directory outside the repository. The following `mkdir`
intentionally fails for an existing directory; verify existing inputs instead of
overwriting them. Stop on any download, extraction or verification failure.

```bash
set -e
SDK_WORK="${XDG_CACHE_HOME:-$HOME/.cache}/tensoracle-sdk-11.2.1.2"
mkdir -p "$(dirname "$SDK_WORK")"
mkdir "$SDK_WORK"
mkdir -p "$SDK_WORK/sdk/include" "$SDK_WORK/sdk/lib"
cp cpp/tensorrt-sdk.sha256 "$SDK_WORK/SHA256SUMS"
export TENSORRT_ROOT="$SDK_WORK/sdk"
```

Download and check the wheel before extracting its library:

```bash
(
  cd "$SDK_WORK"
  WHEEL_NAME=tensorrt_cu13_libs-11.2.1.2-py3-none-manylinux_2_28_x86_64.whl
  curl --fail --location \
    "https://pypi.nvidia.com/tensorrt-cu13-libs/$WHEEL_NAME" --output "$WHEEL_NAME"
  head -n 1 SHA256SUMS | sha256sum --check
  OSS_COMMIT=1dade062a4e796c14ab6b3f32461ad694ec58951
  while read -r expected relative_path; do
    case "$relative_path" in
      sdk/include/*)
        header="${relative_path#sdk/include/}"
        curl --fail --location \
          "https://raw.githubusercontent.com/NVIDIA/TensorRT/$OSS_COMMIT/include/$header" \
          --output "$relative_path"
        ;;
    esac
  done < SHA256SUMS
  unzip -p "$WHEEL_NAME" tensorrt_libs/libnvinfer.so.11 > sdk/lib/libnvinfer.so.11
  sha256sum --check SHA256SUMS
  ln -s libnvinfer.so.11 sdk/lib/libnvinfer.so
)
```

All 15 digests must report `OK`. Check the resolved header version and SONAME:

```bash
c++ -E -P -x c++ -I "$TENSORRT_ROOT/include" - <<'CPP'
#include "NvInferVersion.h"
NV_TENSORRT_MAJOR
NV_TENSORRT_MINOR
NV_TENSORRT_PATCH
NV_TENSORRT_BUILD
CPP
readelf -d "$TENSORRT_ROOT/lib/libnvinfer.so.11" | rg '\(SONAME\)'
```

The four version lines must be `11`, `2`, `1`, `2`, and the SONAME must be
`libnvinfer.so.11`. Keep the checksums with the SDK for subsequent verification.

## Produce and build

Run these commands from `python/` with the verified `TENSORRT_ROOT`:

```bash
uv run --group cubin --locked python ../cpp/generate_cubin_bundle.py \
  --output-dir ../build/cubin-bundle
uv run --group cubin --locked python ../cpp/build_native.py \
  --bundle-dir ../build/cubin-bundle --build-dir ../build/cubin-plugin \
  --tensorrt-root "$TENSORRT_ROOT"
ctest --test-dir ../build/cubin-plugin --output-on-failure --no-tests=error
```

`compile_bundle` produces all three kernels for each of sm86, sm89 and sm120 (nine
cubins), a `bundle.json`
manifest, and `bundle.cpp` containing exactly those bytes. The generator prints
the bundle digest. Generated bundles and the shared library belong under
`build/` and are ignored by Git. The CPU test generates the bundle twice in fresh
processes and compares every output byte, including the embedded C++ source.

The build script checks the input bundle, resolves the installed tensoracle C++
helper, invokes CMake and Ninja, and checks that the resulting
`build/cubin-plugin/liblffw_fc1_plugin.so` and the development library embed the
same bundle. CTest checks both libraries'
`getCreators`, `setLoggerFinder`, and the exported bundle identity with `nm`, and
uses `readelf` to restrict direct ELF dependencies to the CUDA driver, TensorRT
11, and the GNU C/C++ runtime. CTest does not invoke Python or use a GPU.

To invoke CMake directly after producing the bundle, run the following from the
repository root. Resolve the installed helper using the Python environment first;
CMake itself does not discover Python or nanobind:

```bash
TENSORACLE_CPP_ROOT="$(cd python && uv run --locked python -c \
  'from pathlib import Path; import tensoracle; print(Path(tensoracle.__file__).parent / "cpp")')"
cmake -S cpp -B build/cubin-plugin -G Ninja \
  -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release -DCUDAToolkit_ROOT=/usr/local/cuda \
  -DTENSORACLE_CPP_ROOT="$TENSORACLE_CPP_ROOT" -DTENSORRT_ROOT="$TENSORRT_ROOT" \
  -DLFFW_BUNDLE_ROOT="$(pwd)/build/cubin-bundle"
cmake --build build/cubin-plugin --parallel 2
ctest --test-dir build/cubin-plugin --output-on-failure --no-tests=error
```

## Validate engines

The CPU suite includes bundle reproduction and, when `TENSORRT_ROOT` is set, a
native build and its CTest checks. The full suite also imports the evaluation
libraries, so install every dependency group. Run pytest from `python/`:

```bash
uv run --all-groups --locked pytest -q -m 'not gpu'
```

After coordinating use of the shared GPU, run the small plugin checks:

```bash
LFFW_USE_GPU=1 uv run --group cubin --locked pytest -q \
  tests/test_cubin_plugin.py tests/test_qkv_plugin.py -m gpu
```

For fc1 and the product graph checks, supply the large-v3 checkpoint as well:

```bash
LFFW_USE_GPU=1 LFFW_CHECKPOINT_DIR=/absolute/path/to/whisper-large-v3 \
  uv run --all-groups --locked pytest -q \
  tests/test_fc1_plugin.py tests/test_fc1_product_gpu.py -m gpu
```

The product checks inspect the assembled graphs, compare the product fc1 graph
to NumPy, observe the selected cubin during warmup and CUDA graph capture, and
cancel decoding before another token starts. Recognition must work again after
cancellation. After closing every execution context, CUDA module load/unload
pointers must balance without plugin-owned weight allocations.

These tests need a verified SDK and a supported GPU, and do not need a Whisper
checkpoint. They check `open_engine(..., plugins=(library,))` and
`build_serialized_engine(..., plugins=(library,))`, fp16 results against an
independent NumPy calculation (exact for the bias kernel; within the documented
tolerance for random QKV inputs and exact for zero and basis inputs), invalid plugin fields, and saved-engine reuse in
a fresh process whose graph callback must never run. Each runtime process checks
that torch and Triton were not imported and runs with absent SDK paths. GPU test
engines and build files use the repository's temporary-directory cleanup rules.
The QKV tests additionally check the 1280-wide projections, reject incompatible
bundle fields and feature/constant shapes or dtypes, and reuse serialized QKV constants.
They grow one TensorRT execution context into four, close and replace one, and
check all outputs against the independent reference. The plugin must upload zero
weight bytes; all contexts must pass the same weight and bias addresses to its
kernel. Device-wide memory and host RSS are also sampled. After the engine and all
contexts close, module load/unload pointers must balance.

With `BUILD_TESTING` enabled (the CMake default), the build also produces
`liblffw_qkv_driver_observer.so` and `liblffw_qkv_v1_fixture.so`. The GPU tests use
`LD_PRELOAD` to load the separate observer into their subprocesses. It forwards
real CUDA driver calls and records calls originating in the production plugin,
including the constant pointers supplied to the QKV kernel. The production
library has no counter interface or observer dependency. The frozen version-1
fixture creates legacy plans in a separate process for the rejection test; it is
never loaded by the version-2 runtime. Configure with `-DBUILD_TESTING=OFF` when
only the production plugin library is wanted.

For a minimal graph construction example, see `graph_for` in
`python/tests/qkv_plugin_runtime.py`; its matching test data is in
`python/tests/test_qkv_plugin.py`. Generate and build the library with the commands above
before using its creator in another graph. Performance measurements of this
isolated layer do not establish product decoder latency or transcription quality.

## Build a Python distribution

The scikit-build-core backend builds `lffw_fc1_plugin` with testing disabled and
installs only `liblffw_fc1_plugin.so` in `listenflow/fastest_whisper/`. It needs
the verified SDK, tensoracle helper, generated bundle, and CUDA toolkit. It adds
no Python binding. From `python/`, set the CMake inputs and build:

```bash
export TENSORACLE_CPP_ROOT="$(uv run --locked python -c \
  'from pathlib import Path; import tensoracle; print(Path(tensoracle.__file__).parent / "cpp")')"
export LFFW_BUNDLE_ROOT="$(realpath ../build/cubin-bundle)"
# TENSORRT_ROOT remains the verified SDK prepared above.
uv build --out-dir ../dist
uv run --all-groups --locked pytest -q -m 'not gpu'
```

The distribution tests need these same inputs. The source archive carries
`cpp/`, the Python package and CMake entry point, README, and LICENSE. Its
extracted directory can build a wheel with the same verified SDK, tensoracle
helper, and generated bundle; no sibling checkout is needed. Generated cubins,
engine plans, test audio, and development dependencies are absent from the wheel.

An editable `uv sync` registers the Python source without compiling C++. Its
source-only lookup uses the independently built `build/cubin-plugin/` library.
This keeps C++ builds independent and avoids scikit-build-core 1.1.1's editable
namespace finder hiding the existing `listenflow` core package. An installed
package always prefers its own library. A missing required fc1 library fails
instead of selecting standard fp16 layers.

Local builds have a `py3-none-linux_x86_64` tag. Release builds follow
[RELEASE.md](RELEASE.md) to build in the pinned manylinux container, inspect all
ELF dependencies, and test the final wheel outside the source checkout.

The shared inspector's rejection of missing or corrupt cubins, false SM claims,
and another bundle's library is covered by tensoracle's
`test_inspector_distinguishes_missing_corrupt_and_false_sm`,
`test_inspector_detects_plugin_from_another_bundle`, and
`test_inspector_checks_embedded_cubin_bytes_in_plugin` in
[`python/tests/test_bundle.py`](https://github.com/sync-dev-org/tensoracle/blob/v0.5.1/python/tests/test_bundle.py).
The shared runtime also checks cubin digests and the actual device's SM before
loading a module. This repository uses those checks directly rather than
implementing another inspector or launcher.
