# Build the fc1 and development cubin plugins

This directory builds caller-owned CUDA C++ kernels into TensorRT plugin libraries.
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

On a cache hit, tensoracle uses `builder_options.weights` to identify build
conditions, so that file must remain available for the identity check. It does not
call the graph or materialize the constants again. Inference does not import
Triton, torch, or the producer.

The kernel pads the reduction to 2048 and computes four output rows per block
(one warp per row). It accumulates products and Q/V bias in fp32 before one fp16
output conversion.
Keeping one contiguous output avoids multiple-output lifecycle overrides in the
shared plugin helper. The decoder can recover its three `(1, 1, 1280)` tensors
using slices starting at 0, 1280 and 2560. TensorRT constants keep weights out of the
caller-supplied execution I/O while making them available to the plugin.

The correctness tests use deterministic random `(1280, 1280)` weights for each
projection and an independent CPU double dot product followed by fp16
conversion. They allow `rtol=2^-10` (the fp16 relative spacing bound, at most two
ULPs depending on the significand) and
`atol=2^-12` near cancellation, preserving the earlier projection tests' limits.
The padded tree has eleven fp32 addition levels, followed by Q/V bias addition.
Zero and basis-vector
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

The kernel computes four output rows per block (one warp per row), with a padded
2048-wide fp32 reduction. It rounds the dot product to fp16 before adding bias,
and rounds each
edge of `0.5*x*(1+erf(x/sqrt(2)))` to fp16. These operations and rounding match
the adopted prototype. Tests use actual large-v3 weights and an independent
CPU double/std::erf reference with the same fp16 edge conversions, allowing
`rtol=2^-10` and `atol=2^-11` for reduction and erf error near rounding boundaries.
Directed cancellation inputs must match exactly and distinguish rounding before
bias from one final conversion. Native numerical tests call the plugin's
`enqueue`; incompatible field, shape and dtype cases exercise the creator and
TensorRT V3 lifecycle interfaces.

The product replaces all 32 fc1 + GELU layers only in the fp16 decoder.
The fp32 decoder, both encoders and log-mel graph keep their standard layers.
The selected device must have an sm86, sm89 or sm120 cubin; no other device or
precision is selected on failure. A missing native library stops fp16 loading.
No configuration option disables the plugin.

The C++ builder's `tensoracle::builder_options::plugins` includes the library's
SHA-256 in the engine's build conditions. The decoder graph version remains
`5-fp16` because
that existing condition already separates the plugin graph from older engines.
There is no second library digest in the graph version or a runtime manifest.
Saved-engine recovery also probes an execution context, since TensorRT can
defer detection of incompatible plugins until context creation. An unusable
saved engine is rebuilt once automatically. A failed rebuild is returned to
the caller. Product builder settings and timing-cache behavior are unchanged.

## Prerequisites

Use a standalone checkout on Linux x86_64, including WSL2. Install Python
`>=3.12,<3.14`, uv, GCC 15, CMake 3.26 or later, Ninja, the NVIDIA driver,
and binutils (`nm`, `readelf`). CMake and Ninja are build tools installed on the
host; they are not Python project dependencies. Native CUDA headers, nvcc and
`libcudart_static.a` come from the locked Conan toolkit; an OS-installed CUDA
Toolkit is not needed. The native build also links the driver's `libcuda.so` (on WSL2, in
`/usr/lib/wsl/lib`). The driver library must also be available on native Linux.

Install the Python tooling from the repository root, keeping the environment in
`python/.venv`:

```bash
(cd python && uv sync --no-install-project --group dev --group cubin --locked)
uv tool install conan==2.34.0
cmake -P cpp/conan/prepare-tests.cmake
. build/conan-prepare/conanbuild.sh
. build/conan-prepare/conanrun.sh
```

The Conan CLI version must match `[build-system].requires` in
`python/pyproject.toml`. Both CMake entry points use `cpp/conanfile.py`,
`cpp/conan.lock` and the GCC 15 profile in `cpp/conan/profiles/`. The bootstrap
gets `cuda-toolkit/13.2.2`, `tensoracle/2.6.0` and `tensorrt/11.2.1.2` from the
private `sync-dev-org/conan-recipes` Release `v1.66.0`
(`conan-recipes-v1.66.0.tar.gz`). A host `gh` login with read access is
required when the locked recipe is absent from the cache. Conan downloads the
GitHub API asset with `Accept: application/octet-stream`, verifies its SHA-256,
and the bootstrap checks the exported recipe revision against `cpp/conan.lock`.
The helper passes `gh auth token` directly to a mode-0600 `source_credentials.json`
in a temporary Conan home outside the checkout, removes it immediately after the
download on success or failure, and withholds the download child's diagnostics.
The token is absent from command arguments and build output. Package installation
uses Conan's default storage. Tensoracle and TensorRT are host requirements.
CUDA is a tool requirement with `nvrtc=True` and `cuobjdump=True`. The toolkit supplies CUDA 13.2.86's compiler,
NVRTC, ptxas, cuobjdump, NVVM, headers, and the static runtime used by
`liblffw_runtime.so` and `liblffw_fc1_plugin.so`.

Native builds with tests enabled use only prepared dependencies, with no recipe
export, download or package build. Run `prepare-tests.cmake` once before testing;
it also accepts `-DLFFW_CONAN_PROFILE=/absolute/path/to/profile` before `-P`.
For a build without tests, the bootstrap prepares missing dependencies itself.
`-DLFFW_CONAN_OFFLINE=ON` restricts that build to prepared dependencies too.
The host profile replaces dependency requests for CMake with the host's CMake.
Start a new build directory when migrating a pre-Conan CMake cache. A configure
with no build type uses the Conan preset's build type. Reconfiguring repeats
installation with the same lock; an enclosing project or a different supplied
toolchain retains responsibility for its own dependency installation.

The default Python `dev` group supplies `tensoracle[tensorrt]>=2.6.0` for tests
and scripts that use TensorRT's Python binding; the distributed package leaves
the TensorRT runtime to the host. The `cubin` group supplies tensoracle without
its `producer` extra and installs no CUDA build packages from PyPI.
Source `build/conan-prepare/conanbuild.sh`
in the shell that produces cubins or runs their Python tests; it exports Conan's
`CUDAToolkit_ROOT`. Pass that root explicitly to the generator so
`tensoracle.producer.compile_bundle` uses the Conan toolkit for every CUDA input.
Cubin production does not use a GPU. A fresh producer process must not import
torch. Native compilation uses the C++ SDK obtained through Conan; the Python
wheel supplies no headers or libraries to the native build.

## SDK inputs and runtime libraries

The bootstrap verifies the recipe archive SHA-256 and every exported recipe
revision against `cpp/conan.lock`. The recipes verify the tensoracle SDK and
TensorRT headers and libraries against their recorded hashes. Tensoracle is a
prebuilt shared SDK: its inference library is not compiled locally. The plugin
helper compiles the adapter source shipped specifically for that purpose in the
SDK, together with this repository's plugin sources and cubin bundle.

CMake uses `tensoracle::tensoracle` and `TensorRT::nvinfer` from the generated
Conan packages. Both standalone and Python builds use the same SDKs and lock;
no `TENSORACLE_CPP_ROOT` or external `TENSORRT_ROOT` is needed.

The host runtime requirement is TensorRT 11.2.1.2 or newer with ABI 11. Install it
through NVIDIA's CUDA apt repository as described in the
[installation guide](https://github.com/sync-dev-org/listenflow-fastest-whisper#install).
An NVIDIA driver is required; an OS CUDA Toolkit is not. Source tests can use
`build/conan-prepare/conanrun.sh`, which supplies the complete prepared TensorRT
runtime, including builder resources, and the tensoracle SDK library directory.
Installed wheels instead locate the tensoracle wheel relative to their own
library directory and leave TensorRT to the host loader.

## Produce and build

Run these commands from `python/` after preparing Conan dependencies:

```bash
uv run --no-sync --group cubin --locked python ../cpp/generate_cubin_bundle.py \
  --cuda-root "$CUDAToolkit_ROOT" --output-dir ../build/cubin-bundle
uv run --no-sync --group cubin --locked python ../cpp/build_native.py \
  --bundle-dir ../build/cubin-bundle --build-dir ../build/cubin-plugin
ctest --test-dir ../build/cubin-plugin --output-on-failure --no-tests=error -LE gpu
```

`compile_bundle` produces all three kernels for each of sm86, sm89 and sm120 (nine
cubins), a `bundle.json`
schema-version-2 manifest, and `bundle.cpp` containing exactly those bytes.
`kernels.py` declares the CUDA source in `kernels.cu`, the `extern "C"` entry
symbols, ordered public signatures and compile constants. The v2 launch ABI
has no hidden Triton parameters or scratch pointers. The generator prints
the bundle digest. Generated bundles and the shared library belong under
`build/` and are ignored by Git. The CPU test generates the bundle twice in fresh
processes and compares every output byte, including the embedded C++ source.

The build script checks the input bundle, invokes CMake with the Conan SDKs and
Ninja, and checks
that the resulting
`build/cubin-plugin/liblffw_fc1_plugin.so` and the development library embed the
same bundle. CTest checks both libraries'
`getCreators`, `setLoggerFinder`, and the exported bundle identity with `nm`, and
uses `readelf` to restrict direct ELF dependencies to the CUDA driver, TensorRT
11, and the GNU C/C++ runtime. CPU CTest also loads the creators and checks their
names, versions, fields, serialization ownership, clones, shapes and dtypes.
The v2 C++ inspector checks embedded bytes and rejects corrupt, missing and
falsely labelled SM images. These checks run without Python or CUDA initialization.
The native test executables link the same Conan tensoracle SDK; CMake discovers
no Python interpreter.

To invoke CMake directly after producing the bundle, run from the repository root:

```bash
cmake -S cpp -B build/cubin-plugin -G Ninja \
  -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release \
  -DLFFW_BUNDLE_ROOT="$PWD/build/cubin-bundle"
cmake --build build/cubin-plugin --parallel 2
ctest --test-dir build/cubin-plugin --output-on-failure --no-tests=error -LE gpu
```

The generated Conan toolchain sets `CUDAToolkit_ROOT` and
`CMAKE_CUDA_COMPILER`. Select another host profile with
`-DLFFW_CONAN_PROFILE=/absolute/path/to/profile`. To link a driver installed outside
the usual Linux library directories, pass its directory in `CMAKE_LIBRARY_PATH`.

## Validate kernels and plugins

Run the CPU checks from the repository root after generating and building the
bundle. `-LE gpu` excludes all GPU-labelled tests, including during Python's
native-build checks:

```bash
ctest --test-dir build/cubin-plugin --output-on-failure --no-tests=error -LE gpu
(cd python && uv run --no-sync --group cubin --locked pytest -q tests/test_cubin_plugin.py -m 'not gpu')
```

The Python producer check regenerates the bundle in fresh processes and compares
all bytes, including the manifest and embedding source. It checks schema 2,
the three supported SMs and all ordered public signatures. The native-build tests
use the prepared Conan SDKs. The isolated build test hides
`/usr/local` and clears toolkit variables to check the prepared Conan toolkit.

After coordinating use of the shared sm120 GPU, run:

```bash
ctest --test-dir build/cubin-plugin --output-on-failure -L gpu
```

The native tests call the actual plugin's `enqueue` and compare outputs with an
independent CPU double/IEEE-binary16 reference. Bias covers lengths 1, 127, 128,
129 and 4097. QKV covers random, zero and basis inputs and the contiguous Q/K/V
layout. Fc1 covers random inputs and exact cancellation sentinels that distinguish
rounding the dot before bias from a single final conversion. Output guards catch
writes past the requested count or output rows. The QKV and fc1 tolerances are
those documented above. CUDA is initialized only by tests labelled `gpu`.

Native engine scenarios additionally build the same small graphs through
`tensoracle::open_engine`. Separate processes reopen the saved engine with a
callback that fails if called; the tests compare the engine bytes, path and
outputs. QKV and fc1 create four execution contexts, close one, check the
survivors and a replacement, then discard the caller's engine handle while the
last execution remains usable. The driver observer requires identical constant
pointers, no plugin-owned weight allocation or upload, and five balanced module
load/unload pairs in the owning CUDA context. Graph probes reject incompatible
feature and constant shapes and dtypes. A raw serialized QKV version-1 plan is
first run with the legacy fixture, then rejected in a fresh process registering
only the current library. Bias also exercises raw TensorRT serialization and
deserialization in addition to the cached C++ entry.

Run only these engine scenarios with:

```bash
ctest --test-dir build/cubin-plugin --output-on-failure -L gpu -R 'engine_'
```

Each scenario's supervisor creates one private directory under the checkout's
`.nf/tmp/ctest-plugin/`, sets `XDG_CACHE_HOME` only in its child processes, and
removes the directory on success, failure or a child timeout. CPU checks exercise
cleanup after a real failed child and a timed-out child, and validate the
safetensors/case fixtures without initializing CUDA. This test isolation leaves
the caller's environment and normal engine/uv caches unchanged. The supervisor
also terminates the child process group when interrupted by SIGTERM or SIGINT.

CTest supplies the Conan SDK library directories for native tests. The
`LFFW_TEST_TENSORRT_LIBRARY_DIR` option selects the runtime for the supervised
plugin-engine scenarios; its default is the prepared TensorRT SDK's `lib/`.

To add the actual checkpoint's layer-0 fc1 weights, prepare a small safetensors
file with `weight` (FLOAT16, `(5120,1280)`) and `bias` (FLOAT16, `(5120,)`) in a
test directory. The checkpoint is a read-only input. For the unsharded checkpoint,
run from `python/`:

```bash
uv run --no-sync --group cubin --locked python - /path/to/whisper-large-v3/model.safetensors \
  /path/to/test-directory/fc1.safetensors <<'PY'
import sys
import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file
with safe_open(sys.argv[1], framework="numpy") as weights:
    arrays = {
        key: weights.get_tensor(f"model.decoder.layers.0.fc1.{key}").astype(np.float16)
        for key in ("weight", "bias")
    }
save_file(arrays, sys.argv[2])
PY
```

Reconfigure the existing native build with
`-DLFFW_FC1_TEST_WEIGHTS=/path/to/test-directory/fc1.safetensors`, rebuild, and run
`ctest --test-dir build/cubin-plugin --output-on-failure -R lffw_fc1_checkpoint_values`.
This adds real-weight random, zero/basis and directed cancellation inputs around
the GELU origin and minimum. Temporary assets belong outside the checkpoint.

The same native test executable can compare an earlier Triton fc1 cubin with the
new CUDA C++ fc1 cubin after reserving the GPU for performance work:

```bash
build/cubin-plugin/lffw_kernel_gpu benchmark /path/to/old-fc1.cubin \
  build/cubin-bundle/cubins/sm120/fc1_fp16.cubin
```

Both kernels receive identical deterministic half inputs, weights and bias and
use grid 1280 / block 128. The earlier Triton ABI receives its two null scratch
arguments and 64 dynamic shared-memory bytes; the new ABI has four public
arguments and no dynamic shared memory. The program checks both outputs against
the independent reference, warms both kernels, alternates their order across
15 pairs of CUDA-event measurements with 200 launches each, and prints median
microseconds plus register, local-memory and shared-memory attributes. It imposes
no new performance threshold. Isolated layer timings need product inference and
transcription checks before they can establish decoder latency or quality.

With `BUILD_TESTING=ON`, the build retains the separate driver observer and the
frozen QKV version-1 library for native engine-integration checks. Production
libraries have no observer dependency. `BUILD_TESTING=OFF` builds only the
production fc1 plugin.

## Build a Python distribution

The scikit-build-core backend installs exactly three native files under
`listenflow/fastest_whisper/`: `_native.abi3.so`, `liblffw_runtime.so`, and
`liblffw_fc1_plugin.so`. The inference library links the shared Conan tensoracle
SDK. The fc1 plugin uses that SDK's plugin helper and adapter, without adding an
unused tensoracle dependency. Neither target embeds tensoracle's static library.
The wheel contains no SDK headers, CMake packages, tensoracle or TensorRT shared
libraries, or link stubs. CUDA runtime is linked statically.

From `python/`, after preparing the bundle and Conan dependencies:

```bash
uv sync --locked --no-install-project --group evaluation --group oracle --group cubin
export LFFW_BUNDLE_ROOT="$PWD/../build/cubin-bundle"
uv build --out-dir ../dist
uv pip install --no-deps --reinstall-package listenflow-fastest-whisper ../dist/*.whl
uv run --locked --no-sync pytest -q tests/test_distribution.py
```

The source archive carries `cpp/`, the Python package and CMake entry point,
README, and LICENSE, so it can build without a sibling checkout. It is an
internal release build input and is not published. The build requirements supply
nanobind and the pinned Conan CLI as wheels; all C/C++ SDK inputs use Conan.
The generated link maps and CTest verify that both production libraries link
`libcudart_static.a` without a direct `libcudart.so` dependency. Unused static
runtime archive members need not appear in the ELF files.

The Python namespace extends listenflow core. Use `uv sync --no-install-project`,
install a wheel into the venv, and run Python commands with `uv run --no-sync`.
After syncing dependencies or changing source, rebuild and reinstall the wheel.
The root README's native extension steps also prepare the source and diagnostic
builds required by the complete CPU test suite.
Installed wheels use `$ORIGIN/../../tensoracle/cpp/lib` from the inference library
to reach the tensoracle wheel, and `$ORIGIN` from the extension to reach the
inference library. No installed RUNPATH points to TensorRT or a build host.

Local wheels have a `cp312-abi3-linux_x86_64` tag. Release builds follow
[RELEASE.md](RELEASE.md) for the pinned manylinux container and final ELF checks.

The shared inspector's rejection of missing or corrupt cubins, false SM claims,
and another bundle's library is covered by tensoracle's
`test_inspector_distinguishes_missing_corrupt_and_false_sm`,
`test_inspector_detects_plugin_from_another_bundle`, and
`test_inspector_checks_embedded_cubin_bytes_in_plugin` in
[`python/tests/test_bundle.py`](https://github.com/sync-dev-org/tensoracle/blob/v0.5.1/python/tests/test_bundle.py).
The shared runtime also checks cubin digests and the actual device's SM before
loading a module. This repository uses those checks directly rather than
implementing another inspector or launcher.

## Tensoracle 2 model library

### Native inference library

The default C++ build includes `liblffw_runtime.so`. Its C ABI is declared in
`include/listenflow/whisper_runtime.h`; `whisper_runtime.hpp` provides move-only
ownership and converts statuses into C++ exceptions. Neither header includes
tensoracle or CUDA types. The Python extension uses this C++ header.

Prepare Conan dependencies and the bundle as described above, then build without
a Python environment:

```bash
cmake -S cpp -B build/runtime -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON -DLFFW_BUNDLE_ROOT="$PWD/build/cubin-bundle"
cmake --build build/runtime --parallel 2
ctest --test-dir build/runtime --output-on-failure -LE gpu
```

CMake discovers the Conan SDK once for all native targets and CTest supplies its
runtime paths. The standalone build does not discover a Python environment.
Set `LFFW_BUILD_RUNTIME=OFF` for a plugin-only or network-only build.

Open options borrow strings and arrays for the call and copy the values they
retain. Suppression token lists are supplied once: native code keeps normal and
initial-token masks on the device. Decoder calls process one token, or one
position across the selected sampling batch. The Python caller can therefore
check cancellation between calls. GPU completion uses event polling with a
50-microsecond sleep and restores the waiting thread's timer slack afterward.

Each model serializes its own operations and returns `LFFW_BUSY` on overlapping
calls. A handle must not be destroyed concurrently with a call using that same
handle. Closing a model waits for submitted work and releases graphs before
executions and buffers. Child handles then reject inference; they can still be
destroyed. Host output arrays remain owned by their callers. Destroying a null
handle succeeds, and a separately owned diagnostic remains valid until
`lffw_error_destroy`.

The CPU CTests cover C/C++ consumers, the exact 21 exported C functions, timer
slack restoration, retained buffer leases, and every decoder cache bucket.
These tests do not establish GPU numerical correctness or performance. Runtime
symbols use hidden visibility, an explicit linker version script, and
`--exclude-libs,ALL`; embedded backend and network symbols are private.

### Development diagnostics

`-DLFFW_RUNTIME_DIAGNOSTICS=ON` adds a development-only C header,
`listenflow/whisper_runtime_diagnostics.h`, and eight diagnostic functions to the
same inference library. The option defaults to `OFF`; the release library exports
only the original 21 functions. Neither the product C++ header nor `_native`
exposes diagnostics. A Python build with the option enabled also builds
`_lffw_diagnostics.abi3.so` in the build directory. It is not installed or added to
the wheel.

Use the same verified SDK and bundle inputs as the normal Python build, with a
separate build directory, and compile with `--parallel 2`. Import the diagnostic
module from that directory through `PYTHONPATH`. Both extensions must use the
runtime from the same build. Keep product regression tests and the final
development install on an `OFF` build.

`_lffw_diagnostics.Session(flags=7)` is a context manager that observes synchronous
product calls on its own OS thread. Read its `records` after leaving the context,
then call `close()`. Records contain copied Python values and survive model and
session closure. A session owns no model or device pointer. `HOST=1` records host
times; `EVENTS=2` adds CUDA events around actual copies and enqueue/graph replay;
`WAIT=4` retains completion queries and individual sleeps. Wall and wait controls
omit `EVENTS`, so they add no timing events. Event measurements report elapsed
time on the stream, including submission gaps. Host wait times and CPU times are
reported separately. Diagnostics add no event/stream/device synchronization and
preserve completion polling, the 50-microsecond sleep and timer slack restoration.

The CPU observer tests load the diagnostic runtime in separate subprocesses while
the main test process retains the `OFF` development install. Their default module
directory is `build/python-diagnostics`; another build directory can be supplied
through `PYTHONPATH`. Those subprocesses replace only external CUDA/backend calls,
so they require no GPU. The opt-in integration test in
`python/tests/test_runtime_diagnostics_gpu.py` instead requires the `ON` runtime
installed in the development source package, the diagnostic module on
`PYTHONPATH`, and an externally reserved sm120 GPU. Restore the `OFF` install
afterward.

`fp8_trial` visits all 16 decoder buckets through real audio and native prefix KV.
Its reports identify these inputs as `product_prefix` and retain tokens, history,
bucket and precision. They are a different input condition from the former zero
self-KV experiment. Host timeline endpoints are reported with bounds from host
timestamps; diagnostics do not synchronize an extra CUDA event to calibrate them.

### TensorRT layer observations with Nsight Systems

`python/scripts/profile_layers.py` observes the release product under Nsight
Systems CLI 2026.6.1. It requires neither `LFFW_RUNTIME_DIAGNOSTICS=ON` nor a
TensorRT profiler API. Install the CLI under HOME without running package
installation scripts or using root. The Linux x86_64 CLI package is linked from
[NVIDIA's download page](https://developer.nvidia.com/nsight-systems/get-started):

```bash
set -e
mkdir -p .nf/tmp
curl --fail --location \
  https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2026_6/nsightsystems-linux-cli-public-2026.6.1.218-39266636.deb \
  --output .nf/tmp/nsightsystems-linux-cli-public-2026.6.1.218-39266636.deb
printf '%s  %s\n' \
  e71440978244e3883fc769da91a44a03451129ce21988d0db656a6bf7c1cb56b \
  .nf/tmp/nsightsystems-linux-cli-public-2026.6.1.218-39266636.deb | sha256sum --check
mkdir -p "$HOME/.local/opt"
mkdir "$HOME/.local/opt/nsight-systems-cli-2026.6.1"
dpkg-deb --extract .nf/tmp/nsightsystems-linux-cli-public-2026.6.1.218-39266636.deb \
  "$HOME/.local/opt/nsight-systems-cli-2026.6.1"
NSYS="$HOME/.local/opt/nsight-systems-cli-2026.6.1/opt/nvidia/nsight-systems-cli/2026.6.1/bin/nsys"
"$NSYS" --version
```

After reserving the GPU, run from `python/` with the release build installed and
its tensoracle and TensorRT runtime directories on `LD_LIBRARY_PATH`:

```bash
uv run --no-sync --locked python -m scripts.profile_layers \
  --nsys "$NSYS" --checkpoint-dir /absolute/path/to/checkpoint \
  --manifest /absolute/path/to/manifest.csv \
  --output-dir ../.nf/tmp/layers --iterations 2 --buckets 1 8
```

The script starts three sequential product processes, one each for mel, encoder
and decoder. Each runs under `nsys profile --trace=cuda,nvtx --sample=none
--cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop
--cuda-graph-trace=node`. The first eligible test-clean waveform in the manifest
is used. Model loading is outside capture. Decoder prefix preparation, the first
enqueue for each encountered bucket, CUDA Graph construction, and replay are
inside capture; preceding buckets needed for the prefix are included. No graph
is created for the mel or encoder diagnostic. The script adds a context
synchronization only immediately before `cuProfilerStop`; product completion
polling stays unchanged. The script does not reserve the GPU or choose a cache.

`layers.json` separates each component's `enqueue` observations from the
decoder's `graph_replay` observations and retains fused names, operation kinds,
actual sample counts, GPU spans and millisecond summaries. Ordinary enqueue
observations come from `nsys stats --report nvtx_gpu_proj_trace`; the SQLite
export supplies each range's identity, domain and containment. The observer
selects the innermost projected ranges inside a TensorRT
`ExecutionContext::enqueueV3` on the same OS thread, including operations in
`myelin-exec` and unnamed domains. It omits enclosing ranges from timing totals
and retains their TensorRT fusion name in `parent_layer`. Each observation and
summary includes `domain`; an empty string denotes an unnamed domain. GPU spans
include gaps between the first and last enclosed GPU operation; CPU NVTX
duration is never used as GPU time.
Replay observations join the SQLite export's graph-construction ranges to
kernel `sourceGraphNodeId` and `sourceGraphId`, selecting the deepest enclosing
operation across those domains, within the TensorRT enqueue. The TensorRT
fusion name and operation domain are retained here too. Both the GPU span and
summed kernel time are retained for replay. Names are emitted verbatim: they
can represent fusions, reformats, plugins or internal operations and may change
between TensorRT versions. Their count need not equal the source network's
layer count. The existing operation classifier retains unknown names as
`other`. Observations include tracing overhead and are not product latency or
a speed comparison.

Raw `.nsys-rep`, `.sqlite`, CSV, product metadata and logs stay alongside the JSON
under the checkout's `.nf/tmp/`. Use a new output directory for each run. Check
root disk space before and after collection, apply the reserved memory/time
limits, and stop after a failed collection rather than automatically retrying.
The observer tests create small CSV and SQLite fixtures and run without CUDA.

Saved traces can be reprocessed without initializing CUDA or rerunning the
product. From `python/`, retain the original report and write a separate one:

```bash
uv run --no-sync --locked python - <<'PY'
import json
from pathlib import Path
from scripts.profile_layers import read_enqueue_trace, read_graph_trace, summarize_observations
root = Path('../.nf/tmp/layers')
report = json.loads((root / 'layers.json').read_text())
for component in ('mel', 'encoder', 'decoder'):
    rows = read_enqueue_trace(root / f'{component}_nvtx_gpu_proj_trace.csv',
                              root / f'{component}.sqlite')
    report[component]['enqueue'] = summarize_observations(rows)
report['decoder']['graph_replay'] = summarize_observations(
    read_graph_trace(root / 'decoder.sqlite'))
(root / 'layers-reparsed.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
PY
```

### Network construction

`lffw_network` builds the Whisper mel, encoder, greedy decoder, and sampled
decoder as a C++17 static library. Enable it explicitly while the plugin and
Python migration is being integrated. The model library does not need Python,
a producer environment, or a cubin bundle to compile:

```bash
cmake -S cpp -B build/network -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DLFFW_BUILD_NETWORK=ON -DLFFW_BUILD_PLUGINS=OFF -DLFFW_BUILD_RUNTIME=OFF
cmake --build build/network --parallel 2
ctest --test-dir build/network --output-on-failure -LE gpu
```

Prepare the locked Conan dependencies using
`cmake -P cpp/conan/prepare-tests.cmake`. The generated package configurations
provide tensoracle and TensorRT; the toolkit provides CUDA headers. CMake and
Ninja remain host tools. Plugin targets can be built alongside the model by
leaving `LFFW_BUILD_PLUGINS=ON` and providing `LFFW_BUNDLE_ROOT`.

The caller fills `model_source` from its parsed configuration, tokenizer and FP8
calibration. `weights` contains the unsharded checkpoint or all distinct shards,
with safetensors files only. `mel_weights` is a small safetensors file containing
`mel_<n_mels>` with shape `(n_mels, 201)` and the original filter values. Building
mel never reads the full Whisper checkpoint. Auxiliary configuration, tokenizer,
calibration and original mel-filter paths supply **bytes for SHA-256 only**; the
C++ library does not parse JSON or NPZ. Concrete configuration, token IDs and
calibration values also enter the graph identity. Converted and transposed
constants remain owned through completion of engine serialization.

```cpp
#include <listenflow/whisper_network.hpp>
namespace fw = listenflow::fastest_whisper;

// source supplies the fields described above. Keep the caller's CUDA context
// current while opening, executing and destroying GPU resources.
fw::engine_build encoder(source, fw::graph_kind::encoder, fw::precision::fp32);
auto handle = encoder.open();
auto execution = encoder.create_execution(handle);
const auto& io_and_profiles = encoder.declaration();
```

`engine_build::declaration()` supplies the same I/O and profiles used to define
network inputs. Its workspace is 4 GiB and optimization level is 5; tensoracle v2
disables TF32. FP8 uses E4M3 Q/DQ on linear operands with fp16 surrounding
operations and rejects sm86. Only the fp16 decoder requires `fc1_library`,
registering `LffwFp16Fc1` version `1` with its embedded bundle identity. FP32 and
FP8 do not load that plugin.

`open()` automatically rebuilds an unusable saved engine once with the original
declaration. Corrupt, condition-mismatch and I/O-mismatch statuses are recoverable.
For plugin engines it additionally probes an execution context: an internal
TensorRT failure during that initial cached-engine probe is recoverable. Callback
failures, invalid arguments, allocation failures and failures after rebuilding
propagate. The model library has no caller option for requesting recovery.

`create_sampled_executions()` creates one execution for each history bucket
`1,8,16,24,32,48,64,96,128,160,192,224,256,320,384,448` within the configured profile.
Each engine allocates one device scratch region using tensoracle's maximum
`scratch_size()` across input profiles. Its executions, including the initial
cached-plugin probe, borrow that region through tensoracle 2.2.1's caller-owned
scratch API. Executions retain the region after the caller closes the engine
handle; the last owner frees it after all borrowing executions are destroyed.
Executions sharing an engine's scratch must run serially, including CUDA graph
replays. The product runtime uses its model's single stream and serializes model
operations. All execution creation goes through `create_execution()`.
Runtime code can also create these units lazily and use `bucket_shapes()` without
duplicating the I/O declaration.
An explicit `close()` on the last owner reports a scratch release failure and
retains that owner for another `close()` attempt. Destructors release resources
on a best-effort basis.
The caller still owns device I/O buffers, streams, completion, and CUDA Graphs.
Finish outstanding work before destroying executions and buffers.

The public header uses standard C++ types and the model's `tensor_dtype`,
`tensor_spec`, `input_shape`, and `tensor_address`. Each input's min/opt/max lives
with its I/O entry in `build_declaration`; `open()` derives backend options from
that one declaration. `whisper_engine` and `network_execution` hide backend
handles behind private implementations. `network_execution` forwards shape,
dtype and enqueue operations, accepting a stream as `void*`. A consumer does not
need tensoracle or CUDA headers. The model target links tensoracle privately.
Backend failures become `model_error`, retaining the original diagnostic and
integer `status()` for the inference C ABI. Current statuses are invalid argument
1, incompatible version 2, out of memory 3, internal error 4, corrupt engine 100,
condition mismatch 101, I/O mismatch 102, and plugin library error 110. Unknown
future status values are also retained. Local configuration errors remain
`std::invalid_argument`. The public-header CTest target consumes the model library without backend or
CUDA include paths, and checks that its opaque handles can be moved and closed.

### Native GPU comparisons

The opt-in `lffw_network_*_gpu` CTest cases require
`LFFW_NETWORK_FIXTURE_ROOT`, GPU access, and the corresponding prepared input.
Select only prepared graph/precision cases with CTest's `-R` option. Their test
executable writes native output bytes and records callback counts. FP8 structure
cases check the selected GPU's compute capability and skip with a reason and
return code 77 outside sm89 and sm120. The mel graph uses fp32 for every precision
identity. The `values`
cases (fp32 and mel) compare outputs to the 1.x reference using existing bounds.
The `structure` cases (fp16 and fp8) check I/O, E4M3 Q/DQ operands and scales,
required fc1 plugins, execution and reuse. Reduced precision tensor values are
diagnostic; transcription error rates in the product tests determine accuracy.
The fc1 graph inspection preserves the original four assembled-graph cases:
both encoders and the fp32 decoder contain no plugins; the fp16 decoder contains
32 named fc1 layers in order. Each layer has three fp16 inputs shaped
`(1, 1, 1280)`, `(5120, 1280)`, `(5120,)` and one fp16 output `(1, 1, 5120)`.
Sampled decoding applies the same checks to each candidate's slice and plugin
(160 layers for batch 5). The CPU `lffw_network_fc1_contract` case checks rejection
of altered shapes, dtypes, arity and names without CUDA or TensorRT.
`prepare_network_fixture.py statistics` reports native/1.x differences and each
implementation's differences from the saved fp32 reference. `compare` checks
saved fp32/mel bytes independently. Both test programs use the default tensoracle cache. The fixture's
configuration includes an identity specific to its temporary directory, so its
engine files can be removed without altering application caches. Never run GPU
tests without the host's required resource reservation.

`cpp/tests/prepare_network_fixture.py` runs with NumPy, safetensors, tokenizers,
and the existing 1.x package available. `prepare` reads only CPU data. `reference`
runs the original 1.x network and needs a GPU. Use separate processes for the
reference and native library, and point the native executable's library loader
at the same TensorRT runtime libraries. For example:

```bash
python cpp/tests/prepare_network_fixture.py prepare \
  --checkpoint /absolute/path/to/checkpoint --root /absolute/path/to/fixture \
  --graph mel --precision fp32
python cpp/tests/prepare_network_fixture.py reference \
  --checkpoint /absolute/path/to/checkpoint --root /absolute/path/to/fixture \
  --graph mel --precision fp32
LFFW_NETWORK_FIXTURE_ROOT=/absolute/path/to/fixture \
  build/network/lffw_network_gpu mel fp32
python cpp/tests/prepare_network_fixture.py compare \
  --checkpoint /absolute/path/to/checkpoint --root /absolute/path/to/fixture \
  --graph mel --precision fp32 \
  --native-result /absolute/path/to/fixture/mel-fp32-b5-h1-native
```

For decoder comparisons, `prepare --encoder-reference /path/to/encoder-reference.safetensors`
uses cross-KV from the fixed audio's encoder output. Sampled CTest cases use
batch 5 and history 448; prepare those inputs with `--batch 5 --history 448`. The initial dummy cache
position is masked in the same way as product decoding. The GPU test also checks
that an incomplete shape update preserves backend status 1 in `model_error` and
that the last valid shapes still execute correctly.

The native executable's optional arguments are graph, precision, sample batch,
history length, and mode. Modes `reuse`, `corrupt`, and `condition` check saved
engine reuse and one rebuild, and `memory` retains the sampled executions to
report their post-open device-memory increase and the engine's `scratch_size()`.
The shared scratch allocation already exists at `open()`, so that increase
excludes its bytes. The test-only C ABI observer counts
real network callbacks and distinguishes saved-engine error statuses. Only the
fixture's recorded `engine-path.txt` files identify files to clean up after the
run. Baseline paths are recorded in `*-reference-engine-path.txt`. Output and
reference files may be retained for inspection. Timing is disabled unless
`LFFW_NETWORK_MEASURE_TIME` is explicitly set; obtain permission for time
measurements separately from correctness and memory checks.
