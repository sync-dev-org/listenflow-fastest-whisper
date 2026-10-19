# Build and inspect a Linux release wheel

Follow [tensoracle's repository guide, sections 5–6](https://sync-dev-org.github.io/tensoracle/cpp-python-repositories/).
The release target is Linux x86_64, Python 3.12–3.13, and glibc 2.28 or later.
The plugin does not use the Python C API, so the inspected wheel uses
`py3-none-manylinux_2_28_x86_64`. CUDA 13 driver support and the declared
`tensorrt-cu13` dependency are additional runtime requirements.

Prepare the verified SDK and host CUDA toolkit using [README.md](README.md).
Keep host CMake, Ninja, binutils, Docker, and uv available. Use a clean release
commit bearing the exact version tag. The Python backend preserves that version
in the source archive. Untagged commits use `0.0.0.dev0` for development checks.

From `python/`, with `TENSORRT_ROOT` naming the verified SDK:

```bash
uv sync --group cubin --locked
bash ../cpp/build_manylinux_wheel.sh ../dist/release
```

The output directory must be new. The script produces the cubin bundle without
a GPU, builds the source archive, and compiles that archive inside the exact
manylinux image recorded in `manylinux-image.txt`. It mounts the SDK, toolkit,
and installed tensoracle C++ helper read-only. It does not mount GPU devices or
depend on another repository. The container supplies CMake and GNU Make; the
script does not install CMake or Ninja into the Python environment.
The script removes each container after execution. Remove the image when its
verification is complete if the development host needs the disk space.

The output directory records the source revision and patch, complete archive
source manifest, image manifest/configuration digests and architecture, locked
Python dependencies, hash-pinned build dependencies, SDK checksums, bundle
digest, compiler/linker/CMake/Make/Python/CUDA/backend versions, build log,
artifact hashes, disk measurements, and raw and final ELF/auditwheel reports.
The CMake settings are Release, `BUILD_TESTING=OFF`, Unix Makefiles, two build
jobs, and the four explicit build-input roots in `build_manylinux_wheel.sh`.
Archive these records with the runtime verification results for every release.

Review `elf-header.txt`, `elf-versions.txt`, and `elf-dynamic.txt` for every
shipped ELF. The wheel must contain only the production fc1 library as native
code, with ELF64 x86-64 architecture. Compare symbol versions numerically with
the intended baseline. There must be no absolute SDK or build-machine RPATH or
RUNPATH. `check_plugin.cmake` also checks registration exports and direct
dependencies. The tensoracle inspector compares the embedded bytes and digest
with all nine cubins, including fc1 for sm86, sm89, and sm120.

`auditwheel repair` excludes `libcuda.so.1` (the NVIDIA driver) and
`libnvinfer.so.11` (the declared TensorRT dependency). Never copy either into
this wheel, and never ship link-time stubs. These exclusions alone do not
establish compatibility. `auditwheel show` still reports external NVIDIA
libraries and can describe the wheel as `linux_x86_64`; retain that report.
Check the plugin's versioned symbols separately and verify that the declared
TensorRT wheel and host driver resolve its external dependencies at runtime.
Keep the final `py3-none-manylinux_2_28_x86_64` tag only when these checks pass.

Create separate temporary Python 3.12 and 3.13 environments, install the final
wheel with its declared dependencies, and run outside the source tree. Remove
SDK, producer, source `PYTHONPATH`, custom library paths, and toolkit environment
variables from the runtime process. The package must import and load its own
fc1 library. After coordinating shared GPU use, use
`python/tests/wheel_runtime.py` with the independent interpreter's `-I` option,
a local large-v3 checkpoint, and the small-boat and silence WAVs. Pass
`--checkpoint PATH --audio PATH --audio PATH`. Save its JSON output, then start
a new process with the same cache and `--reuse --expect FIRST_RESULT.json`.
The probe checks preparation, streaming, fp16 token sequences, unchanged saved
engine bytes and timestamps, and the absence of producer imports. Compare
`observed` with the development build's result as well. A separate CUDA-driver
observer can verify fc1 launches and the selected SM without changing the plugin.

Retain installed dependency versions, GPU/driver/runtime versions, test commands,
elapsed times, and disk measurements with the build records. Run the existing
fc1 correctness and engine-cache recovery GPU tests with tensoracle 1.1.0. Record
sm86 and sm89 as build/inspection coverage when those GPUs are unavailable.
Check artifact size against the current upload limit before publication. These
commands build and inspect artifacts; publication is a separate action.
