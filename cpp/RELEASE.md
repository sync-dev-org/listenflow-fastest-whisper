# Build and inspect a Linux release wheel

The release target is Linux x86_64 (including WSL2), Python 3.12–3.13, and
glibc 2.28 or later. The wheel tag is
`cp312-abi3-manylinux_2_28_x86_64`. Its native files are
`listenflow/fastest_whisper/liblffw_runtime.so`, `_native.abi3.so`, and
`liblffw_fc1_plugin.so`. The plugin embeds nine cubins for sm86, sm89 and sm120.
The wheel reads `libtensoracle.so.2` from the declared tensoracle dependency's
wheel. The host supplies the NVIDIA driver and TensorRT runtime; see the
[installation instructions](https://github.com/sync-dev-org/listenflow-fastest-whisper#install)
for their minimum versions and the TensorRT apt packages.

Keep host CMake, Ninja, binutils, Docker, uv and the pinned Conan CLI available.
`python/pyproject.toml` pins Conan's version in `[build-system].requires`.
The host's `gh` authentication must be able to read `sync-dev-org/conan-recipes`.
Use a clean release commit bearing its exact version tag. Untagged commits use
`0.0.0.dev0` for development checks. The source archive preserves that version.
The source archive is an internal build input: publish only the inspected wheel.

Prepare the Python tools from `python/`:

```bash
uv sync --no-install-project --group cubin --locked
```

Check the host's free space and the image reference in `cpp/manylinux-image.txt`.
Pull that exact digest if it is absent, coordinating the download and disk space
with other users of the host. The build script uses the already installed image
and does not pull, replace or remove images. From the repository root:

```bash
bash cpp/build_manylinux_wheel.sh dist/release
```

The output directory must be new. The script obtains and verifies the recipe
archive through `cpp/conan/fetch_recipe.py`. That host-only child reads the
host's `gh` token and places it in a temporary Conan `source_credentials.json`
under `/tmp`, deletes the credentials immediately after downloading, and
withholds the child's diagnostics on failure. The archive URL and SHA256 have
one definition in `cpp/conan/recipe-archive.json`. The script places the
verified archive inside the internal sdist. The container receives that archive
without credentials, and its bootstrap checks the SHA256 and each exported
recipe's revision against `cpp/conan.lock`, even if recipes are already cached.

Host cubin generation uses `generate_cubin_bundle.py --cuda-root` with the
Conan-provided CUDA redist root. Only that producer command receives the
toolkit's `lib` directory in `LD_LIBRARY_PATH` so NVRTC finds its builtins.
The container obtains tensoracle's prebuilt
SDK, TensorRT's SDK and CUDA's redist components through the same consumer
recipe and lock. It does not mount host TensorRT or CUDA SDKs or use a PyPI
wheel as the C++ SDK. It mounts only the internal source archive, generated
bundle, build script, output directory and host driver link library. GPU
devices are not exposed.

The fixed image supplies GCC 14.2.1, GNU Make and CMake. The release profile is
`cpp/conan/profiles/manylinux-gcc-toolset-14`, with `cuda-toolkit/*:mode=redist`
and system CMake replacing Conan's CMake tool dependency. `LFFW_RELEASE_BUILD=1`
selects this profile and `conan install '--build=*'`. The container obtains the
pinned Conan CLI and other build tools from hash-pinned resolutions of
`[build-system].requires`. It uses two CPU jobs, a 16 GiB RAM limit, Release,
Unix Makefiles and `BUILD_TESTING=OFF`. Its new, writable `CONAN_HOME` and
package storage live under `/tmp`. Git discovery is disabled while building
the extracted sdist. The container is removed after execution. The script
samples root-disk space every five seconds and stops the container below
10 GiB free. It removes its host staging directory on success or failure.
If the build or verification fails, it removes the unverified `final/` candidates
and preserves the failure status and diagnostic records.

The output directory retains:

- Source revision, patch, extracted-source SHA256 manifest, source version and
  source-archive hash; Python lock and hash-pinned build requirements.
- Image reference, manifest digest, configuration digest and architecture;
  container configuration, exit code, build log and disk measurements.
- Compiler, linker, CMake, Make, Python, Conan, CUDA compiler, backend and
  installed build-package versions; Conan profile, lock, resolved graph,
  toolchain, install log, recipe archive hash and exported recipe revisions.
- Driver checksum, bundle manifest and digest, nine-cubin comparison result,
  plugin export/dependency check, raw/final `auditwheel show` reports, final
  wheel hash, member list and reports for every ELF.

`inspect_release_wheel.py` checks ELF64 x86-64, the exact wheel tag and native
file set, and records each ELF's `NEEDED`, `RPATH`, `RUNPATH` and maximum
`GLIBC_*`, `GLIBCXX_*`, `CXXABI_*` versions. It compares numeric components
against 2.28, 3.4.25 and 1.3.11 respectively. The inference library's sole
RUNPATH must be `$ORIGIN/../../tensoracle/cpp/lib`; the extension uses
`$ORIGIN`, and the plugin has no search path. Absolute build/Conan paths and
TensorRT paths must not appear. `NEEDED` must not contain `libcudart.so`.
`auditwheel repair` excludes `libtensoracle.so.2`, `libnvinfer.so.11` and
`libcuda.so.1`. Never bundle them or driver stubs. The exclusions require
separate runtime verification; retain auditwheel's external-library report.
The tensoracle bundle inspector compares the plugin's embedded bytes and
digest with all nine original cubins. `check_plugin.cmake` checks exports and
direct dependencies.

After the container finishes, create separate temporary Python 3.12 and 3.13
environments outside the records directory. Install the final wheel with its
declared dependencies and run outside the source tree with `python -I`.
Remove build/toolkit variables, source `PYTHONPATH` and custom library paths.
Import the package and its native extension and inspect `/proc/self/maps`:
exactly one resolved `libtensoracle.so.2` must come from that environment's
`site-packages/tensoracle/cpp/lib`. Save installed versions, commands and
results. Verify the runtime dependency lower bound against the SDK version.
On a verification host without apt's TensorRT runtime, obtain the host
operator's decision before installing the TensorRT wheel in the verification
environment through tensoracle's `tensorrt` extra (`tensoracle[tensorrt]`).
Set `LD_LIBRARY_PATH` to that environment's `site-packages/tensorrt_libs`
directory alone. Keep `PYTHONPATH`, SDK variables
and all other library paths unset. Verify that tensoracle still resolves from
the wheel's RUNPATH. Record the exact `LD_LIBRARY_PATH` string and that native
import without this explicit path was not verified on that host. This setup
checks import and GPU operation with the specified TensorRT loader path; it
does not change the package's host-runtime requirement or wheel RUNPATH.

Coordinate sm120 GPU use before running representative large-v3 fp16 and fp8
tests with the installed wheel. `python/tests/wheel_runtime.py` accepts
`--precision fp16` or `--precision fp8`, `--checkpoint PATH`, and repeated
`--audio PATH` arguments for the small-boat and silence WAVs. Run it with the
independent interpreter's `-I` option. Set `XDG_CACHE_HOME` to a temporary
engine-cache directory outside the records. Save its JSON result; a fresh
process with the same cache, `--reuse --expect FIRST_RESULT.json`, checks
unchanged engine inodes/hashes and identical tokens and transcripts. The probe
also rejects producer/torch/triton imports. Keep each run within the agreed GPU,
RAM, time and disk limits and record disk space before and after each run.
Remove its engine cache after the command. sm86 and sm89 receive build and
inspection coverage; GPU measurements use sm120.

Run the Release and standalone C++ CPU CTests (`ctest ... -LE gpu`) and the
Python CPU suite without skips. Retain their results along with the independent
Python and GPU checks. Delete verification environments, source copies, engine
caches and other intermediates after use. Measure the records directory before
copying it; it must contain records and the final wheel, without venvs or
`hosts/runs/` execution environments. Check the wheel's size and restricted
terms before publication. Release publication is performed separately.
