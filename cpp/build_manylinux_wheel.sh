#!/usr/bin/env bash
# Build the production wheel from its source archive, without exposing a GPU.
set -euo pipefail

if [[ "${1:-}" == --inside ]]; then
    export PATH=/opt/python/cp313-cp313/bin:$PATH
    python -m pip install --disable-pip-version-check --no-cache-dir --require-hashes \
        --target /tmp/backend -r /release/build-requirements.txt
    export PYTHONPATH=/tmp/backend
    mkdir /tmp/source
    archives=(/release/*.tar.gz)
    [[ ${#archives[@]} == 1 ]]
    python -m tarfile -e "${archives[0]}" /tmp/source --filter data
    sources=(/tmp/source/*)
    cd "${sources[0]}"
    python - <<'PY'
from hashlib import sha256
from pathlib import Path
Path('/release/source-manifest.sha256').write_text(''.join(
    f'{sha256(path.read_bytes()).hexdigest()}  {path.as_posix()}\n'
    for path in sorted(Path('.').rglob('*')) if path.is_file()
))
PY
    export CMAKE_GENERATOR='Unix Makefiles'
    export CMAKE_BUILD_PARALLEL_LEVEL=2
    export CUDAToolkit_ROOT=/cuda TENSORRT_ROOT=/sdk
    export TENSORACLE_CPP_ROOT=/tensoracle-cpp LFFW_BUNDLE_ROOT=/bundle
    python -c 'from scikit_build_core.build import build_wheel; print(build_wheel("/release/raw"))'
    export LD_LIBRARY_PATH=/sdk/lib:/cuda/lib
    auditwheel show /release/raw/*.whl > /release/auditwheel-raw.txt
    auditwheel repair --only-plat --plat manylinux_2_28_x86_64 \
        --exclude libcuda.so.1 --exclude libnvinfer.so.11 \
        -w /release/final /release/raw/*.whl
    auditwheel show /release/final/*.whl > /release/auditwheel-final.txt
    python -m zipfile -e /release/final/*.whl /release/inspect
    library=/release/inspect/listenflow/fastest_whisper/liblffw_fc1_plugin.so
    LC_ALL=C readelf --file-header --wide "$library" > /release/elf-header.txt
    LC_ALL=C readelf --version-info --wide "$library" > /release/elf-versions.txt
    LC_ALL=C readelf --dynamic --wide "$library" > /release/elf-dynamic.txt
    {
        gcc --version
        ld --version
        cmake --version
        make --version
        python --version
        auditwheel --version
        python -m pip freeze --path /tmp/backend
    } > /release/build-tools.txt
    exit
fi

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
release=$(realpath -m "${1:?Pass a new output directory}")
sdk=$(realpath "${TENSORRT_ROOT:?Set TENSORRT_ROOT to the verified SDK}")
image=$(cat "$repo/cpp/manylinux-image.txt")
[[ "$image" =~ @sha256:[0-9a-f]{64}$ ]]
mkdir -p "$(dirname "$release")"
mkdir "$release"
cd "$repo/python"
cuda=$(uv run --group cubin --locked python -c \
    'import os; from importlib.metadata import distribution; from pathlib import Path; print(Path(os.environ.get("CUDAToolkit_ROOT") or distribution("nvidia-cuda-runtime").locate_file("nvidia/cu13")).resolve())')
if [[ -n "${LFFW_CUDA_DRIVER_LIBRARY:-}" ]]; then
    driver=$(realpath "$LFFW_CUDA_DRIVER_LIBRARY")
elif [[ -r /usr/lib/wsl/lib/libcuda.so.1 ]]; then
    driver=/usr/lib/wsl/lib/libcuda.so.1
else
    driver=$(ldconfig -p | awk '/libcuda.so.1 .*x86-64/ && !found {print $NF; found=1}')
    driver=$(realpath "${driver:?Set LFFW_CUDA_DRIVER_LIBRARY to the NVIDIA driver libcuda.so.1}")
fi
df -h / > "$release/disk-before.txt"
git rev-parse HEAD > "$release/source-revision.txt"
git diff --binary HEAD > "$release/source.patch"
cp uv.lock "$release/uv.lock"
uv run --group cubin --locked python ../cpp/generate_cubin_bundle.py --output-dir "$release/bundle"
helper=$(uv run --group cubin --locked python -c \
    'from pathlib import Path; import tensoracle; print(Path(tensoracle.__file__).parent / "cpp")')
cp ../cpp/tensorrt-sdk.sha256 "$release/tensorrt-sdk.sha256"
uv run --group cubin --locked python - "$sdk" "$release" <<'PY'
import hashlib, sys
from pathlib import Path
sdk, output = map(Path, sys.argv[1:])
verified = []
for line in (output / 'tensorrt-sdk.sha256').read_text().splitlines():
    expected, name = line.split()
    if name.startswith('sdk/'):
        path = sdk / name.removeprefix('sdk/')
        with path.open('rb') as source:
            actual = hashlib.file_digest(source, 'sha256').hexdigest()
        if actual != expected:
            raise SystemExit(f'SDK checksum mismatch: {name}')
        verified.append(f'{actual}  {name}\n')
(output / 'verified-sdk.sha256').write_text(''.join(verified))
PY
{
    uv --version
    cmake --version
    ninja --version
    uv run --group cubin --locked python -c \
        'from importlib.metadata import version; print("\n".join(f"{name}=={version(name)}" for name in ("tensoracle", "nvidia-cuda-runtime", "nvidia-cuda-crt", "nvidia-cuda-nvcc", "nvidia-cuda-cuobjdump", "nvidia-nvvm")))'
} > "$release/host-tools.txt"
sha256sum "$driver" > "$release/cuda-driver.sha256"
uv build --sdist --out-dir "$release"
uv run --locked python - "$release" <<'PY'
import sys, tomllib
from pathlib import Path
output = Path(sys.argv[1])
requires = tomllib.loads(Path('pyproject.toml').read_text())['build-system']['requires']
(output / 'build-requirements.in').write_text('\n'.join(requires) + '\n')
PY
uv pip compile "$release/build-requirements.in" --python-version 3.13 \
    --generate-hashes --output-file "$release/build-requirements.txt"
docker pull "$image"
docker buildx imagetools inspect --raw "$image" > "$release/image-manifest.json"
docker image inspect "$image" --format \
    '{{json .RepoDigests}} {{.Id}} {{.Architecture}} {{.Os}}' > "$release/image.txt"
docker run --rm --cpus 2 --memory 16g --user "$(id -u):$(id -g)" \
    --mount "type=bind,src=$release,dst=/release" \
    --mount "type=bind,src=$repo/cpp/build_manylinux_wheel.sh,dst=/build.sh,readonly" \
    --mount "type=bind,src=$release/bundle,dst=/bundle,readonly" \
    --mount "type=bind,src=$helper,dst=/tensoracle-cpp,readonly" \
    --mount "type=bind,src=$sdk,dst=/sdk,readonly" \
    --mount "type=bind,src=$cuda/include,dst=/cuda/include,readonly" \
    --mount "type=bind,src=$driver,dst=/cuda/lib/libcuda.so,readonly" \
    --mount "type=bind,src=$driver,dst=/cuda/lib/libcuda.so.1,readonly" \
    "$image" bash /build.sh --inside > "$release/build.log" 2>&1
uv run --group cubin --locked python - "$release" <<'PY'
import sys
from pathlib import Path
import tensorrt
from tensoracle.bundle import inspect_bundle
output = Path(sys.argv[1])
library = output / 'inspect/listenflow/fastest_whisper/liblffw_fc1_plugin.so'
digest = inspect_bundle(output / 'bundle', plugin=library)
(output / 'bundle-inspection.txt').write_text(digest + '\n')
PY
cmake "-DPLUGIN_LIBRARY=$release/inspect/listenflow/fastest_whisper/liblffw_fc1_plugin.so" \
    "-DNM=$(command -v nm)" "-DREADELF=$(command -v readelf)" \
    -P ../cpp/tests/check_plugin.cmake
sha256sum "$release"/*.tar.gz "$release"/final/*.whl > "$release/artifacts.sha256"
df -h / > "$release/disk-after.txt"
printf 'Build records and final wheel: %s\n' "$release"
