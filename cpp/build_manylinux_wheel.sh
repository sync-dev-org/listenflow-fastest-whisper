#!/usr/bin/env bash
# Build and inspect the production wheel without exposing a GPU to Docker.
set -euo pipefail

if [[ "${1:-}" == --inside ]]; then
    trap 'printf "%s\n" "$?" > /release/container-exit-code.txt' EXIT
    export PATH=/opt/python/cp313-cp313/bin:$PATH
    python -m pip install --disable-pip-version-check --no-cache-dir --require-hashes \
        --target /tmp/backend -r /release/build-requirements.txt
    export PYTHONPATH=/tmp/backend PATH=/tmp/backend/bin:$PATH
    export CONAN_HOME=/tmp/conan-home
    mkdir -p "$CONAN_HOME" /tmp/conan-packages /tmp/source
    printf 'core.cache:storage_path=/tmp/conan-packages\n' > "$CONAN_HOME/global.conf"
    python -m tarfile -e /inputs/source.tar.gz /tmp/source --filter data
    sources=(/tmp/source/*)
    cd "${sources[0]}"
    export GIT_DIR=/nonexistent GIT_CEILING_DIRECTORIES="$PWD"
    export CMAKE_GENERATOR='Unix Makefiles' CMAKE_BUILD_PARALLEL_LEVEL=2
    export LFFW_RELEASE_BUILD=1 LFFW_BUNDLE_ROOT=/inputs/bundle
    python - <<'PY'
from hashlib import sha256
from pathlib import Path
import email
Path('/release/source-manifest.sha256').write_text(''.join(
    f'{sha256(path.read_bytes()).hexdigest()}  {path.as_posix()}\n'
    for path in sorted(Path('.').rglob('*')) if path.is_file()
))
from scikit_build_core.build import build_wheel
wheel = build_wheel('/tmp/raw', {'build-dir': '/tmp/wheel-build'})
version = email.message_from_string(Path('PKG-INFO').read_text())['Version']
assert wheel.startswith('listenflow_fastest_whisper-' + version + '-')
Path('/release/source-version.txt').write_text(version + '\n')
print(wheel)
PY
    mkdir /release/conan
    cp /tmp/wheel-build/conan/* /release/conan/
    profile=$PWD/cpp/conan/profiles/manylinux-gcc-toolset-14
    conan graph info cpp "--profile:host=$profile" "--profile:build=$profile" \
        --lockfile=cpp/conan.lock --no-remote --format=json > /release/conan/graph.json
    python - <<'PY'
import json, subprocess
from pathlib import Path
nodes = json.loads(Path('/release/conan/graph.json').read_text())['graph']['nodes'].values()
roots = {}
for node in nodes:
    if node['ref']:
        reference = node['ref'] + ':' + node['package_id']
        roots[node['ref'].split('/')[0]] = subprocess.check_output(
            ['conan', 'cache', 'path', reference], text=True).strip()
Path('/tmp/sdk-roots.json').write_text(json.dumps(roots))
PY
    cuda=$(python -c 'import json; print(json.load(open("/tmp/sdk-roots.json"))["cuda-toolkit"])')
    tensoracle=$(python -c 'import json; print(json.load(open("/tmp/sdk-roots.json"))["tensoracle"] + "/tensoracle/cpp/lib")')
    tensorrt=$(python -c 'import json; print(json.load(open("/tmp/sdk-roots.json"))["tensorrt"] + "/lib")')
    {
        gcc --version
        ld --version
        cmake --version
        make --version
        python --version
        conan --version
        "$cuda/bin/nvcc" --version
        auditwheel --version
        python -m pip freeze --path /tmp/backend
    } > /release/build-tools.txt
    export LD_LIBRARY_PATH=$tensoracle:$tensorrt:/usr/lib64
    auditwheel show /tmp/raw/*.whl > /release/auditwheel-raw.txt
    auditwheel repair --only-plat --plat manylinux_2_28_x86_64 \
        --exclude libcuda.so.1 --exclude libnvinfer.so.11 --exclude libtensoracle.so.2 \
        -w /release/final /tmp/raw/*.whl
    auditwheel show /release/final/*.whl > /release/auditwheel-final.txt
    python cpp/inspect_release_wheel.py /release/final/*.whl --output /release/elf \
        > /release/wheel-inspection.json
    python -m zipfile -e /release/final/*.whl /tmp/inspect
    cmake -DPLUGIN_LIBRARY=/tmp/inspect/listenflow/fastest_whisper/liblffw_fc1_plugin.so \
        "-DNM=$(command -v nm)" "-DREADELF=$(command -v readelf)" \
        -P cpp/tests/check_plugin.cmake > /release/plugin-inspection.txt
    exit
fi

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
release=$(realpath -m "${1:?Pass a new output directory}")
image=$(cat "$repo/cpp/manylinux-image.txt")
[[ "$image" =~ @sha256:[0-9a-f]{64}$ ]]
# Pulling a new image is an explicit preparation step; never change host images here.
docker image inspect "$image" > /dev/null
mkdir -p "$(dirname "$release")" "$repo/.nf/tmp"
mkdir "$release"
stage=$(mktemp -d "$repo/.nf/tmp/release-build.XXXXXX")
container=
cleanup() {
    local command_status=$?
    if [[ -n "$container" ]]; then
        docker rm -f "$container" > /dev/null 2>&1 || true
    fi
    if (( command_status != 0 )); then
        rm -rf -- "$release/final"
    fi
    rm -rf -- "$stage"
    df -h / > "$release/disk-after.txt"
    return "$command_status"
}
trap cleanup EXIT
if [[ -n "${LFFW_CUDA_DRIVER_LIBRARY:-}" ]]; then
    driver=$(realpath "$LFFW_CUDA_DRIVER_LIBRARY")
elif [[ -r /usr/lib/wsl/lib/libcuda.so.1 ]]; then
    driver=/usr/lib/wsl/lib/libcuda.so.1
else
    driver=$(ldconfig -p | awk '/libcuda.so.1 .*x86-64/ && !found {print $NF; found=1}')
    driver=$(realpath "${driver:?Set LFFW_CUDA_DRIVER_LIBRARY to the NVIDIA driver libcuda.so.1}")
fi
df -h / > "$release/disk-before.txt"
cd "$repo"
git rev-parse HEAD > "$release/source-revision.txt"
git diff --binary HEAD > "$release/source.patch"
cp cpp/conan.lock cpp/conan/recipe-archive.json "$release/"
cp cpp/conan/profiles/manylinux-gcc-toolset-14 "$release/conan-profile.txt"
cp python/uv.lock "$release/uv.lock"
# fetch_recipe.py confines gh's token to a temporary host-only credential file.
cp cpp/conan/fetch_recipe.py "$stage/conanfile.py"
cp cpp/conan/recipe-archive.json "$stage/recipe-archive.json"
conan source "$stage" -vwarning > "$release/recipe-fetch.txt" 2>&1
cmake -P cpp/conan/prepare-tests.cmake > "$release/host-conan-prepare.txt" 2>&1
conan graph info cpp --profile:host=cpp/conan/profiles/linux-gcc-toolset-15 \
    --profile:build=cpp/conan/profiles/linux-gcc-toolset-15 --lockfile=cpp/conan.lock \
    --no-remote --format=json > "$stage/host-graph.json" 2> "$release/host-graph.txt"
cuda=$(python3 - "$stage/host-graph.json" <<'PY'
import json, subprocess, sys
nodes = json.load(open(sys.argv[1]))['graph']['nodes'].values()
node, = [node for node in nodes if (node['ref'] or '').startswith('cuda-toolkit/')]
print(subprocess.check_output(['conan', 'cache', 'path', node['ref'] + ':' + node['package_id']], text=True).strip())
PY
)
cd "$repo/python"
LD_LIBRARY_PATH="$cuda/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    uv run --no-sync python ../cpp/generate_cubin_bundle.py --cuda-root "$cuda" \
    --output-dir "$stage/bundle" > "$release/bundle-generation.txt" 2>&1
cp "$stage/bundle/bundle.json" "$release/bundle-manifest.json"
uv build --sdist --out-dir "$stage/sdist" > "$release/sdist-build.txt" 2>&1
python3 - "$stage" "$release" <<'PY'
import hashlib, json, sys, tarfile, tomllib
from pathlib import Path
stage, output = map(Path, sys.argv[1:])
receipt = json.loads((stage / 'recipe-archive.json').read_text())
assert hashlib.sha256((stage / 'recipes.tar.gz').read_bytes()).hexdigest() == receipt['sha256']
original, = (stage / 'sdist').glob('*.tar.gz')
with tarfile.open(original) as source, tarfile.open(stage / 'source.tar.gz', 'w:gz') as target:
    for member in source.getmembers():
        target.addfile(member, source.extractfile(member) if member.isfile() else None)
    root = source.getmembers()[0].name.split('/')[0]
    target.add(stage / 'recipes.tar.gz', arcname=root + '/cpp/conan/recipes.tar.gz')
(output / 'sdist.sha256').write_text(hashlib.sha256((stage / 'source.tar.gz').read_bytes()).hexdigest() + '  source.tar.gz\n')
requires = tomllib.loads(Path('pyproject.toml').read_text())['build-system']['requires']
(output / 'build-requirements.in').write_text('\n'.join(requires) + '\n')
PY
uv pip compile "$release/build-requirements.in" --python-version 3.13 \
    --generate-hashes --output-file "$release/build-requirements.txt" \
    > "$release/build-requirements-resolution.txt" 2>&1
{
    uv --version
    cmake --version
    ninja --version
    uv run --no-sync python -m pip freeze 2>/dev/null || uv pip freeze
} > "$release/host-tools.txt"
sha256sum "$driver" > "$release/cuda-driver.sha256"
docker buildx imagetools inspect --raw "$image" > "$release/image-manifest.json"
docker image inspect "$image" --format \
    '{{json .RepoDigests}} {{.Id}} {{.Architecture}} {{.Os}}' > "$release/image.txt"
container=$(docker create --rm --cpus 2 --memory 16g --user "$(id -u):$(id -g)" \
    --mount "type=bind,src=$release,dst=/release" \
    --mount "type=bind,src=$stage,dst=/inputs,readonly" \
    --mount "type=bind,src=$repo/cpp/build_manylinux_wheel.sh,dst=/build.sh,readonly" \
    --mount "type=bind,src=$driver,dst=/usr/lib64/libcuda.so,readonly" \
    --mount "type=bind,src=$driver,dst=/usr/lib64/libcuda.so.1,readonly" \
    "$image" bash /build.sh --inside)
docker inspect "$container" > "$release/container.json"
docker start --attach "$container" > "$release/build.log" 2>&1 &
build_pid=$!
while kill -0 "$build_pid" 2>/dev/null; do
    available=$(df --output=avail -B1 / | tail -1)
    printf '%s %s\n' "$(date -Iseconds)" "$available" >> "$release/disk-during.txt"
    if (( available < 10 * 1024 * 1024 * 1024 )); then
        docker stop --time 5 "$container" > /dev/null
        wait "$build_pid" || true
        printf 'Stopped container: root disk has less than 10 GiB free\n' >&2
        exit 1
    fi
    sleep 5
done
# Docker attach does not propagate the contained program's exit code.
wait "$build_pid"
[[ $(cat "$release/container-exit-code.txt") == 0 ]]
container=
python3 -m zipfile -e "$release"/final/*.whl "$stage/inspect"
uv run --no-sync python - "$stage" "$release" <<'PY'
import json, sys
from pathlib import Path
from tensoracle.bundle import inspect_bundle
stage, output = map(Path, sys.argv[1:])
library = stage / 'inspect/listenflow/fastest_whisper/liblffw_fc1_plugin.so'
digest = inspect_bundle(stage / 'bundle', plugin=library)
assert len(json.loads((output / 'bundle-manifest.json').read_text())['entries']) == 9
(output / 'bundle-inspection.txt').write_text(digest + '\n')
PY
sha256sum "$release"/final/*.whl > "$release/artifacts.sha256"
printf 'Build records and final wheel: %s\n' "$release"
