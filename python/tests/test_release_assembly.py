"""Exercise release assembly with real sdists and isolated external processes."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TOKEN = b"release-test-only-private-token"


@pytest.fixture
def release_process(tmp_path):
    checkout = tmp_path / "checkout"
    project = checkout / "python"
    cpp = checkout / "cpp"
    conan = cpp / "conan"
    (conan / "profiles").mkdir(parents=True)
    package = project / "src/listenflow/fastest_whisper"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (project / ".gitignore").write_text(".venv\n")
    (project / "uv.lock").write_text("# fixture\n")
    requires = tomllib.loads((ROOT / "python/pyproject.toml").read_text())["build-system"][
        "requires"
    ]
    (project / "pyproject.toml").write_text(
        '[project]\nname="listenflow-fastest-whisper"\nversion="1.2.3"\n'
        'requires-python=">=3.12,<3.14"\n'
        '[build-system]\nbuild-backend="scikit_build_core.build"\nrequires='
        + json.dumps(requires)
        + '\n[tool.scikit-build]\nwheel.packages=["src/listenflow"]\n'
        '[tool.scikit-build.sdist.force-include]\n"../cpp"="cpp"\n'
    )
    script = Path(os.environ.get("LFFW_TEST_RELEASE_SCRIPT", ROOT / "cpp/build_manylinux_wheel.sh"))
    shutil.copyfile(script, cpp / "build_manylinux_wheel.sh")
    for filename in ("fetch_recipe.py", "prepare-tests.cmake"):
        shutil.copyfile(ROOT / "cpp/conan" / filename, conan / filename)
    shutil.copyfile(ROOT / "cpp/conan.lock", cpp / "conan.lock")
    shutil.copyfile(
        ROOT / "cpp/conan/profiles/manylinux-gcc-toolset-14",
        conan / "profiles/manylinux-gcc-toolset-14",
    )
    shutil.copyfile(ROOT / "cpp/manylinux-image.txt", cpp / "manylinux-image.txt")
    (cpp / "tensorrt-sdk.sha256").write_text("")
    # Kernel correctness belongs to the existing producer tests. This process
    # supplies deterministic bundle input, including the base script's old CLI.
    (cpp / "generate_cubin_bundle.py").write_text(
        "import argparse,json\nfrom pathlib import Path\n"
        "p=argparse.ArgumentParser();p.add_argument('--cuda-root');p.add_argument('--output-dir',type=Path,required=True)\n"
        "a=p.parse_args();a.output_dir.mkdir(parents=True)\n"
        "(a.output_dir/'bundle.json').write_text(json.dumps({'bundle_sha256':'fixture-bundle','entries':list(range(9))}))\n"
        "(a.output_dir/'bundle.cpp').write_text('// generated fixture bundle\\n')\n"
    )
    recipes_source = tmp_path / "recipes/recipes/probe"
    recipes_source.mkdir(parents=True)
    (recipes_source / "conanfile.py").write_text("# verified fixture recipe\n")
    recipes = tmp_path / "recipes.tar.gz"
    with tarfile.open(recipes, "w:gz") as archive:
        archive.add(tmp_path / "recipes/recipes", arcname="recipes")
    receipt = {
        "url": "https://example.invalid/fixture",
        "sha256": hashlib.sha256(recipes.read_bytes()).hexdigest(),
    }
    (conan / "recipe-archive.json").write_text(json.dumps(receipt))
    driver = tmp_path / "driver/libcuda.so.1"
    driver.parent.mkdir()
    driver.write_bytes(b"external driver link library")
    cuda, tensorrt = (tmp_path / "host-sdks" / name for name in ("cuda", "tensorrt"))
    for sdk in (cuda, tensorrt):
        (sdk / "include").mkdir(parents=True)
    modules = tmp_path / "external-modules"
    tensoracle = modules / "tensoracle"
    (tensoracle / "cpp").mkdir(parents=True)
    (tensoracle / "__init__.py").write_text("")
    (tensoracle / "bundle.py").write_text(
        "import json\n"
        "def inspect_bundle(root,plugin=None):\n"
        "    assert plugin is None or plugin.is_file()\n"
        "    return json.loads((root/'bundle.json').read_text())['bundle_sha256']\n"
    )
    (modules / "tensorrt.py").write_text("")
    info = modules / "tensoracle-2.2.3.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Name: tensoracle\nVersion: 2.2.3\n")
    for name in ("runtime", "crt", "nvcc", "cuobjdump", "nvvm"):
        distribution = "nvidia-nvvm" if name == "nvvm" else "nvidia-cuda-" + name
        info = modules / (distribution.replace("-", "_") + "-13.2.dist-info")
        info.mkdir()
        (info / "METADATA").write_text(f"Name: {distribution}\nVersion: 13.2\n")
    tools = tmp_path / "bin"
    tools.mkdir()
    shutil.copyfile(Path(__file__).with_name("release_process_shim.py"), tools / "shim.py")
    (tools / "shim.py").chmod(0o755)
    for name in ("docker", "conan", "uv", "cmake", "python", "gh"):
        (tools / name).symlink_to("shim.py")
    context = {
        "uv": shutil.which("uv"),
        "python": sys.executable,
        "bwrap": shutil.which("bwrap"),
        "bin": str(tools),
        "log": str(tmp_path / "calls.jsonl"),
        "state": str(tmp_path / "docker.json"),
        "recipes": str(recipes),
        "cuda": str(cuda),
        "external_modules": str(modules),
        "corrupt_archive": False,
        "execute_inside": False,
        "container_exit": 23,
        "attach_exit": 0,
    }
    assert context["uv"] and context["bwrap"], (
        "release tests require the existing uv and bubblewrap tools"
    )
    for command in (
        ["git", "init", "--quiet"],
        ["git", "add", "."],
        [
            "git",
            "-c",
            "user.name=Release test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "Fixture",
        ],
    ):
        subprocess.run(command, cwd=checkout, capture_output=True, check=True)
    home, temporary = tmp_path / "home", tmp_path / "tmp"
    home.mkdir()
    temporary.mkdir()
    (tmp_path / "unrelated.txt").write_text("keep unrelated output\n")
    (home / ".conan2").mkdir()
    (home / ".conan2/source_credentials.json").write_bytes(TOKEN)
    context_path = tmp_path / "context.json"
    environment = {
        "PATH": str(tools) + os.pathsep + os.defpath,
        "HOME": str(home),
        "TMPDIR": str(temporary),
        "UV_CACHE_DIR": str(Path.home() / ".cache/uv"),
        "LFFW_RELEASE_TEST_CONTEXT": str(context_path),
        "LFFW_CUDA_DRIVER_LIBRARY": str(driver),
        "CUDAToolkit_ROOT": str(cuda),
        "TENSORRT_ROOT": str(tensorrt),
    }
    output = tmp_path / "release"

    def run(**settings):
        context.update(settings)
        context_path.write_text(json.dumps(context))
        result = subprocess.run(
            ["bash", str(cpp / "build_manylinux_wheel.sh"), str(output)],
            cwd=checkout,
            env=environment,
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
        )
        state = (
            json.loads(Path(context["state"]).read_text())
            if Path(context["state"]).exists()
            else None
        )
        return result, state

    return run, output, receipt, [cuda, tensorrt, tensoracle / "cpp"]


@pytest.mark.req("SP-LFFASTESTWHISPER-024")
@pytest.mark.req("SP-LFFASTESTWHISPER-058")
def test_release_passes_verified_source_and_bundle_without_host_sdks(release_process):
    """The real shell sends the verified internal sdist and selects release bootstrap inside."""
    run, output, receipt, host_sdks = release_process
    result, state = run(execute_inside=True)
    assert state is not None, result.stdout + result.stderr
    assert state["archive"]["recipe_sha256"] == receipt["sha256"]
    assert state["archive"]["receipt"] == receipt
    assert not state["archive"]["credential_files"] and not state["archive"]["token_found"]
    assert TOKEN.decode() not in json.dumps(state)
    assert "--rm" in state["argv"] and "--inside" in state["argv"]
    assert not any(
        Path(mount["src"]).is_relative_to(sdk) for mount in state["mounts"] for sdk in host_sdks
    )
    assert not any(option.startswith("--gpus") for option in state["argv"])
    observed = json.loads((output / "backend-inputs.json").read_text())
    assert observed["LFFW_RELEASE_BUILD"] == "1"
    assert observed["CONAN_HOME"] == "/tmp/conan-home"
    assert observed["LFFW_BUNDLE_ROOT"] == "/inputs/bundle"
    assert all(
        observed[name] is None
        for name in ("CUDAToolkit_ROOT", "TENSORRT_ROOT", "TENSORACLE_CPP_ROOT")
    )
    assert observed["recipe_sha256"] == receipt["sha256"]
    assert len(observed["bundle_manifest"]["entries"]) == 9


@pytest.mark.req("SP-LFFASTESTWHISPER-024")
@pytest.mark.parametrize("attach_exit,container_exit", [(0, 23), (17, 0)])
def test_release_failure_cannot_leave_a_publishable_candidate(
    release_process, attach_exit, container_exit
):
    """A failed container build fails the shell and removes any unverified final wheel."""
    run, output, _, _ = release_process
    result, state = run(attach_exit=attach_exit, container_exit=container_exit)
    assert state is not None, result.stdout + result.stderr
    assert result.returncode != 0, result.stdout + result.stderr
    assert "Build records and final wheel:" not in result.stdout
    assert not list(output.glob("final/*.whl")), "Failed build left a wheel among final candidates"
    assert (output / "container-exit-code.txt").read_text().strip() == str(container_exit)
    assert (output / "conan.lock").is_file()
    assert (output.parent / "unrelated.txt").read_text() == "keep unrelated output\n"


@pytest.mark.req("SP-LFFASTESTWHISPER-024")
def test_successful_release_retains_its_inspected_candidate(release_process):
    """A successful build keeps its wheel after the shell finishes cleanup."""
    run, output, _, _ = release_process
    result, state = run(attach_exit=0, container_exit=0)
    assert state is not None, result.stdout + result.stderr
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Build records and final wheel:" in result.stdout
    assert len(list(output.glob("final/*.whl"))) == 1
    assert (output / "bundle-inspection.txt").read_text().strip() == "fixture-bundle"


@pytest.mark.req("SP-LFFASTESTWHISPER-024")
def test_release_refuses_existing_output_without_removing_its_files(release_process):
    """A refused existing output keeps its previous wheel, diagnostics and neighboring files."""
    run, output, _, _ = release_process
    (output / "final").mkdir(parents=True)
    (output / "final/previous.whl").write_bytes(b"previous release")
    (output / "previous.log").write_text("previous diagnostics\n")
    result, state = run()
    assert result.returncode != 0
    assert state is None
    assert (output / "final/previous.whl").read_bytes() == b"previous release"
    assert (output / "previous.log").read_text() == "previous diagnostics\n"
    assert (output.parent / "unrelated.txt").read_text() == "keep unrelated output\n"


@pytest.mark.req("SP-LFFASTESTWHISPER-058")
def test_release_rejects_changed_recipe_bytes_before_creating_a_container(release_process):
    """The source assembler verifies recipe bytes before handing a source archive to Docker."""
    run, output, _, _ = release_process
    result, state = run(corrupt_archive=True)
    assert result.returncode != 0
    assert state is None, "A changed recipe archive reached Docker"
    events = [json.loads(line) for line in (output.parent / "calls.jsonl").read_text().splitlines()]
    assert any(event.get("sdist_exit_code") == 0 for event in events)
    assert not list(output.glob("final/*.whl"))
