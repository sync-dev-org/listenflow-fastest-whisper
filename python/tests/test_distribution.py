"""Check the sources and public metadata delivered through PyPI."""

import email
import os
import re
import shutil
import subprocess
import tarfile
import tomllib
import unicodedata
import zipfile
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit

import pytest
import yaml
from packaging.requirements import Requirement

from test_cubin_plugin import bundle, native_plugin  # noqa: F401 -- pytest fixtures

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
PUBLIC_REPOSITORY = "https://github.com/sync-dev-org/listenflow-fastest-whisper"


@pytest.mark.req("SP-LFFASTESTWHISPER-056")
def test_runtime_dependencies_leave_tensorrt_to_the_host():
    """Installing the package does not request a TensorRT wheel, directly or by extra."""
    project = tomllib.loads((PROJECT / "pyproject.toml").read_text())
    dependencies = list(map(Requirement, project["project"]["dependencies"]))
    assert not any(requirement.name.startswith("tensorrt") for requirement in dependencies)
    tensoracle = next(requirement for requirement in dependencies if requirement.name == "tensoracle")
    assert not tensoracle.extras


@pytest.mark.req("SP-LFFASTESTWHISPER-037")
def test_tensoracle_dependency_has_only_the_gpu_verified_minimum():
    """The package accepts tensoracle 2.6.0 and later without an upper bound."""
    project = tomllib.loads((PROJECT / "pyproject.toml").read_text())
    dependencies = map(Requirement, project["project"]["dependencies"])
    tensoracle = next(requirement for requirement in dependencies if requirement.name == "tensoracle")
    assert str(tensoracle.specifier) == ">=2.6.0"


@pytest.mark.req("SP-LFFASTESTWHISPER-060")
def test_development_installs_tensoracle_with_its_tensorrt_extra():
    """The default development group installs the TensorRT Python binding through tensoracle."""
    project = tomllib.loads((PROJECT / "pyproject.toml").read_text())
    development = {
        requirement.name: requirement
        for requirement in map(Requirement, project["dependency-groups"]["dev"])
    }
    assert development["tensoracle"].extras == {"tensorrt"}
    assert str(development["tensoracle"].specifier) == ">=2.6.0"


@pytest.mark.req("SP-LFFASTESTWHISPER-037")
def test_development_lock_selects_the_gpu_verified_tensoracle():
    """The reproducible development environment selects tensoracle 2.6.0."""
    lock = tomllib.loads((PROJECT / "uv.lock").read_text())
    assert next(package for package in lock["package"] if package["name"] == "tensoracle")[
        "version"
    ] == "2.6.0"


@pytest.fixture(scope="module")
def distributions(request):
    bundle_dir, _ = request.getfixturevalue("native_plugin")
    build_env = {
        "LFFW_BUNDLE_ROOT": str(bundle_dir),
    }
    temporary_root = ROOT / ".nf" / "tmp"
    temporary_root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="distribution-", dir=temporary_root) as temporary:
        output = Path(temporary) / "dist"
        result = subprocess.run(
            ["uv", "build", "--out-dir", str(output)],
            cwd=PROJECT,
            env={**os.environ, **build_env, "TMPDIR": temporary},
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        yield {
            "sdist": next(output.glob("*.tar.gz")),
            "wheel": next(output.glob("*.whl")),
            "env": build_env,
        }


def _distribution_files(distributions):
    with tarfile.open(distributions["sdist"]) as archive:
        sdist = {
            PurePosixPath(*PurePosixPath(member.name).parts[1:]): archive.extractfile(member).read()
            for member in archive.getmembers()
            if member.isfile()
        }
    with zipfile.ZipFile(distributions["wheel"]) as archive:
        wheel = {
            PurePosixPath(name): archive.read(name)
            for name in archive.namelist()
            if not name.endswith("/")
        }
    return {"sdist": sdist, "wheel": wheel}


@pytest.mark.req("SP-LFFASTESTWHISPER-007")
def test_sdist_contains_only_package_sources_and_build_inputs(distributions):
    """The source archive includes the C++ build inputs and omits Python tests and audio fixtures."""
    with tarfile.open(distributions["sdist"]) as archive:
        members = [member.name for member in archive.getmembers()]
        assert len(members) == len(set(members)), "source paths must occur only once"
    paths = _distribution_files(distributions)["sdist"]
    package = PurePosixPath("src/listenflow/fastest_whisper")
    allowed_files = {
        PurePosixPath(name)
        for name in (
            "README.md",
            "LICENSE",
            "pyproject.toml",
            "CMakeLists.txt",
            "PKG-INFO",
            ".gitignore",
        )
    }
    assert not [
        str(path)
        for path in paths
        if path not in allowed_files and package not in path.parents and path.parts[0] != "cpp"
    ]
    assert allowed_files <= paths.keys()
    assert package / "__init__.py" in paths
    assert package / "assets/mel_filters.npz" in paths
    assert (
        paths[package / "assets/fp8-scales.json"]
        == (PROJECT / "src/listenflow/fastest_whisper/assets/fp8-scales.json").read_bytes()
    )
    assert PurePosixPath("cpp/CMakeLists.txt") in paths
    assert PurePosixPath("cpp/README.md") in paths
    assert PurePosixPath("cpp/fc1_plugin.cpp") in paths
    assert PurePosixPath("cpp/fc1_registration.cpp") in paths


@pytest.mark.req("SP-LFFASTESTWHISPER-023")
def test_wheel_contains_only_the_package_and_distribution_metadata(distributions):
    """The wheel installs only the Python package, three native libraries and metadata."""
    paths = _distribution_files(distributions)["wheel"]
    top_levels = {path.parts[0] for path in paths}
    assert "listenflow" in top_levels
    assert len(top_levels) == 2
    assert all(name == "listenflow" or name.endswith(".dist-info") for name in top_levels)
    assert PurePosixPath("listenflow/fastest_whisper/__init__.py") in paths
    assert PurePosixPath("listenflow/fastest_whisper/assets/mel_filters.npz") in paths
    assert (
        paths[PurePosixPath("listenflow/fastest_whisper/assets/fp8-scales.json")]
        == (PROJECT / "src/listenflow/fastest_whisper/assets/fp8-scales.json").read_bytes()
    )
    package = PurePosixPath("listenflow/fastest_whisper")
    native = {
        package / name
        for name in (
            "_native.abi3.so",
            "liblffw_runtime.so",
            "liblffw_fc1_plugin.so",
        )
    }
    assert {path for path, data in paths.items() if data.startswith(b"\x7fELF")} == native
    assert {path for path in paths if ".so" in path.name} == native
    assert not [path for path in paths if path.suffix in {".h", ".hpp", ".cmake", ".a"}]

    wheel_metadata = next(
        data for path, data in paths.items() if str(path).endswith(".dist-info/WHEEL")
    )
    assert b"Root-Is-Purelib: false" in wheel_metadata
    assert b"Tag: cp312-abi3-" in wheel_metadata


@pytest.mark.req("SP-LFFASTESTWHISPER-054")
@pytest.mark.req("SP-LFFASTESTWHISPER-055")
@pytest.mark.req("SP-LFFASTESTWHISPER-056")
@pytest.mark.req("SP-LFFASTESTWHISPER-061")
def test_wheel_native_libraries_use_relative_dependency_paths(distributions, tmp_path):
    """The wheel reads tensoracle from its sibling wheel and TensorRT from the host loader."""
    paths = _distribution_files(distributions)["wheel"]
    for path, data in paths.items():
        if not data.startswith(b"\x7fELF"):
            continue
        library = tmp_path / path.name
        library.write_bytes(data)
        result = subprocess.run(
            ["readelf", "--dynamic", str(library)],
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, "LC_ALL": "C"},
        ).stdout
        needed = re.findall(r"\(NEEDED\).*?\[([^]]+)\]", result)
        assert not any(name.startswith("libcudart.so") for name in needed)
        runpaths = re.findall(r"\((?:RPATH|RUNPATH)\).*?\[([^]]+)\]", result)
        entries = [entry for group in runpaths for entry in group.split(":")]
        assert all(entry.startswith("$ORIGIN") for entry in entries), result
        assert not any("tensorrt" in entry.lower() for entry in entries), result
        if path.name == "liblffw_runtime.so":
            assert "libtensoracle.so.2" in needed, result
            assert "$ORIGIN/../../tensoracle/cpp/lib" in entries, result
        elif path.name == "liblffw_fc1_plugin.so":
            assert "libtensoracle.so.2" not in needed, result


@pytest.mark.req("SP-LFFASTESTWHISPER-037")
@pytest.mark.req("SP-LFFASTESTWHISPER-056")
def test_wheel_metadata_describes_the_public_package(distributions):
    """PyPI receives the README, author, MIT license, supported Python versions, and public URL."""
    files = _distribution_files(distributions)["wheel"]
    path = next(path for path in files if str(path).endswith(".dist-info/METADATA"))
    metadata = email.message_from_bytes(files[path])
    assert metadata.get_payload(decode=True).decode("utf-8") == (ROOT / "README.md").read_text(
        encoding="utf-8"
    )
    assert metadata["Description-Content-Type"] == "text/markdown"
    assert metadata["Author-email"] == "minamik <mia@sync.dev>"
    assert metadata["License-Expression"] == "MIT"
    assert metadata.get_all("License-File") == ["LICENSE"]
    assert files[path.parent / "licenses/LICENSE"] == (ROOT / "LICENSE").read_bytes()
    assert metadata["Requires-Python"] == "<3.14,>=3.12"
    assert set(metadata.get_all("Project-URL", [])) == {f"Repository, {PUBLIC_REPOSITORY}"}
    assert {
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    } <= set(metadata.get_all("Classifier", []))
    project = tomllib.loads((PROJECT / "pyproject.toml").read_text())
    assert {str(Requirement(value)) for value in metadata.get_all("Requires-Dist", [])} == {
        str(Requirement(value)) for value in project["project"]["dependencies"]
    }


@pytest.mark.req("SP-LFFASTESTWHISPER-007")
def test_source_archive_rebuilds_the_native_wheel(distributions, tmp_path):
    """The extracted source archive builds a wheel without access to its original checkout."""
    with tarfile.open(distributions["sdist"]) as archive:
        archive.extractall(tmp_path, filter="data")
    source = next(tmp_path.glob("listenflow_fastest_whisper-*"))
    output = tmp_path / "rebuilt"
    built = subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(output)],
        cwd=source,
        env={**os.environ, **distributions["env"]},
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert built.returncode == 0, built.stdout + built.stderr
    with zipfile.ZipFile(next(output.glob("*.whl"))) as archive:
        rebuilt = {name: archive.read(name) for name in archive.namelist()}
    with zipfile.ZipFile(distributions["wheel"]) as archive:
        assert set(rebuilt) == set(archive.namelist())
        for name, data in rebuilt.items():
            if name.endswith(("/METADATA", "/WHEEL", "/LICENSE", "/fp8-scales.json")):
                assert data == archive.read(name)


@pytest.mark.req("SP-LFFASTESTWHISPER-007")
def test_release_version_requires_an_exact_tag_and_survives_source_archiving(tmp_path):
    """An exact v tag supplies the release version; a later commit uses the development fallback."""
    checkout = tmp_path / "checkout"
    project = checkout / "python"
    shutil.copytree(PROJECT / "src", project / "src", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(ROOT / "cpp", checkout / "cpp", ignore=shutil.ignore_patterns("__pycache__"))
    for name in ("pyproject.toml", "CMakeLists.txt"):
        shutil.copyfile(PROJECT / name, project / name)
    for name in ("README.md", "LICENSE"):
        shutil.copyfile(ROOT / name, checkout / name)
        (project / name).symlink_to(Path("..") / name)
    commands = (
        ["git", "init", "--quiet"],
        ["git", "add", "."],
        [
            "git",
            "-c",
            "user.name=Packaging test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "Source",
        ],
        ["git", "tag", "v8.7.6"],
    )
    for command in commands:
        subprocess.run(command, cwd=checkout, check=True, capture_output=True)
    for version in ("8.7.6", "0.0.0.dev0"):
        output = tmp_path / version
        built = subprocess.run(
            ["uv", "build", "--sdist", "--out-dir", str(output)],
            cwd=project,
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
        assert built.returncode == 0, built.stdout + built.stderr
        assert next(output.glob("*.tar.gz")).name == f"listenflow_fastest_whisper-{version}.tar.gz"
        if version == "8.7.6":
            unpacked = tmp_path / "tagged-source"
            with tarfile.open(next(output.glob("*.tar.gz"))) as archive:
                archive.extractall(unpacked, filter="data")
            built = subprocess.run(
                [
                    "uv",
                    "build",
                    "--wheel",
                    "-Cwheel.cmake=false",
                    "-Cwheel.py-api=py3",
                    "--out-dir",
                    str(output),
                ],
                cwd=next(unpacked.iterdir()),
                capture_output=True,
                text=True,
                check=False,
                timeout=120,
            )
            assert built.returncode == 0, built.stdout + built.stderr
            assert next(output.glob("*.whl")).name.startswith("listenflow_fastest_whisper-8.7.6-")
            (checkout / "after-tag").touch()
            subprocess.run(["git", "add", "after-tag"], cwd=checkout, check=True)
            subprocess.run(
                [
                    "git",
                    "-c",
                    "user.name=Packaging test",
                    "-c",
                    "user.email=test@example.invalid",
                    "commit",
                    "--quiet",
                    "-m",
                    "After tag",
                ],
                cwd=checkout,
                check=True,
                capture_output=True,
            )


def test_distributions_omit_paths_excluded_from_the_public_repository(distributions):
    """Both archives respect the public repository's exclusions when release tooling exists."""
    settings = ROOT / "noren.yaml"
    if not settings.is_file():
        pytest.skip("the public repository does not include release tooling")
    config = yaml.safe_load(settings.read_text(encoding="utf-8"))
    exclusions = [
        PurePosixPath(line.strip().strip("/"))
        for line in config["exclude"].splitlines()
        if line.strip()
    ]
    for paths in _distribution_files(distributions).values():
        assert not [
            str(path)
            for path in paths
            if any(path == excluded or excluded in path.parents for excluded in exclusions)
        ]


def test_distributions_contain_no_restricted_terms(distributions):
    """Archive file names and contents, including metadata, pass case-insensitive word checks."""
    word_list = ROOT / "noren-words.txt"
    if not word_list.is_file():
        pytest.skip("the public repository does not include the private word list")
    words = [
        unicodedata.normalize("NFKC", word.strip()).casefold()
        for word in word_list.read_text(encoding="utf-8").splitlines()
        if word.strip()
    ]
    violations = []
    for kind, files in _distribution_files(distributions).items():
        for path, data in files.items():
            normalized = unicodedata.normalize(
                "NFKC", str(path) + "\n" + data.decode("utf-8", errors="replace")
            ).casefold()
            if any(word in normalized for word in words):
                violations.append(f"{kind}: {path}")
    assert not violations, f"Restricted terms found in archive files: {violations}"


def test_readme_links_and_images_resolve_on_pypi():
    """README targets use absolute URLs or page anchors and omit private issue references."""

    class Targets(HTMLParser):
        def __init__(self):
            super().__init__()
            self.targets = []

        def handle_starttag(self, tag, attrs):
            attribute = {"a": "href", "img": "src"}.get(tag)
            if attribute:
                self.targets.extend(value for key, value in attrs if key == attribute)

    text = (ROOT / "README.md").read_text(encoding="utf-8")
    prose = re.sub(r"(?ms)^(```|~~~).*?^\1[^\n]*$", "", text)
    parser = Targets()
    parser.feed(prose)
    targets = parser.targets + re.findall(r"!?\[[^\]]*\]\(\s*<?([^\s)>]+)", prose)
    targets += re.findall(r"(?m)^\s*\[[^\]]+\]:\s*<?([^\s>]+)", prose)
    assert targets
    assert not [
        target
        for target in targets
        if not (target and (urlsplit(target).scheme or target.startswith("#")))
    ]
    assert not [target for target in targets if target.startswith(f"{PUBLIC_REPOSITORY}/issues")]


def test_install_instructions_use_the_pypi_package():
    """The primary installation command uses the published distribution with its core dependency."""
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    install = text.split("## Install\n", 1)[1].split("\n## ", 1)[0]
    assert "pip install listenflow-fastest-whisper" in install
    assert "[tool.uv.sources]" not in install
