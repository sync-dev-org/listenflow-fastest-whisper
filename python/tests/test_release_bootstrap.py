"""Release builds consume verified recipe copies in a fresh Conan home."""

import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def release_source(tmp_path):
    cpp = tmp_path / "source/cpp"
    conan = cpp / "conan"
    conan.mkdir(parents=True)
    shutil.copyfile(ROOT / "cpp/conan/bootstrap.cmake", conan / "bootstrap.cmake")
    requirements = tomllib.loads((ROOT / "python/pyproject.toml").read_text())["build-system"][
        "requires"
    ]
    (pinned_conan,) = [
        requirement for requirement in requirements if requirement.startswith("conan==")
    ]
    (cpp.parent / "pyproject.toml").write_text(
        "[build-system]\nrequires = " + json.dumps([pinned_conan]) + "\n"
    )
    counter = tmp_path / "builds.txt"
    recipe = tmp_path / "recipes/release-probe"
    recipe.mkdir(parents=True)
    (recipe / "conanfile.py").write_text(
        "from pathlib import Path\n"
        "from conan import ConanFile\n"
        "class Probe(ConanFile):\n"
        "    name = 'release-probe'\n"
        "    version = '1.0'\n"
        "    def build(self):\n"
        f"        with Path({str(counter)!r}).open('a') as output:\n"
        "            output.write('build\\n')\n"
    )
    environment = {**os.environ, "CONAN_HOME": str(tmp_path / "conan-home")}
    exported = subprocess.run(
        ["conan", "export", str(recipe), "--format=json"],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    reference = json.loads(exported.stdout)["reference"]
    (cpp / "conan.lock").write_text(
        json.dumps(
            {
                "version": "0.5",
                "requires": [reference],
                "build_requires": [],
                "python_requires": [],
                "config_requires": [],
            }
        )
    )
    (cpp / "conanfile.py").write_text(
        "from conan import ConanFile\n"
        "from conan.tools.cmake import CMakeToolchain\n"
        "class Consumer(ConanFile):\n"
        "    settings = 'os', 'arch', 'compiler', 'build_type'\n"
        "    requires = 'release-probe/1.0'\n"
        "    def generate(self):\n"
        "        CMakeToolchain(self).generate()\n"
    )
    profile = tmp_path / "profile"
    shutil.copyfile(ROOT / "cpp/conan/profiles/linux-gcc-toolset-15", profile)
    with tarfile.open(conan / "recipes.tar.gz", "w:gz") as archive:
        archive.add(tmp_path / "recipes", arcname="recipes")
    (conan / "recipe-archive.json").write_text(
        json.dumps(
            {
                "url": "https://example.invalid/never-fetch-during-release",
                "sha256": hashlib.sha256((conan / "recipes.tar.gz").read_bytes()).hexdigest(),
            }
        )
    )
    script = tmp_path / "install.cmake"
    script.write_text(
        f'include("{conan / "bootstrap.cmake"}")\n'
        f'lffw_install_conan("{profile}" "{tmp_path / "generated"}" FALSE)\n'
    )
    environment["LFFW_RELEASE_BUILD"] = "1"
    return cpp, counter, environment, script


def install(source):
    return subprocess.run(
        ["cmake", "-P", str(source[3])],
        env=source[2],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


@pytest.mark.req("SP-LFFASTESTWHISPER-024")
def test_release_build_rebuilds_cached_dependencies(release_source):
    """Release installs rebuild their dependencies even when a cached binary is available."""
    for _ in range(2):
        result = install(release_source)
        assert result.returncode == 0, result.stdout + result.stderr
    assert release_source[1].read_text().splitlines() == ["build", "build"]


@pytest.mark.req("SP-LFFASTESTWHISPER-058")
@pytest.mark.parametrize("damage", ["checksum", "revision", "missing"])
def test_release_rejects_invalid_recipe_copies_even_with_a_warm_cache(release_source, damage):
    """A release refuses a missing archive, changed bytes or a revision outside its lock."""
    cpp = release_source[0]
    if damage == "checksum":
        with (cpp / "conan/recipes.tar.gz").open("ab") as output:
            output.write(b"changed")
        expected = "checksum"
    elif damage == "revision":
        lock = json.loads((cpp / "conan.lock").read_text())
        lock["requires"][0] = "release-probe/1.0#" + "0" * 32
        (cpp / "conan.lock").write_text(json.dumps(lock))
        expected = "lock requires"
    else:
        (cpp / "conan/recipes.tar.gz").unlink()
        expected = "recipes.tar.gz"
    result = install(release_source)
    assert result.returncode != 0
    assert expected in " ".join((result.stdout + result.stderr).lower().split())
    assert not release_source[1].exists()
