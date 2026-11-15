"""Private recipe downloads keep credentials private and remove them after use."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.req("SP-LFFASTESTWHISPER-054")
@pytest.mark.req("SP-LFFASTESTWHISPER-057")
@pytest.mark.req("SP-LFFASTESTWHISPER-060")
def test_native_dependency_graph_uses_locked_conan_sdks():
    """Native builds obtain tensoracle, TensorRT and CUDA from revision-locked Conan packages."""
    cpp = ROOT / "cpp"
    profile = cpp / "conan/profiles/linux-gcc-toolset-15"
    result = subprocess.run(
        [
            "conan",
            "graph",
            "info",
            str(cpp),
            f"--profile:host={profile}",
            f"--profile:build={profile}",
            f"--lockfile={cpp / 'conan.lock'}",
            "--no-remote",
            "--format=json",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    nodes = [node for node in json.loads(result.stdout)["graph"]["nodes"].values() if node["ref"]]
    by_name = {node["ref"].split("/", 1)[0]: node for node in nodes}
    assert {"tensoracle", "tensorrt", "cuda-toolkit"} <= by_name.keys()
    lock = json.loads((cpp / "conan.lock").read_text())
    for name in ("tensoracle", "tensorrt"):
        assert by_name[name]["context"] == "host"
        assert by_name[name]["ref"] in lock["requires"]
    assert by_name["tensoracle"]["ref"] == "tensoracle/2.6.0#7b6a0f1572c2d5aacf186fdd7f965f22"
    assert by_name["tensoracle"]["options"]["shared"] == "True"
    assert by_name["cuda-toolkit"]["context"] == "build"
    assert by_name["cuda-toolkit"]["ref"] in lock["build_requires"]
    assert by_name["cuda-toolkit"]["options"]["mode"] == "redist"


@pytest.mark.req("SP-LFFASTESTWHISPER-057")
@pytest.mark.parametrize("download_status", [0, 9])
def test_recipe_download_removes_credentials_and_withholds_child_diagnostics(
    tmp_path, download_status
):
    """Successful and failed downloads leave no credentials or token in their output."""
    conan = shutil.which("conan")
    assert conan, "install the pinned Conan CLI before testing recipe downloads"
    # Use the real Conan interpreter, recipe and files. Only gh and the network
    # download subprocess are replaced at their external process boundary.
    token = "test-only-private-recipe-token"
    build_tmp = tmp_path / "build-tmp"
    build_tmp.mkdir()
    recipe = tmp_path / "conanfile.py"
    recipe.write_text(
        f"""import json
import os
from pathlib import Path
import runpy
import stat
from subprocess import CompletedProcess
from unittest.mock import patch
from conan import ConanFile
from conan.errors import ConanException

class Check(ConanFile):
    def source(self):
        acquire = runpy.run_path({str(ROOT / "cpp/conan/fetch_recipe.py")!r})['RecipeArchive']
        credentials_paths = []
        def external_run(command, **kwargs):
            if command == ['gh', 'auth', 'token']:
                return CompletedProcess(command, 0, {token!r} + '\\n', '')
            assert {token!r} not in repr(command)
            credentials = Path(kwargs['env']['CONAN_HOME']) / 'source_credentials.json'
            credentials_paths.append(credentials)
            assert not credentials.is_relative_to(Path(os.environ['TMPDIR']))
            assert stat.S_IMODE(credentials.stat().st_mode) == 0o600
            assert stat.S_IMODE(credentials.parent.stat().st_mode) == 0o700
            entry, = json.loads(credentials.read_text())['credentials']
            assert entry['token'] == {token!r}
            assert entry['url'] == 'https://api.github.com/repos/sync-dev-org/conan-recipes/releases/assets/'
            assert entry['headers'] == {{'Accept': 'application/octet-stream'}}
            (credentials.parent.parent / 'recipes.tar.gz').write_bytes(b'verified archive')
            return CompletedProcess(command, {download_status}, {token!r}, {token!r})
        try:
            with patch('subprocess.run', side_effect=external_run):
                acquire.source(self)
        except ConanException as error:
            assert {download_status} != 0
            assert {token!r} not in str(error)
            assert 'Cannot download and verify' in str(error)
        else:
            assert {download_status} == 0
            assert (Path(self.source_folder) / 'recipes.tar.gz').read_bytes() == b'verified archive'
        assert len(credentials_paths) == 1
        assert not credentials_paths[0].exists()
        assert not credentials_paths[0].parent.exists()
"""
    )
    checked = subprocess.run(
        [conan, "source", str(recipe), "-vwarning"],
        env={**os.environ, "CONAN_HOME": str(tmp_path / "conan-home"), "TMPDIR": str(build_tmp)},
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert token not in checked.stdout + checked.stderr
    assert checked.returncode == 0, checked.stdout + checked.stderr
