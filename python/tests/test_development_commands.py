"""Documented commands preserve the prepared namespace-package environment."""

import re
import shlex
from pathlib import Path

import pytest


@pytest.mark.req("SP-LFFASTESTWHISPER-031")
@pytest.mark.req("SP-LFFASTESTWHISPER-033")
def test_documented_uv_commands_do_not_install_the_project_as_editable():
    """Source commands use prepared dependencies without syncing an editable namespace package."""
    root = Path(__file__).resolve().parents[2]
    for guide in (root / "README.md", root / "cpp/README.md"):
        blocks = re.findall(r"(?ms)^```(?:bash|sh|console)\n(.*?)^```", guide.read_text())
        for block in blocks:
            for line in block.replace("\\\n", " ").splitlines():
                if "uv " not in line:
                    continue
                tokens = shlex.split(line)
                if "uv" not in tokens:
                    continue
                command = tokens[tokens.index("uv") + 1 :]
                if command[0] == "run":
                    assert "--no-sync" in command, f"{guide.name}: {line}"
                elif command[0] == "sync":
                    assert "--no-install-project" in command, f"{guide.name}: {line}"
