"""Download the private recipe archive with short-lived Conan source credentials.

Run through the pinned Conan CLI so standalone CMake needs no Python discovery.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from conan import ConanFile
from conan.errors import ConanException


class RecipeArchive(ConanFile):
    def source(self):
        # Shared with the release archive's verification, including inside Docker.
        archive = json.loads(Path(__file__).with_name("recipe-archive.json").read_text())
        url, digest = archive["url"], archive["sha256"]
        # Build backends can set TMPDIR inside the checkout. Credentials must
        # stay outside it, even for those builds (the native target is Linux).
        with TemporaryDirectory(prefix="lffw-recipe-", dir="/tmp") as temporary:
            stage = Path(temporary)
            home = stage / "conan-home"
            home.mkdir(mode=0o700)
            token = subprocess.run(
                ["gh", "auth", "token"], capture_output=True, text=True, check=False
            )
            if token.returncode or not token.stdout.strip():
                raise ConanException("gh could not supply a token for the private recipe archive")
            credentials = home / "source_credentials.json"
            with open(credentials, "x", opener=lambda p, f: os.open(p, f, 0o600)) as output:
                json.dump(
                    {
                        "credentials": [
                            {
                                "url": "https://api.github.com/repos/sync-dev-org/conan-recipes/releases/assets/",
                                "token": token.stdout.strip(),
                                "headers": {"Accept": "application/octet-stream"},
                            }
                        ]
                    },
                    output,
                )
            recipe = stage / "conanfile.py"
            recipe.write_text(
                "from conan import ConanFile\n"
                "from conan.tools.files import download\n"
                "class Download(ConanFile):\n"
                "    def source(self):\n"
                f"        download(self, {url!r}, 'recipes.tar.gz', sha256={digest!r})\n"
            )
            try:
                fetched = subprocess.run(
                    [sys.executable, "-m", "conans.conan", "source", str(recipe), "-vquiet"],
                    env={**os.environ, "CONAN_HOME": str(home)},
                    capture_output=True,
                    text=True,
                    check=False,
                )
            finally:
                credentials.unlink()
            # Never forward a credential-bearing child's diagnostics, including on failure.
            if fetched.returncode:
                raise ConanException(
                    "Cannot download and verify conan-recipes v1.66.0. "
                    "Check the host's gh access and network; child diagnostics are withheld."
                )
            shutil.copyfile(stage / "recipes.tar.gz", Path(self.source_folder) / "recipes.tar.gz")
