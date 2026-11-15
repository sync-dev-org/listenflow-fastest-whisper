"""Build development plugins against the revision-locked Conan SDKs."""

import argparse
import subprocess
from pathlib import Path

from tensoracle.bundle import inspect_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    args = parser.parse_args()
    bundle = args.bundle_dir.resolve()
    build = args.build_dir.resolve()
    inspect_bundle(bundle)
    subprocess.run(
        [
            "cmake",
            "-S",
            str(Path(__file__).resolve().parent),
            "-B",
            str(build),
            "-G",
            "Ninja",
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DLFFW_BUNDLE_ROOT={bundle}",
        ],
        check=True,
    )
    subprocess.run(["cmake", "--build", str(build), "--parallel", "2"], check=True)
    library = build / "liblffw_cubin_plugin.so"
    inspect_bundle(bundle, plugin=build / "liblffw_fc1_plugin.so")
    inspect_bundle(bundle, plugin=library)
    print(build / "liblffw_fc1_plugin.so")
    print(library)


if __name__ == "__main__":
    main()
