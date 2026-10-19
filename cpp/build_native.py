"""Build a development plugin against a verified SDK and the installed tensoracle."""

import argparse
import subprocess
from pathlib import Path

import tensoracle
from tensoracle.bundle import inspect_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--tensorrt-root", type=Path, required=True)
    parser.add_argument("--cuda-root", type=Path, default=Path("/usr/local/cuda"))
    args = parser.parse_args()
    bundle = args.bundle_dir.resolve()
    build = args.build_dir.resolve()
    inspect_bundle(bundle)
    helper = Path(tensoracle.__file__).parent / "cpp"
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
            f"-DCUDAToolkit_ROOT={args.cuda_root.resolve()}",
            f"-DTENSORACLE_CPP_ROOT={helper}",
            f"-DTENSORRT_ROOT={args.tensorrt_root.resolve()}",
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
