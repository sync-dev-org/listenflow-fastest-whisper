"""Generate the sm86, sm89 and sm120 cubins and their embedded C++ source on CPU."""

import argparse
from pathlib import Path

from kernels import kernel_specs
from tensoracle.producer import compile_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = compile_bundle(kernel_specs(), args.output_dir)
    print(manifest["bundle_sha256"])


if __name__ == "__main__":
    main()
