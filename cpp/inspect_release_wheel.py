"""Check the release tag, package contents and every ELF against manylinux 2.28."""

import argparse
import hashlib
import json
import os
import re
import subprocess
import zipfile
from pathlib import Path, PurePosixPath

TAG = "cp312-abi3-manylinux_2_28_x86_64"
PACKAGE = "listenflow/fastest_whisper/"
NATIVE = {"_native.abi3.so", "liblffw_runtime.so", "liblffw_fc1_plugin.so"}
LIMITS = {"GLIBC": "2.28", "GLIBCXX": "3.4.25", "CXXABI": "1.3.11"}


def version_key(version):
    parts = tuple(map(int, version.split(".")))
    return parts + (0,) * (3 - len(parts))


def symbol_versions(text):
    result = {}
    for name in LIMITS:
        versions = re.findall(rf"\b{name}_(\d+(?:\.\d+)+)\b", text)
        result[name] = max(versions, key=version_key) if versions else None
    return result


def validate_versions(versions):
    for name, limit in LIMITS.items():
        version = versions[name]
        if version is not None and version_key(version) > version_key(limit):
            raise ValueError(f"{name}_{version} exceeds manylinux 2.28's {name}_{limit}")


def readelf(path, option):
    return subprocess.run(
        ["readelf", option, "--wide", str(path)],
        env={**os.environ, "LC_ALL": "C"},
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def inspect_wheel(wheel, output):
    wheel, output = Path(wheel), Path(output)
    if not wheel.name.endswith(f"-{TAG}.whl"):
        raise ValueError(f"Expected release tag {TAG}: {wheel.name}")
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "wheel": wheel.name,
        "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        "elf": {},
    }
    with zipfile.ZipFile(wheel) as archive:
        names = [name for name in archive.namelist() if not name.endswith("/")]
        if len(names) != len(set(names)):
            raise ValueError("Wheel has duplicate paths")
        metadata = {
            name.split("/", 1)[0] for name in names if name.split("/", 1)[0].endswith(".dist-info")
        }
        if len(metadata) != 1:
            raise ValueError("Expected one distribution metadata directory")
        (info,) = metadata
        files = {}
        for name in names:
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Unsafe wheel path: {name}")
            if not name.startswith((PACKAGE, info + "/")):
                raise ValueError(f"Unexpected wheel member: {name}")
            if path.suffix in {".h", ".hpp", ".cmake", ".a"}:
                raise ValueError(f"Build input in wheel: {name}")
            files[name] = archive.read(name)
        native = {
            name
            for name, data in files.items()
            if data.startswith(b"\x7fELF") or ".so" in PurePosixPath(name).name
        }
        if native != {PACKAGE + name for name in NATIVE}:
            raise ValueError(f"Unexpected native files: {sorted(native)}")
        tags = re.findall(r"^Tag: (.*)$", files[info + "/WHEEL"].decode(), re.MULTILINE)
        if tags != [TAG]:
            raise ValueError(f"Unexpected WHEEL tags: {tags}")
        report["files"] = sorted(files)
        for name in sorted(native):
            path = output / PurePosixPath(name).name
            path.write_bytes(files[name])
            header = readelf(path, "--file-header")
            dynamic = readelf(path, "--dynamic")
            versions = readelf(path, "--version-info")
            for suffix, text in (("header", header), ("dynamic", dynamic), ("versions", versions)):
                (output / f"{path.name}.{suffix}.txt").write_text(text)
            if "ELF64" not in header or "Advanced Micro Devices X86-64" not in header:
                raise ValueError(f"Expected ELF64 x86-64: {name}")
            needed = re.findall(r"\(NEEDED\).*?\[([^]]+)\]", dynamic)
            search = {
                kind.lower(): [
                    entry
                    for group in re.findall(rf"\({kind}\).*?\[([^]]+)\]", dynamic)
                    for entry in group.split(":")
                ]
                for kind in ("RPATH", "RUNPATH")
            }
            if any(library.startswith("libcudart.so") for library in needed):
                raise ValueError(f"Dynamic CUDA runtime dependency: {name}")
            expected_paths = {
                "liblffw_runtime.so": ["$ORIGIN/../../tensoracle/cpp/lib"],
                "_native.abi3.so": ["$ORIGIN"],
                "liblffw_fc1_plugin.so": [],
            }
            if search["rpath"] or search["runpath"] != expected_paths[path.name]:
                raise ValueError(f"Unexpected library search path: {name}: {search}")
            if path.name == "liblffw_runtime.so" and "libtensoracle.so.2" not in needed:
                raise ValueError("Inference library does not link libtensoracle.so.2")
            maximum = symbol_versions(versions)
            validate_versions(maximum)
            report["elf"][path.name] = {"needed": needed, **search, "symbol_versions": maximum}
            # Keep text records, rather than another extracted copy of the wheel.
            path.unlink()
    (output / "inspection.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(inspect_wheel(arguments.wheel, arguments.output), indent=2))
