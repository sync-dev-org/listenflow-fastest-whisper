#!/usr/bin/python3
"""External build tools for the release shell's process-boundary tests.

Docker records its inputs and runs the real --inside shell in bubblewrap.
The external compiler/backend stops after recording its actual environment.
No real Docker, GPU, credentials or recipe service is contacted.
"""

import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

TOKEN = b"release-test-only-private-token"
tool = Path(sys.argv[0]).name
arguments = sys.argv[1:]

# This branch also runs inside the ephemeral container filesystem. It needs
# neither the host test configuration nor a Python package installation.
if tool == "python":
    if arguments[:2] == ["-m", "pip"]:
        if "install" in arguments:
            package = Path("/tmp/backend/scikit_build_core")
            package.mkdir(parents=True)
            (package / "__init__.py").write_text("")
            (package / "build.py").write_text(
                "import hashlib, json, os\nfrom pathlib import Path\n"
                "def build_wheel(*args, **kwargs):\n"
                "    source=Path.cwd()\n"
                "    archive=source/'cpp/conan/recipes.tar.gz'\n"
                "    bundle=Path(os.environ['LFFW_BUNDLE_ROOT'])\n"
                "    result={name:os.environ.get(name) for name in "
                "('LFFW_RELEASE_BUILD','LFFW_BUNDLE_ROOT','CONAN_HOME',"
                "'CUDAToolkit_ROOT','TENSORRT_ROOT','TENSORACLE_CPP_ROOT')}\n"
                "    result['recipe_sha256']=hashlib.sha256(archive.read_bytes()).hexdigest() if archive.exists() else None\n"
                "    result['bundle_manifest']=json.loads((bundle/'bundle.json').read_text())\n"
                "    Path('/release/backend-inputs.json').write_text(json.dumps(result))\n"
                "    raise SystemExit(int(os.environ['TEST_BUILD_EXIT_CODE']))\n"
            )
        raise SystemExit(0)
    os.execv("/usr/bin/python3", ["python3", *arguments])

context_path = Path(os.environ["LFFW_RELEASE_TEST_CONTEXT"])
context = json.loads(context_path.read_text())
log = Path(context["log"])


def save_event(**values):
    with log.open("a") as output:
        output.write(json.dumps({"tool": tool, "argv": arguments, **values}) + "\n")


def mounts_from(command):
    return [
        dict(
            item.split("=", 1) if "=" in item else (item, "")
            for item in command[index + 1].split(",")
        )
        for index, value in enumerate(command)
        if value == "--mount"
    ]


def save_archive(path):
    with tarfile.open(path) as archive:
        files = {
            "/".join(Path(member.name).parts[1:]): archive.extractfile(member).read()
            for member in archive.getmembers()
            if member.isfile()
        }
    receipt = files.get("cpp/conan/recipe-archive.json")
    embedded = files.get("cpp/conan/recipes.tar.gz")
    payloads = dict(files)
    if embedded is not None:
        with tarfile.open(fileobj=io.BytesIO(embedded), mode="r:gz") as recipes:
            payloads.update(
                {
                    "recipe/" + member.name: recipes.extractfile(member).read()
                    for member in recipes.getmembers()
                    if member.isfile()
                }
            )
    return {
        "files": sorted(files),
        "receipt": None if receipt is None else json.loads(receipt),
        "recipe_sha256": None if embedded is None else hashlib.sha256(embedded).hexdigest(),
        "token_found": any(TOKEN in data for data in payloads.values()),
        "credential_files": [
            name for name in payloads if Path(name).name == "source_credentials.json"
        ],
    }


if tool == "gh":
    raise SystemExit("A release test must never request real credentials")
if tool == "cmake":
    save_event()
    raise SystemExit(0)
if tool == "conan":
    save_event()
    if arguments[0] == "source":
        data = Path(context["recipes"]).read_bytes()
        Path(arguments[1], "recipes.tar.gz").write_bytes(
            data + b"corrupt" if context["corrupt_archive"] else data
        )
    elif arguments[:2] == ["graph", "info"]:
        print(
            json.dumps(
                {
                    "graph": {
                        "nodes": {
                            "1": {"ref": "cuda-toolkit/13.2.2#fixture", "package_id": "fixture"}
                        }
                    }
                }
            )
        )
    elif arguments[:2] == ["cache", "path"]:
        print(context["cuda"])
    else:
        raise SystemExit("Unexpected Conan operation")
    raise SystemExit(0)
if tool == "uv":
    save_event()
    if arguments[0] == "build":
        # The archive builder and source inclusion are real and offline.
        built = subprocess.run(
            [context["uv"], "build", "--offline", "--python", context["python"], *arguments[1:]],
            check=False,
        )
        save_event(sdist_exit_code=built.returncode)
        raise SystemExit(built.returncode)
    if arguments[:2] == ["pip", "compile"]:
        Path(arguments[arguments.index("--output-file") + 1]).write_text(
            "# backend is supplied at the external boundary\n"
        )
        raise SystemExit(0)
    if arguments[0] == "run":
        command = arguments[arguments.index("python") + 1 :]
        environment = {**os.environ, "PYTHONPATH": context["external_modules"]}
        os.execve("/usr/bin/python3", ["python3", *command], environment)
    print("fixture-tool==1.0")
    raise SystemExit(0)
if tool != "docker":
    raise SystemExit("Unexpected external tool: " + tool)

state_path = Path(context["state"])
if arguments[0] in {"create", "run"}:
    mounts = mounts_from(arguments)
    release = Path(next(item["src"] for item in mounts if item["dst"] == "/release"))
    inputs = next((Path(item["src"]) for item in mounts if item["dst"] == "/inputs"), release)
    archive = inputs / "source.tar.gz" if inputs != release else next(release.glob("*.tar.gz"))
    state = {
        "mounts": mounts,
        "release": str(release),
        "argv": arguments,
        "archive": save_archive(archive),
        "client_environment": dict(os.environ),
    }
    state_path.write_text(json.dumps(state))
    save_event(state=state)
    if arguments[0] == "create":
        print("release-test-container")
        raise SystemExit(0)
if arguments[0] == "start" or arguments[0] == "run":
    state = json.loads(state_path.read_text())
    release = Path(state["release"])
    if context["execute_inside"]:
        command = [context["bwrap"], "--unshare-all", "--die-with-parent", "--tmpfs", "/"]
        for directory in ("/usr", "/bin", "/lib", "/lib64"):
            if Path(directory).exists():
                command += ["--ro-bind", directory, directory]
        command += [
            "--dev",
            "/dev",
            "--proc",
            "/proc",
            "--tmpfs",
            "/tmp",
            "--ro-bind",
            context["bin"],
            "/test-bin",
        ]
        for item in state["mounts"]:
            if "libcuda.so" not in item["dst"]:
                command += [
                    "--ro-bind" if "readonly" in item else "--bind",
                    item["src"],
                    item["dst"],
                ]
        environment = {
            "PATH": "/test-bin:/usr/bin:/bin",
            "TEST_BUILD_EXIT_CODE": str(context["container_exit"]),
        }
        for index, value in enumerate(state["argv"]):
            if value in {"--env", "-e"}:
                name, separator, setting = state["argv"][index + 1].partition("=")
                environment[name] = setting if separator else state["client_environment"][name]
        completed = subprocess.run(
            [*command, "bash", "/build.sh", "--inside"], env=environment, check=False
        )
        save_event(inside_returncode=completed.returncode)
        if not (release / "backend-inputs.json").exists():
            raise SystemExit("Inside shell did not reach the external backend")
    else:
        (release / "container-exit-code.txt").write_text(str(context["container_exit"]) + "\n")
        candidate = (
            release / "final/listenflow_fastest_whisper-1.2.3-cp312-abi3-manylinux_2_28_x86_64.whl"
        )
        candidate.parent.mkdir()
        with zipfile.ZipFile(candidate, "w") as archive:
            archive.writestr("listenflow/fastest_whisper/liblffw_fc1_plugin.so", b"unverified")
        plugin = release / "inspect/listenflow/fastest_whisper/liblffw_fc1_plugin.so"
        plugin.parent.mkdir(parents=True)
        plugin.write_bytes(b"unverified")
        save_event(candidate_created=True)
    # Deliberately independent from the build's actual exit status.
    raise SystemExit(context["attach_exit"])
if arguments[0] == "inspect":
    print("[]")
elif arguments[:3] == ["buildx", "imagetools", "inspect"]:
    print('{"config":{"digest":"sha256:fixture"}}')
else:
    save_event()
