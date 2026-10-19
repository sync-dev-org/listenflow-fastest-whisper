"""Load the required fc1 implementation shipped inside the Python package."""

import ctypes
from importlib.resources import files
from pathlib import Path

import numpy as np
import tensorrt as trt


def _fc1_library() -> Path:
    library = files(__package__).joinpath("liblffw_fc1_plugin.so")
    if library.is_file():
        return Path(str(library))
    # Editable installs use the independently built C++ target. Restrict that
    # lookup to an actual source checkout, rather than an installed wheel.
    source = Path(__file__).resolve()
    if source.parents[2].name == "src":
        root = source.parents[4]
        development = root / "build/cubin-plugin/liblffw_fc1_plugin.so"
        if (root / "cpp/CMakeLists.txt").is_file() and development.is_file():
            return development
    raise FileNotFoundError(
        f"Required fp16 fc1 plugin library is missing: {library}. "
        "Reinstall a complete native wheel or build the package as described in cpp/README.md."
    )


def _fc1_plugin():
    # open_engine registers and retains this library before invoking the graph.
    library = ctypes.CDLL(str(_fc1_library()))
    digest = (ctypes.c_char * 65).in_dll(library, "tensoracle_bundle_sha256").value
    creator = trt.get_plugin_registry().get_creator(
        "LffwFp16Fc1", "1", "listenflow.fastest_whisper"
    )
    if creator is None:
        raise RuntimeError("Required fp16 fc1 plugin creator is unavailable")
    plugin = creator.create_plugin(
        "fc1",
        trt.PluginFieldCollection(
            [
                trt.PluginField(
                    "bundle_sha256", np.frombuffer(digest, np.int8), trt.PluginFieldType.CHAR
                )
            ]
        ),
        trt.TensorRTPhase.BUILD,
    )
    if plugin is None:
        raise RuntimeError("Required fp16 fc1 plugin rejected its embedded bundle")
    return plugin
