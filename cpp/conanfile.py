"""Native SDK inputs shared by standalone and Python builds."""

from pathlib import Path

from conan import ConanFile
from conan.tools.cmake import CMakeDeps, CMakeToolchain


class WhisperDependencies(ConanFile):
    settings = "os", "arch", "compiler", "build_type"

    def requirements(self):
        self.requires("tensoracle/[>=2.6.0 <3]")
        self.requires("tensorrt/[>=11.2.1.2 <12]")

    def build_requirements(self):
        self.tool_requires("cuda-toolkit/13.2.2", options={"nvrtc": True, "cuobjdump": True})

    def generate(self):
        CMakeDeps(self).generate()
        cuda = self.dependencies.build["cuda-toolkit"]
        root = cuda.conf_info.get("user.cuda:root")
        toolchain = CMakeToolchain(self, generator="Ninja")
        toolchain.user_presets_path = False
        toolchain.blocks.remove("output_dirs")
        # SDK helper functions need a root as well as the generated TensorRT target.
        toolchain.variables["LFFW_TENSORRT_ROOT"] = str(
            self.dependencies["tensorrt"].package_folder
        )
        toolchain.variables["LFFW_TENSORACLE_LIBRARY_DIR"] = self.dependencies[
            "tensoracle"
        ].cpp_info.libdirs[0]
        # Conan owns these inputs, including when an older caller passes a PyPI header root.
        self.conf.define(
            "tools.cmake.cmaketoolchain:extra_variables",
            {
                name: {"value": value, "cache": True, "type": "FILEPATH", "force": True}
                for name, value in {
                    "CUDAToolkit_ROOT": root,
                    "CMAKE_CUDA_COMPILER": str(Path(root, "bin", "nvcc")),
                }.items()
            },
        )
        toolchain.generate()
