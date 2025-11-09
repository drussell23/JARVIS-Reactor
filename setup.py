"""
Setup script for Reactor Core with MLForge C++ bindings
"""
import os
import sys
import subprocess
from pathlib import Path
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    """CMake extension for building C++ bindings"""

    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    """Custom build command for CMake extensions"""

    def build_extension(self, ext):
        if isinstance(ext, CMakeExtension):
            extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

            # Create build directory
            build_temp = Path(self.build_temp)
            build_temp.mkdir(parents=True, exist_ok=True)

            cmake_args = [
                f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
                f"-DPYTHON_EXECUTABLE={sys.executable}",
                "-DCMAKE_BUILD_TYPE=Release",
            ]

            build_args = ["--config", "Release"]

            # Run CMake
            subprocess.check_call(
                ["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp
            )

            # Run build
            subprocess.check_call(
                ["cmake", "--build", "."] + build_args, cwd=self.build_temp
            )
        else:
            super().build_extension(ext)


setup(
    name="reactor-core",
    version="1.0.0",
    packages=find_packages(),
    ext_modules=[CMakeExtension("reactor_core.reactor_core_native")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
)
