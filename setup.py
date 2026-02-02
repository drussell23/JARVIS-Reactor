"""
Setup script for JARVIS Reactor with MLForge C++ bindings

v190.0: Enhanced with intelligent pybind11 discovery that works with
pip's isolated build environment by finding pybind11 via Python module.
"""
import os
import sys
import subprocess
from pathlib import Path
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext


def get_pybind11_cmake_dir() -> str:
    """
    v190.0: Discover pybind11 cmake directory using the Python module.

    This works reliably even in pip's isolated build environment because
    we use the current Python interpreter (sys.executable) which has
    pybind11 installed via build dependencies in pyproject.toml.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pybind11", "--cmakedir"],
            capture_output=True,
            text=True,
            check=True,
        )
        cmake_dir = result.stdout.strip()
        if cmake_dir and os.path.exists(cmake_dir):
            return cmake_dir
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # Fallback: try to find pybind11 in site-packages
    try:
        import pybind11
        cmake_dir = os.path.join(os.path.dirname(pybind11.__file__), "share", "cmake", "pybind11")
        if os.path.exists(cmake_dir):
            return cmake_dir
    except ImportError:
        pass

    return ""


class CMakeExtension(Extension):
    """CMake extension for building C++ bindings"""

    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    """
    v190.0: Enhanced CMake build command with intelligent pybind11 discovery.

    Properly locates pybind11 cmake files even in pip's isolated build
    environment, fixing the 'Could not find pybind11Config.cmake' error.
    """

    def build_extension(self, ext):
        if isinstance(ext, CMakeExtension):
            extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

            # Create build directory
            build_temp = Path(self.build_temp)
            build_temp.mkdir(parents=True, exist_ok=True)

            # v190.0: Get pybind11 cmake directory
            pybind11_dir = get_pybind11_cmake_dir()

            cmake_args = [
                f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
                f"-DPYTHON_EXECUTABLE={sys.executable}",
                "-DCMAKE_BUILD_TYPE=Release",
            ]

            # v190.0: Add pybind11 cmake directory if found
            if pybind11_dir:
                cmake_args.append(f"-Dpybind11_DIR={pybind11_dir}")
                print(f"[v190.0] Using pybind11 from: {pybind11_dir}")
            else:
                print("[v190.0] WARNING: pybind11 cmake directory not found, cmake may fail")

            build_args = ["--config", "Release"]

            # Run CMake
            print(f"[v190.0] Running cmake with args: {cmake_args}")
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
    name="jarvis-reactor",
    version="2.2.0",
    packages=find_packages(),
    ext_modules=[CMakeExtension("reactor_core.reactor_core_native")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
)
