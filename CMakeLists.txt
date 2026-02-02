cmake_minimum_required(VERSION 3.15)
project(ReactorCore VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# macOS SDK for Apple platforms
if(APPLE)
    set(CMAKE_OSX_SYSROOT /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk)
endif()

# v190.0: Honor PYTHON_EXECUTABLE hint from pip/setup.py
# This ensures we use the correct venv Python, not system Python
if(DEFINED PYTHON_EXECUTABLE AND EXISTS "${PYTHON_EXECUTABLE}")
    message(STATUS "[v190.0] Using specified Python: ${PYTHON_EXECUTABLE}")
    set(Python3_EXECUTABLE "${PYTHON_EXECUTABLE}" CACHE FILEPATH "Python executable" FORCE)
endif()

# Find Python using the specified executable as hint
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
message(STATUS "[v190.0] Python3 found: ${Python3_EXECUTABLE}")

# v190.0: Find pybind11 using Python's pybind11 module for reliable discovery
# This works with venv-installed pybind11 regardless of cmake prefix path
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE pybind11_RESULT
    ERROR_QUIET
)

if(pybind11_RESULT EQUAL 0 AND EXISTS "${pybind11_CMAKE_DIR}")
    message(STATUS "[v190.0] Found pybind11 via Python module: ${pybind11_CMAKE_DIR}")
    set(pybind11_DIR "${pybind11_CMAKE_DIR}" CACHE PATH "pybind11 cmake directory" FORCE)
else()
    message(STATUS "[v190.0] pybind11 not found via Python module (result=${pybind11_RESULT})")
    # Check if pybind11_DIR was passed from setup.py
    if(DEFINED pybind11_DIR AND EXISTS "${pybind11_DIR}")
        message(STATUS "[v190.0] Using pybind11_DIR from command line: ${pybind11_DIR}")
    else()
        message(STATUS "[v190.0] Trying system paths...")
    endif()
endif()

find_package(pybind11 CONFIG REQUIRED)
message(STATUS "[v190.0] pybind11 found: ${pybind11_VERSION}")

# Add MLForge as subdirectory (C++ core)
add_subdirectory(mlforge)

# Create Python bindings module
pybind11_add_module(reactor_core_native
    bindings/reactor_bindings.cpp
)

# Link against MLForge library
target_link_libraries(reactor_core_native PRIVATE MLForgeLib)

# Include directories
target_include_directories(reactor_core_native PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/mlforge/include
)

# Installation
install(TARGETS reactor_core_native DESTINATION reactor_core)
