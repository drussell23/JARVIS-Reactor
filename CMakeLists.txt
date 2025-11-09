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

# Find Python and pybind11
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

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
