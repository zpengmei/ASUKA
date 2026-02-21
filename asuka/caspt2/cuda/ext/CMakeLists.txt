cmake_minimum_required(VERSION 3.18)

project(caspt2_cuda_ext LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Prefer static CUDA runtime for Python extension modules (see rationale in
# `asuka/cuda/ext/CMakeLists.txt`).
set(CMAKE_CUDA_RUNTIME_LIBRARY Static)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)

pybind11_add_module(
  _caspt2_cuda_ext
  MODULE
    src/caspt2_cuda_ext.cpp
    src/caspt2_cuda_kernels.cu
)

target_compile_options(
  _caspt2_cuda_ext
  PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo>
)

target_link_libraries(
  _caspt2_cuda_ext
  PRIVATE
    CUDA::cudart_static
)

# Allow the build script to direct the output into the in-tree python package.
if (DEFINED GUGA_CUDA_EXT_OUTPUT_DIR)
  set_target_properties(
    _caspt2_cuda_ext
    PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY "${GUGA_CUDA_EXT_OUTPUT_DIR}"
  )
endif()

