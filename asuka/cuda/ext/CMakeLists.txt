cmake_minimum_required(VERSION 3.18)

project(guga_cuda_ext LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Prefer static CUDA runtime for Python extension modules.
#
# On some systems, dynamically loading `libcudart.so` into an already-running
# process (as Python does for extension modules) can crash inside early runtime
# calls like `cudaGetDevice`. Linking the runtime statically avoids that failure
# mode and matches nvcc's default for standalone executables.
set(CMAKE_CUDA_RUNTIME_LIBRARY Static)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)

pybind11_add_module(
  _guga_cuda_ext
  MODULE
    src/guga_cuda_ext.cpp
    ../../cuguga/kernels/gpu/cuda_ext/guga_cuda_kernels.cu
    ../../cuguga/kernels/gpu/cuda_ext/guga_cuda_fixed_sparse_kernels.cu
)

target_link_libraries(
  _guga_cuda_ext
  PRIVATE
    CUDA::cudart_static
    CUDA::cublas
    CUDA::cublasLt
)

target_compile_options(
  _guga_cuda_ext
  PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo>
)

target_include_directories(
  _guga_cuda_ext
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/gpu
)

# CUDA 13.x opt-in: allow ptxas to spill registers/stack into shared memory via PTX pragma.
# This can help register/stack-heavy kernels (e.g. segment-walk) but may reduce occupancy.
option(GUGA_CUDA_ENABLE_SMEM_SPILLING "Enable PTX enable_smem_spilling pragma (CUDA 13.x)" OFF)
if (GUGA_CUDA_ENABLE_SMEM_SPILLING)
  target_compile_definitions(_guga_cuda_ext PRIVATE GUGA_CUDA_ENABLE_SMEM_SPILLING=1)
endif()

# Experimental: Kernel1A warp32 shared-stack variant (often a performance tradeoff).
option(GUGA_CUDA_ENABLE_KERNEL1A_SHARED_STACK "Enable experimental Kernel1A shared-memory DFS stack (warp32)" OFF)
if (GUGA_CUDA_ENABLE_KERNEL1A_SHARED_STACK)
  target_compile_definitions(_guga_cuda_ext PRIVATE GUGA_CUDA_ENABLE_KERNEL1A_SHARED_STACK=1)
endif()

# Allow the build script to direct the output into the in-tree python package.
if (DEFINED GUGA_CUDA_EXT_OUTPUT_DIR)
  set_target_properties(
    _guga_cuda_ext
    PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY "${GUGA_CUDA_EXT_OUTPUT_DIR}"
  )
endif()
