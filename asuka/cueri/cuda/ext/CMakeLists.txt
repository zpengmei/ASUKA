cmake_minimum_required(VERSION 3.18)

set(CMAKE_CUDA_RUNTIME_LIBRARY Static)

project(cueri_cuda_ext LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(CUERI_FAST_BOYS "Enable fast Boys moments for small-root Rys (NROOTS<=3)" OFF)
option(CUERI_FAST_DEV_DIRECT_JK "Build a reduced direct-J/K-focused extension without derivative/DF-heavy CUDA sources" OFF)
option(CUERI_FAST_DEV_STEP2_ONLY "Build a minimal step2-only extension for specialized ERI/J/K kernel work" OFF)

if (CUERI_FAST_DEV_DIRECT_JK AND CUERI_FAST_DEV_STEP2_ONLY)
  message(FATAL_ERROR "CUERI_FAST_DEV_DIRECT_JK and CUERI_FAST_DEV_STEP2_ONLY are mutually exclusive")
endif()

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)

set(CUERI_PYBIND_SOURCES
  src/cueri_cuda_ext.cpp
  src/cueri_cuda_ext_part1.cpp
  src/cueri_cuda_ext_part2.cpp
  src/cueri_cuda_ext_part3.cpp
  src/cueri_cuda_ext_part9.cpp
)

set(CUERI_KERNEL_SOURCES
  src/cueri_cuda_kernels.cu
  src/cueri_cuda_kernels_step2_part1.cu
  src/cueri_cuda_kernels_step2_part2.cu
  src/cueri_cuda_kernels_step2_part3.cu
  src/cueri_cuda_kernels_step2_part4.cu
  src/cueri_cuda_kernels_dense_scatter.cu
)

if (NOT CUERI_FAST_DEV_STEP2_ONLY)
  list(APPEND CUERI_PYBIND_SOURCES
    src/cueri_cuda_ext_part4.cpp
  )
  list(APPEND CUERI_KERNEL_SOURCES
    src/generated/cueri_cuda_kernels_wave1_generated_part1.cu
    src/generated/cueri_cuda_kernels_wave1_generated_part2.cu
    src/generated/cueri_cuda_kernels_wave1_generated_part3.cu
    src/generated/cueri_cuda_kernels_wave1_generated_part4.cu
    src/generated/cueri_cuda_kernels_wave1_generated_part5.cu
    src/generated/cueri_cuda_kernels_wave1_generated_part6.cu
    src/generated/cueri_cuda_kernels_wave1_generated_part7.cu
    src/generated/cueri_cuda_kernels_wave1_generated_part8.cu
    src/generated/cueri_cuda_kernels_wave1_generated_part9.cu
    src/generated/cueri_cuda_kernels_wave1_generated_part10.cu
    src/generated/cueri_cuda_kernels_wave1_generated_part11.cu
    src/generated/cueri_cuda_kernels_wave1_generated_part12.cu
    src/generated/cueri_cuda_kernels_wave2_generated_part1.cu
    src/generated/cueri_cuda_kernels_wave2_generated_part2.cu
    src/generated/cueri_cuda_kernels_wave2_generated_part3.cu
    src/generated/cueri_cuda_kernels_wave2_generated_part4.cu
    src/generated/cueri_cuda_kernels_wave2_generated_part5.cu
    src/generated/cueri_cuda_kernels_wave2_generated_part6.cu
    src/generated/cueri_cuda_kernels_wave2_generated_part7.cu
    src/generated/cueri_cuda_kernels_wave2_generated_part8.cu
    src/generated/cueri_cuda_kernels_wave2_generated_part9.cu
    src/cueri_cuda_kernels_rys_generic_part1.cu
  )
endif()

if (NOT CUERI_FAST_DEV_DIRECT_JK AND NOT CUERI_FAST_DEV_STEP2_ONLY)
  list(APPEND CUERI_PYBIND_SOURCES
    src/cueri_cuda_ext_part5.cpp
    src/cueri_cuda_ext_part6.cpp
    src/cueri_cuda_ext_part7.cpp
    src/cueri_cuda_ext_part8.cpp
  )
  list(APPEND CUERI_KERNEL_SOURCES
    src/cueri_cuda_kernels_rys_generic_deriv_part1.cu
    src/cueri_cuda_kernels_rys_generic_deriv_part2.cu
    src/cueri_cuda_kernels_df_deriv_part1.cu
    src/cueri_cuda_kernels_df_deriv_part2.cu
    src/cueri_cuda_kernels_df_deriv_part3.cu
    src/cueri_cuda_kernels_df_deriv_part4.cu
    src/cueri_cuda_kernels_df_deriv_part5.cu
    src/cueri_cuda_kernels_df_deriv_part6.cu
    src/cueri_cuda_kernels_df_deriv_part7.cu
    src/cueri_cuda_kernels_df_deriv_part8.cu
    src/cueri_cuda_kernels_df_deriv_part9.cu
    src/cueri_cuda_kernels_df_deriv_part10.cu
    src/cueri_cuda_kernels_df_deriv_part11.cu
    src/cueri_cuda_kernels_df_deriv_part12.cu
    src/cueri_cuda_kernels_df_deriv_part13.cu
    src/cueri_cuda_kernels_df_deriv_part14.cu
    src/cueri_cuda_kernels_df_deriv_part15.cu
    src/cueri_cuda_kernels_df_deriv_part16.cu
    src/cueri_cuda_kernels_df_deriv_part17.cu
    src/cueri_cuda_rys.cu
    src/cueri_cuda_kernels_cart2sph_eri.cu
    src/cueri_cuda_kernels_sph.cu
    src/cueri_cuda_kernels_int1e_sph.cu
  )
endif()

pybind11_add_module(
  _cueri_cuda_ext
  MODULE
  NO_EXTRAS
  ${CUERI_PYBIND_SOURCES}
  ${CUERI_KERNEL_SOURCES}
)

target_link_libraries(
  _cueri_cuda_ext
  PRIVATE
    CUDA::cudart_static
)

set_property(TARGET _cueri_cuda_ext PROPERTY CUDA_RUNTIME_LIBRARY Static)
set_target_properties(_cueri_cuda_ext PROPERTIES CUDA_VISIBILITY_PRESET "default")
set_target_properties(_cueri_cuda_ext PROPERTIES CXX_VISIBILITY_PRESET "default")
set_target_properties(_cueri_cuda_ext PROPERTIES VISIBILITY_INLINES_HIDDEN OFF)
set_target_properties(_cueri_cuda_ext PROPERTIES LINKER_LANGUAGE CUDA)
# Separable compilation is NOT needed: all __device__ functions are either
# defined in anonymous namespace within the same TU or included via .cuh
# headers.  Keeping it OFF avoids the serial nvcc --device-link step that
# serializes the final phase of the build and drops CPU utilization to ~3%.
# CUDA_SEPARABLE_COMPILATION OFF  (this is the default; line kept for docs)

target_compile_options(
  _cueri_cuda_ext
  PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)

if (CUERI_FAST_BOYS)
  target_compile_definitions(_cueri_cuda_ext PRIVATE CUERI_FAST_BOYS=1)
endif()
if (CUERI_FAST_DEV_DIRECT_JK)
  target_compile_definitions(_cueri_cuda_ext PRIVATE CUERI_FAST_DEV_DIRECT_JK=1)
endif()
if (CUERI_FAST_DEV_STEP2_ONLY)
  target_compile_definitions(_cueri_cuda_ext PRIVATE CUERI_FAST_DEV_STEP2_ONLY=1)
endif()

target_include_directories(
  _cueri_cuda_ext
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

if (DEFINED CUERI_CUDA_EXT_OUTPUT_DIR)
  set_target_properties(
    _cueri_cuda_ext
    PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY "${CUERI_CUDA_EXT_OUTPUT_DIR}"
  )
endif()
