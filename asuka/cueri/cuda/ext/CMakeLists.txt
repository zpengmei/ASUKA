cmake_minimum_required(VERSION 3.18)

set(CMAKE_CUDA_RUNTIME_LIBRARY Static)

project(cueri_cuda_ext LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(CUERI_FAST_BOYS "Enable fast Boys moments for small-root Rys (NROOTS<=3)" OFF)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)

pybind11_add_module(
  _cueri_cuda_ext
  MODULE
  NO_EXTRAS
    src/cueri_cuda_ext.cpp
    src/cueri_cuda_kernels.cu
    src/cueri_cuda_rys.cu
    src/cueri_cuda_kernels_step2.cu
    src/generated/cueri_cuda_kernels_wave1_generated.cu
    src/generated/cueri_cuda_kernels_wave2_generated.cu
    src/cueri_cuda_kernels_rys_generic.cu
    src/cueri_cuda_kernels_rys_generic_deriv.cu
    src/cueri_cuda_kernels_dense_scatter.cu
    src/cueri_cuda_kernels_df_deriv.cu
    src/cueri_cuda_kernels_cart2sph_eri.cu
    src/cueri_cuda_kernels_sph.cu
    src/cueri_cuda_kernels_int1e_sph.cu
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
set_target_properties(_cueri_cuda_ext PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(_cueri_cuda_ext PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_compile_options(
  _cueri_cuda_ext
  PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)

if (CUERI_FAST_BOYS)
  target_compile_definitions(_cueri_cuda_ext PRIVATE CUERI_FAST_BOYS=1)
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
