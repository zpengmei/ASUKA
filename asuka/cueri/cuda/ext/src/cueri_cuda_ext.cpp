#include "cueri_cuda_ext_common.h"

void cueri_bind_part1(py::module_& m);
void cueri_bind_part2(py::module_& m);
void cueri_bind_part3(py::module_& m);
void cueri_bind_part4(py::module_& m);
void cueri_bind_part5(py::module_& m);
void cueri_bind_part6(py::module_& m);
void cueri_bind_part7(py::module_& m);
void cueri_bind_part8(py::module_& m);
void cueri_bind_part9(py::module_& m);

PYBIND11_MODULE(_cueri_cuda_ext, m) {
  m.doc() = "cuERI CUDA extension (Step 1 kernels + Step 2 utilities)";
  constexpr int kCudaLmax = 5;
  constexpr int kCudaNrootsMax = 11;
  m.attr("CUDA_MAX_L") = py::int_(kCudaLmax);
  m.attr("CUDA_MAX_NROOTS") = py::int_(kCudaNrootsMax);
  m.def("kernel_limits_device", [=]() {
    py::dict out;
    out["lmax"] = py::int_(kCudaLmax);
    out["nroots_max"] = py::int_(kCudaNrootsMax);
    return out;
  });

  cueri_bind_part1(m);
  cueri_bind_part2(m);
  cueri_bind_part3(m);
  cueri_bind_part4(m);
  cueri_bind_part5(m);
  cueri_bind_part6(m);
  cueri_bind_part7(m);
  cueri_bind_part8(m);
  cueri_bind_part9(m);
}
