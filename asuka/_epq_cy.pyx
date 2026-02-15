# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: cdivision=True

"""
Optional Cython implementation of the E_pq segment-walk hot loop.

Build locally (dev):
  python -m asuka.build.epq_ext build_ext --inplace

Notes:
  - Requires Cython (no vendored C++ fallback).
"""

from __future__ import annotations

import numpy as np
cimport numpy as cnp

from cython.parallel cimport prange, threadid
from libcpp.vector cimport vector
from libc.math cimport fabs, sqrt
from libc.stddef cimport size_t
from libc.stdint cimport uint64_t

cnp.import_array()

cdef extern from "asuka/include/cpu/openmp_compat.h":
    int guga_have_openmp()
    int guga_openmp_max_threads()
    void guga_openmp_set_num_threads(int n)


def have_openmp() -> bool:
    return bool(guga_have_openmp())


def openmp_max_threads() -> int:
    return int(guga_openmp_max_threads())


def openmp_set_num_threads(n: int) -> None:
    n = int(n)
    if n < 1:
        raise ValueError("OpenMP thread count must be >= 1")
    guga_openmp_set_num_threads(n)


include "cuguga/kernels/cpu/epq/_epq_cy_core.pxi"
include "cuguga/kernels/cpu/epq/_epq_cy_sparse_ops.pxi"
