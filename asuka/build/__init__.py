"""Build helpers for optional native extensions.

These helpers intentionally live inside the Python package so they can be run as:

  python -m asuka.build.epq_ext build_ext --inplace
  python -m asuka.build.guga_cuda_ext
  python -m asuka.build.guga_cuda_linalg_ext
"""

