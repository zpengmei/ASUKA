from __future__ import annotations

"""Self-contained linear algebra helpers for ASUKA.

This package provides triangular-solve and Cholesky utilities that replace
``scipy.linalg`` and ``cupyx.scipy.linalg`` in ASUKA's DF (density fitting)
and adjoint paths, making those paths dependency-free.
"""
