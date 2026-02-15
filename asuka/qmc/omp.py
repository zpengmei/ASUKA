from __future__ import annotations


def have_openmp() -> bool:
    try:
        import asuka._epq_cy as m  # type: ignore
    except Exception:
        return False
    try:
        return bool(m.have_openmp())
    except Exception:
        return False


def openmp_max_threads() -> int:
    try:
        import asuka._epq_cy as m  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("asuka._epq_cy is not available (build the Cython extension)") from e
    return int(m.openmp_max_threads())


def openmp_set_num_threads(n: int) -> None:
    n = int(n)
    if n < 1:
        raise ValueError("OpenMP thread count must be >= 1")

    try:
        import asuka._epq_cy as m  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("asuka._epq_cy is not available (build the Cython extension)") from e

    if not bool(m.have_openmp()):  # pragma: no cover
        raise RuntimeError("asuka._epq_cy was built without OpenMP support")

    m.openmp_set_num_threads(int(n))


def maybe_set_openmp_threads(n: int | None) -> None:
    if n is None:
        return
    openmp_set_num_threads(int(n))

