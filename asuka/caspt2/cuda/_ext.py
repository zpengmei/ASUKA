from __future__ import annotations


def _load_ext():
    try:
        from asuka import _caspt2_cuda_ext  # type: ignore[import-not-found]
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "CASPT2 CUDA extension was not built/installed. "
            "Reinstall ASUKA with a CUDA toolkit available (nvcc), "
            "or set ASUKA_SKIP_CUDA_EXT=1 for CPU-only."
        ) from e
    return _caspt2_cuda_ext


ext = _load_ext()

