from __future__ import annotations

import importlib
import os
import platform
import shutil
import sys


def _find_nvcc() -> str | None:
    for key in ("GUGA_CUDA_NVCC", "CUDACXX", "NVCC"):
        val = os.getenv(key)
        if val and os.path.exists(val):
            return os.path.abspath(val)
    return shutil.which("nvcc")


def _try_import(modname: str):
    try:
        return importlib.import_module(modname), None
    except Exception as e:  # pragma: no cover
        return None, e


def main() -> None:
    print("ASUKA environment check (cuGUGA core)")
    print(f"- python: {sys.version.split()[0]}")
    print(f"- platform: {platform.platform()}")

    _, epq_err = _try_import("asuka._epq_cy")
    if epq_err is None:
        print("- epq (_epq_cy): OK")
    else:
        print(f"- epq (_epq_cy): MISSING ({type(epq_err).__name__}: {epq_err})")
        print("  hint: rebuild with `python -m pip install -e .` (requires a C++ compiler)")

    cp, cp_err = _try_import("cupy")
    if cp_err is None:
        try:
            ndev = int(cp.cuda.runtime.getDeviceCount())
        except Exception:
            ndev = 0
        if ndev > 0:
            dev = cp.cuda.Device()
            props = cp.cuda.runtime.getDeviceProperties(int(dev.id))
            name = props.get("name", b"").decode(errors="ignore") if isinstance(props.get("name", b""), (bytes, bytearray)) else str(props.get("name"))
            cc_major = props.get("major", None)
            cc_minor = props.get("minor", None)
            cc = f"{cc_major}.{cc_minor}" if cc_major is not None and cc_minor is not None else "unknown"
            print(f"- cupy: OK (devices={ndev}, active={dev.id}, name={name}, cc={cc})")
        else:
            print("- cupy: OK (no CUDA devices detected)")
    else:
        print(f"- cupy: MISSING ({type(cp_err).__name__}: {cp_err})")
        print("  hint: install with `python -m pip install -e '.[cuda]'` (or '.[cuda12]')")

    nvcc = _find_nvcc()
    if nvcc is None:
        print("- nvcc: NOT FOUND")
        print("  hint: install a CUDA toolkit (nvcc) or set GUGA_CUDA_NVCC=/path/to/nvcc")
    else:
        print(f"- nvcc: {nvcc}")

    _, ext_err = _try_import("asuka._guga_cuda_ext")
    if ext_err is None:
        print("- cuda ext (_guga_cuda_ext): OK")
    else:
        print(f"- cuda ext (_guga_cuda_ext): MISSING ({type(ext_err).__name__}: {ext_err})")
        print("  hint: build with `python -m asuka.build.guga_cuda_ext` once nvcc is available")

    _, linalg_err = _try_import("asuka._guga_cuda_linalg_ext")
    if linalg_err is None:
        print("- cuda linalg ext (_guga_cuda_linalg_ext): OK")
    else:
        print(f"- cuda linalg ext (_guga_cuda_linalg_ext): MISSING ({type(linalg_err).__name__}: {linalg_err})")
        print("  hint: build with `python -m asuka.build.guga_cuda_linalg_ext` once nvcc is available")


if __name__ == "__main__":  # pragma: no cover
    main()
