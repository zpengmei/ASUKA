from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from asuka.semiempirical import am1_gradient
from asuka.semiempirical.gpu import has_cupy, has_cuda_device
from asuka.semiempirical.params import ANGSTROM_TO_BOHR

MOPAC_EXE = shutil.which("mopac")
HARTREE_TO_KCAL = 627.509474
CUDA_OK = has_cupy() and has_cuda_device()

_CASES = [
    ("h2", ["H", "H"], np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]], dtype=float)),
    (
        "hcn",
        ["H", "C", "N"],
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.06], [0.0, 0.0, 2.22]], dtype=float),
    ),
    (
        "ch4",
        ["C", "H", "H", "H", "H"],
        np.array(
            [
                [0.0000, 0.0000, 0.0000],
                [0.6291, 0.6291, 0.6291],
                [0.6291, -0.6291, -0.6291],
                [-0.6291, 0.6291, -0.6291],
                [-0.6291, -0.6291, 0.6291],
            ],
            dtype=float,
        ),
    ),
    (
        "h2o",
        ["O", "H", "H"],
        np.array([[0.0, 0.0, 0.0], [0.0, 0.757, 0.586], [0.0, -0.757, 0.586]], dtype=float),
    ),
    ("co", ["C", "O"], np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.128]], dtype=float)),
    ("n2", ["N", "N"], np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.10]], dtype=float)),
]


def _parse_vector_hook(text: str, key: str) -> np.ndarray:
    pat = re.compile(rf"{re.escape(key)}\[(\d+)\]\s*=", re.IGNORECASE)
    m = pat.search(text)
    if not m:
        raise RuntimeError(f"Unable to find hook key '{key}' in MOPAC output")
    n = int(m.group(1))
    tail = text[m.end() :]
    nums = []
    for tok in re.findall(r"[+-]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[DEde][+-]?\d+)?", tail):
        nums.append(float(tok.replace("D", "E").replace("d", "E")))
        if len(nums) >= n:
            break
    if len(nums) < n:
        raise RuntimeError(f"Hook '{key}' expected {n} values but found {len(nums)}")
    return np.asarray(nums[:n], dtype=float)


def _run_mopac_gradient(case_name: str, symbols: list[str], coords_ang: np.ndarray) -> np.ndarray:
    if not MOPAC_EXE:
        pytest.skip("mopac executable not available in PATH")

    with tempfile.TemporaryDirectory(prefix=f"mopac_grad_{case_name}_") as td:
        td_path = Path(td)
        inp = td_path / f"{case_name}.mop"
        lines = ["AM1 1SCF XYZ GRADIENTS PRECISE AUX", f"{case_name} ASUKA gradient parity", ""]
        for sym, xyz in zip(symbols, coords_ang):
            x, y, z = [float(v) for v in xyz]
            lines.append(f"{sym} {x:.10f} 1 {y:.10f} 1 {z:.10f} 1")
        inp.write_text("\n".join(lines) + "\n", encoding="utf-8")

        proc = subprocess.run(
            [MOPAC_EXE, str(inp)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"MOPAC failed for {case_name} with rc={proc.returncode}\n"
                f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
            )

        out_path = td_path / f"{case_name}.out"
        out_text = out_path.read_text(encoding="utf-8", errors="replace")
        try:
            vec = _parse_vector_hook(out_text, "GRADIENTS:KCAL/MOL/ANGSTROM")
        except RuntimeError:
            # Fallback for OpenMOPAC builds that print only the
            # "FINAL POINT AND DERIVATIVES" table in .out.
            vals = []
            pat = re.compile(
                r"^\s*\d+\s+\d+\s+\w+\s+CARTESIAN\s+[XYZ]\s+"
                r"[+-]?\d+\.\d+\s+([+-]?\d+\.\d+)\s+KCAL/ANGSTROM\s*$"
            )
            for line in out_text.splitlines():
                m = pat.match(line)
                if m:
                    vals.append(float(m.group(1)))
            vec = np.asarray(vals, dtype=float)
        nat = len(symbols)
        if vec.size != 3 * nat:
            raise RuntimeError(
                f"Expected {3 * nat} gradient values for {case_name}, got {vec.size}"
            )
        return vec.reshape((nat, 3))


def _ha_bohr_to_kcal_ang(grad_ha_bohr: np.ndarray) -> np.ndarray:
    return np.asarray(grad_ha_bohr, dtype=float) * HARTREE_TO_KCAL * ANGSTROM_TO_BOHR


def test_am1_openmopac_gradient_parity():
    if not MOPAC_EXE:
        pytest.skip("mopac executable not available in PATH")
    if not CUDA_OK:
        pytest.skip("CUDA analytical gradient backend unavailable")

    for case_name, symbols, coords_ang in _CASES:
        grad_mopac = _run_mopac_gradient(case_name, symbols, coords_ang)
        grad_asuka = am1_gradient(
            symbols,
            coords_ang,
            coords_unit="angstrom",
            device="cuda",
            max_iter=120,
            conv_tol=1e-9,
            fock_mode="ri",
            gradient_backend="cuda_analytic",
        )
        grad_asuka_kcal_ang = _ha_bohr_to_kcal_ang(grad_asuka)

        diff = grad_asuka_kcal_ang - grad_mopac
        rms = float(np.sqrt(np.mean(diff * diff)))
        mx = float(np.max(np.abs(diff)))
        assert rms <= 0.03, f"{case_name}: rms={rms:.6f}"
        assert mx <= 0.10, f"{case_name}: max={mx:.6f}"
