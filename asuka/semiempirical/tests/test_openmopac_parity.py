from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from asuka.semiempirical import SemiempiricalCalculator
from asuka.semiempirical.gpu import has_cupy, has_cuda_device
from asuka.semiempirical.params import load_params

HARTREE_TO_KCAL = 627.509474
MOPAC_EXE = shutil.which("mopac")
CUDA_OK = has_cupy() and has_cuda_device()

_Z_BY_SYM = {"H": 1, "C": 6, "N": 7, "O": 8}
_ELEMS = ("H", "C", "N", "O")

_CASES = [
    (
        "h2",
        ["H", "H"],
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]], dtype=float),
    ),
    (
        "hcn",
        ["H", "C", "N"],
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.06],
                [0.0, 0.0, 2.22],
            ],
            dtype=float,
        ),
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
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.757, 0.586],
                [0.0, -0.757, 0.586],
            ],
            dtype=float,
        ),
    ),
    (
        "co",
        ["C", "O"],
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.128]], dtype=float),
    ),
    (
        "n2",
        ["N", "N"],
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.10]], dtype=float),
    ),
]


def _run_mopac_hof_kcal(case_name: str, symbols: list[str], coords_ang: np.ndarray) -> float:
    if not MOPAC_EXE:
        pytest.skip("mopac executable not available in PATH")

    with tempfile.TemporaryDirectory(prefix=f"mopac_{case_name}_") as td:
        td_path = Path(td)
        inp = td_path / f"{case_name}.mop"
        lines = ["AM1 1SCF XYZ PRECISE AUX", f"{case_name} ASUKA parity", ""]
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

        aux = (td_path / f"{case_name}.aux").read_text(encoding="utf-8", errors="replace")
        match = re.search(r"HEAT_OF_FORMATION:KCAL/MOL=([+-]?\d+\.\d+D[+-]\d+)", aux)
        if not match:
            raise RuntimeError(f"Failed to parse HEAT_OF_FORMATION from MOPAC AUX for {case_name}")
        return float(match.group(1).replace("D", "E"))


@pytest.mark.cuda
@pytest.mark.skipif(not CUDA_OK, reason="CuPy/CUDA unavailable")
def test_am1_openmopac_hof_parity_cpu_and_cuda():
    if not MOPAC_EXE:
        pytest.skip("mopac executable not available in PATH")

    params = load_params("AM1")
    cpu_calc = SemiempiricalCalculator(method="AM1", charge=0, device="cpu")
    cuda_calc = SemiempiricalCalculator(method="AM1", charge=0, device="cuda")

    # Compare molecular energetics to OpenMOPAC heat of formation.
    # We fit per-element additive reference constants once (to absorb any
    # atom-reference convention differences) and then assert molecule residuals.
    rows = []
    for case_name, symbols, coords_ang in _CASES:
        mopac_hof = _run_mopac_hof_kcal(case_name, symbols, coords_ang)
        cpu = cpu_calc.energy(
            symbols,
            coords_ang,
            coords_unit="angstrom",
            max_iter=120,
            conv_tol=1e-9,
            fock_mode="ri",
        )
        cuda = cuda_calc.energy(
            symbols,
            coords_ang,
            coords_unit="angstrom",
            max_iter=120,
            conv_tol=1e-9,
            fock_mode="w",
        )

        sum_eisol = sum(params.elements[_Z_BY_SYM[sym]].eisol for sym in symbols)
        lhs_cpu = float((cpu.energy_total - sum_eisol) * HARTREE_TO_KCAL)
        lhs_cuda = float((cuda.energy_total - sum_eisol) * HARTREE_TO_KCAL)
        counts = np.asarray([symbols.count(e) for e in _ELEMS], dtype=float)
        rows.append((case_name, lhs_cpu, lhs_cuda, float(mopac_hof), counts))

    A = np.vstack([r[4] for r in rows])
    b = np.asarray([r[3] - r[1] for r in rows], dtype=float)
    ref_consts, *_ = np.linalg.lstsq(A, b, rcond=None)

    for case_name, lhs_cpu, lhs_cuda, hof_mopac, counts in rows:
        hof_cpu = lhs_cpu + float(np.dot(counts, ref_consts))
        hof_cuda = lhs_cuda + float(np.dot(counts, ref_consts))
        assert abs(hof_cpu - hof_mopac) <= 1e-4, case_name
        assert abs(hof_cuda - hof_mopac) <= 1e-4, case_name
        assert abs(hof_cuda - hof_cpu) <= 1e-5, case_name
