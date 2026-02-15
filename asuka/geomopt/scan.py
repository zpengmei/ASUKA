from __future__ import annotations

"""1D relaxed internal-coordinate scans (distance / angle).

A *relaxed scan* sets an internal coordinate to a sequence of target values and
optimizes all remaining degrees of freedom at each point.

This module builds on:

- :class:`asuka.geomopt.constraints.DistanceConstraint`
- :class:`asuka.geomopt.constraints.AngleConstraint`
- :func:`asuka.geomopt.optimizer.optimize_cartesian` (projected L-BFGS)

Restart / initial guess behavior
-------------------------------
During a scan, each point uses the previous optimized geometry as the initial
guess. You can also **restart** a scan by reading an existing result file; the
last geometry in the file is used as the initial guess for the next point.

Units
-----
- Geometries: Bohr
- Distance targets: Bohr
- Angle targets: radians
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .constraints import AngleConstraint, DistanceConstraint, InternalCoordinateConstraint
from .optimizer import EnergyGradFn, GeomOptSettings, optimize_cartesian


@dataclass(frozen=True)
class ScanPointResult:
    """Result at a single scan target."""

    target: float
    achieved: float
    energy: float
    converged: bool
    message: str
    n_steps: int
    n_eval: int
    coords_final_bohr: np.ndarray  # (natm,3)


@dataclass(frozen=True)
class ScanResult:
    """Results for a whole scan."""

    constraint_type: str  # "distance" or "angle"
    indices: tuple[int, ...]
    unit: str  # "Bohr" or "radian"
    points: tuple[ScanPointResult, ...]

    def targets(self) -> np.ndarray:
        return np.asarray([p.target for p in self.points], dtype=np.float64)

    def energies(self) -> np.ndarray:
        return np.asarray([p.energy for p in self.points], dtype=np.float64)

    def achieved_values(self) -> np.ndarray:
        return np.asarray([p.achieved for p in self.points], dtype=np.float64)


@dataclass(frozen=True)
class ScanSettings:
    """Settings controlling the scan driver."""

    opt: GeomOptSettings = GeomOptSettings()
    save_path: str | None = None
    restart_path: str | None = None
    verbose: int = 0


@dataclass(frozen=True)
class Scan2DPointResult:
    """Result at a single 2D scan target (i,j)."""

    i: int
    j: int
    target1: float
    target2: float
    achieved1: float
    achieved2: float
    energy: float
    converged: bool
    message: str
    n_steps: int
    n_eval: int
    coords_final_bohr: np.ndarray  # (natm,3)


@dataclass(frozen=True)
class Scan2DResult:
    """Results for a 2D relaxed scan on a rectangular grid."""

    constraint_types: tuple[str, str]
    indices1: tuple[int, ...]
    indices2: tuple[int, ...]
    units: tuple[str, str]
    values1: np.ndarray  # (n1,)
    values2: np.ndarray  # (n2,)
    points: tuple[Scan2DPointResult, ...]  # row-major (i fastest outer, j inner)

    @property
    def n1(self) -> int:
        return int(np.asarray(self.values1).size)

    @property
    def n2(self) -> int:
        return int(np.asarray(self.values2).size)

    @property
    def shape(self) -> tuple[int, int]:
        return self.n1, self.n2

    def energies_grid(self) -> np.ndarray:
        return np.asarray([p.energy for p in self.points], dtype=np.float64).reshape(self.shape)


@dataclass(frozen=True)
class Scan2DSettings:
    """Settings controlling the 2D scan driver."""

    opt: GeomOptSettings = GeomOptSettings()
    save_path: str | None = None
    restart_path: str | None = None
    verbose: int = 0


def _point_to_dict(p: ScanPointResult) -> dict[str, Any]:
    return {
        "target": float(p.target),
        "achieved": float(p.achieved),
        "energy": float(p.energy),
        "converged": bool(p.converged),
        "message": str(p.message),
        "n_steps": int(p.n_steps),
        "n_eval": int(p.n_eval),
        "coords_bohr": np.asarray(p.coords_final_bohr, dtype=np.float64).tolist(),
    }


def _dict_to_point(d: dict[str, Any]) -> ScanPointResult:
    coords = np.asarray(d["coords_bohr"], dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("invalid coords_bohr in scan file")
    return ScanPointResult(
        target=float(d["target"]),
        achieved=float(d["achieved"]),
        energy=float(d["energy"]),
        converged=bool(d["converged"]),
        message=str(d.get("message", "")),
        n_steps=int(d.get("n_steps", 0)),
        n_eval=int(d.get("n_eval", 0)),
        coords_final_bohr=coords,
    )


def save_scan_result(result: ScanResult, path: str | Path) -> None:
    path = Path(path)
    payload = {
        "version": 1,
        "constraint_type": str(result.constraint_type),
        "indices": [int(i) for i in result.indices],
        "unit": str(result.unit),
        "points": [_point_to_dict(p) for p in result.points],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def load_scan_result(path: str | Path) -> ScanResult:
    path = Path(path)
    payload = json.loads(path.read_text())
    if int(payload.get("version", 0)) != 1:
        raise ValueError("unsupported scan file version")

    ctype = str(payload["constraint_type"])
    idx = tuple(int(i) for i in payload["indices"])
    unit = str(payload["unit"])
    pts = tuple(_dict_to_point(d) for d in payload.get("points", []))

    return ScanResult(constraint_type=ctype, indices=idx, unit=unit, points=pts)


def _as_values(values: Iterable[float] | np.ndarray) -> np.ndarray:
    v = np.asarray(list(values), dtype=np.float64).reshape((-1,))
    if v.size == 0:
        raise ValueError("scan values list is empty")
    if not np.all(np.isfinite(v)):
        raise ValueError("scan values must be finite")
    return v


def _normalize_constraint_type(constraint_type: str) -> str:
    ct = str(constraint_type).strip().lower()
    if ct in ("distance", "dist", "bond"):
        return "distance"
    if ct in ("angle", "bend"):
        return "angle"
    raise ValueError(f"unknown constraint_type={constraint_type!r}")


def _prefix_allclose(a: np.ndarray, b: np.ndarray, *, atol: float = 1e-12) -> bool:
    a = np.asarray(a, dtype=np.float64).reshape((-1,))
    b = np.asarray(b, dtype=np.float64).reshape((-1,))
    if int(a.size) > int(b.size):
        return False
    if int(a.size) == 0:
        return True
    return bool(np.allclose(a, b[: int(a.size)], atol=float(atol), rtol=0.0))


def _constraint_from_spec(constraint_type: str, indices: tuple[int, ...]) -> tuple[InternalCoordinateConstraint, str]:
    ct = _normalize_constraint_type(constraint_type)
    if ct == "distance":
        if len(indices) != 2:
            raise ValueError("distance constraint requires 2 indices")
        return DistanceConstraint(indices[0], indices[1]), "Bohr"
    if ct == "angle":
        if len(indices) != 3:
            raise ValueError("angle constraint requires 3 indices")
        return AngleConstraint(indices[0], indices[1], indices[2]), "radian"
    raise ValueError(f"unknown constraint_type={constraint_type!r}")


def _run_scan(
    energy_grad: EnergyGradFn,
    coords0_bohr: np.ndarray,
    *,
    constraint_type: str,
    indices: tuple[int, ...],
    values: Iterable[float] | np.ndarray,
    settings: ScanSettings | None = None,
) -> ScanResult:
    st = settings or ScanSettings()

    values_arr = _as_values(values)
    constraint, unit = _constraint_from_spec(constraint_type, indices)

    points: list[ScanPointResult] = []
    coords_init = np.asarray(coords0_bohr, dtype=np.float64)
    start_idx = 0

    restart_path = st.restart_path
    if restart_path is not None and Path(restart_path).exists():
        prev = load_scan_result(restart_path)
        if prev.constraint_type.strip().lower() != str(constraint_type).strip().lower():
            raise ValueError("restart file constraint_type mismatch")
        if tuple(prev.indices) != tuple(indices):
            raise ValueError("restart file indices mismatch")

        points = list(prev.points)
        start_idx = len(points)
        if points:
            coords_init = np.asarray(points[-1].coords_final_bohr, dtype=np.float64)

        if int(st.verbose) >= 1:
            print(f"[scan] restart loaded: {start_idx} points")

    if start_idx > int(values_arr.size):
        raise ValueError("restart file has more points than requested values")

    for i in range(start_idx, int(values_arr.size)):
        target = float(values_arr[i])
        if int(st.verbose) >= 1:
            print(f"[scan] point {i+1}/{values_arr.size} target={target:.10g} {unit}")

        res = optimize_cartesian(
            energy_grad,
            coords_init,
            settings=st.opt,
            constraint=constraint,
            constraint_target=target,
        )

        coords_fin = np.asarray(res.coords_final_bohr, dtype=np.float64)
        achieved = float(constraint.value(coords_fin))

        points.append(
            ScanPointResult(
                target=target,
                achieved=achieved,
                energy=float(res.energy_final),
                converged=bool(res.converged),
                message=str(res.message),
                n_steps=int(res.n_steps),
                n_eval=int(res.n_eval),
                coords_final_bohr=coords_fin,
            )
        )
        coords_init = coords_fin

        if st.save_path is not None:
            tmp = ScanResult(constraint_type=str(constraint_type), indices=tuple(indices), unit=unit, points=tuple(points))
            save_scan_result(tmp, st.save_path)

    return ScanResult(constraint_type=str(constraint_type), indices=tuple(indices), unit=unit, points=tuple(points))


def distance_scan(
    energy_grad: EnergyGradFn,
    coords0_bohr: np.ndarray,
    *,
    i: int,
    j: int,
    values_bohr: Iterable[float] | np.ndarray,
    settings: ScanSettings | None = None,
) -> ScanResult:
    """Run a relaxed distance scan between atoms i and j."""

    return _run_scan(
        energy_grad,
        coords0_bohr,
        constraint_type="distance",
        indices=(int(i), int(j)),
        values=values_bohr,
        settings=settings,
    )


def angle_scan(
    energy_grad: EnergyGradFn,
    coords0_bohr: np.ndarray,
    *,
    i: int,
    j: int,
    k: int,
    values_rad: Iterable[float] | np.ndarray,
    settings: ScanSettings | None = None,
) -> ScanResult:
    """Run a relaxed angle scan for atoms i-j-k (angle at j)."""

    return _run_scan(
        energy_grad,
        coords0_bohr,
        constraint_type="angle",
        indices=(int(i), int(j), int(k)),
        values=values_rad,
        settings=settings,
    )


def _point2d_to_dict(p: Scan2DPointResult) -> dict[str, Any]:
    return {
        "i": int(p.i),
        "j": int(p.j),
        "target1": float(p.target1),
        "target2": float(p.target2),
        "achieved1": float(p.achieved1),
        "achieved2": float(p.achieved2),
        "energy": float(p.energy),
        "converged": bool(p.converged),
        "message": str(p.message),
        "n_steps": int(p.n_steps),
        "n_eval": int(p.n_eval),
        "coords_bohr": np.asarray(p.coords_final_bohr, dtype=np.float64).tolist(),
    }


def _dict_to_point2d(d: dict[str, Any]) -> Scan2DPointResult:
    coords = np.asarray(d["coords_bohr"], dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("invalid coords_bohr in scan2d file")
    return Scan2DPointResult(
        i=int(d["i"]),
        j=int(d["j"]),
        target1=float(d["target1"]),
        target2=float(d["target2"]),
        achieved1=float(d["achieved1"]),
        achieved2=float(d["achieved2"]),
        energy=float(d["energy"]),
        converged=bool(d["converged"]),
        message=str(d.get("message", "")),
        n_steps=int(d.get("n_steps", 0)),
        n_eval=int(d.get("n_eval", 0)),
        coords_final_bohr=coords,
    )


def save_scan2d_result(result: Scan2DResult, path: str | Path) -> None:
    path = Path(path)
    payload = {
        "version": 1,
        "dim": 2,
        "constraint_types": [str(result.constraint_types[0]), str(result.constraint_types[1])],
        "indices1": [int(i) for i in result.indices1],
        "indices2": [int(i) for i in result.indices2],
        "units": [str(result.units[0]), str(result.units[1])],
        "values1": np.asarray(result.values1, dtype=np.float64).reshape((-1,)).tolist(),
        "values2": np.asarray(result.values2, dtype=np.float64).reshape((-1,)).tolist(),
        "points": [_point2d_to_dict(p) for p in result.points],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def load_scan2d_result(path: str | Path) -> Scan2DResult:
    path = Path(path)
    payload = json.loads(path.read_text())
    if int(payload.get("version", 0)) != 1 or int(payload.get("dim", 0)) != 2:
        raise ValueError("unsupported scan2d file version")

    ctype = tuple(str(x) for x in payload["constraint_types"])
    if len(ctype) != 2:
        raise ValueError("invalid constraint_types in scan2d file")
    idx1 = tuple(int(i) for i in payload["indices1"])
    idx2 = tuple(int(i) for i in payload["indices2"])
    units = tuple(str(x) for x in payload["units"])
    if len(units) != 2:
        raise ValueError("invalid units in scan2d file")
    v1 = np.asarray(payload["values1"], dtype=np.float64).reshape((-1,))
    v2 = np.asarray(payload["values2"], dtype=np.float64).reshape((-1,))
    pts = tuple(_dict_to_point2d(d) for d in payload.get("points", []))

    return Scan2DResult(
        constraint_types=(ctype[0], ctype[1]),
        indices1=idx1,
        indices2=idx2,
        units=(units[0], units[1]),
        values1=v1,
        values2=v2,
        points=pts,
    )


def scan2d(
    energy_grad: EnergyGradFn,
    coords0_bohr: np.ndarray,
    *,
    constraint1_type: str,
    indices1: tuple[int, ...],
    values1: Iterable[float] | np.ndarray,
    constraint2_type: str,
    indices2: tuple[int, ...],
    values2: Iterable[float] | np.ndarray,
    settings: Scan2DSettings | None = None,
) -> Scan2DResult:
    """Run a 2D relaxed scan with two simultaneous constraints."""

    st = settings or Scan2DSettings()

    v1 = _as_values(values1)
    v2 = _as_values(values2)

    ct1 = _normalize_constraint_type(constraint1_type)
    ct2 = _normalize_constraint_type(constraint2_type)
    c1, u1 = _constraint_from_spec(ct1, indices1)
    c2, u2 = _constraint_from_spec(ct2, indices2)

    points_grid: list[list[Scan2DPointResult | None]] = [[None for _ in range(int(v2.size))] for _ in range(int(v1.size))]
    coords_grid: list[list[np.ndarray | None]] = [[None for _ in range(int(v2.size))] for _ in range(int(v1.size))]

    restart_path = st.restart_path
    if restart_path is not None and Path(restart_path).exists():
        prev = load_scan2d_result(restart_path)
        ct_prev = tuple(_normalize_constraint_type(s) for s in prev.constraint_types)
        ct_now = (ct1, ct2)
        if ct_prev != ct_now:
            raise ValueError("restart file constraint_types mismatch")
        if tuple(prev.indices1) != tuple(indices1) or tuple(prev.indices2) != tuple(indices2):
            raise ValueError("restart file indices mismatch")

        if not _prefix_allclose(prev.values1, v1) or not _prefix_allclose(prev.values2, v2):
            raise ValueError("restart file values mismatch (expected prefix match)")

        for p in prev.points:
            i = int(p.i)
            j = int(p.j)
            if not (0 <= i < int(v1.size) and 0 <= j < int(v2.size)):
                continue
            # Only reuse if targets match the current grid.
            if abs(float(p.target1) - float(v1[i])) > 1e-10:
                continue
            if abs(float(p.target2) - float(v2[j])) > 1e-10:
                continue
            points_grid[i][j] = p
            coords_grid[i][j] = np.asarray(p.coords_final_bohr, dtype=np.float64)

        if int(st.verbose) >= 1:
            n_loaded = sum(1 for row in points_grid for p in row if p is not None)
            print(f"[scan2d] restart loaded: {n_loaded} points")

    coords0 = np.asarray(coords0_bohr, dtype=np.float64)

    for i in range(int(v1.size)):
        for j in range(int(v2.size)):
            if points_grid[i][j] is not None:
                continue

            target1 = float(v1[i])
            target2 = float(v2[j])
            if int(st.verbose) >= 1:
                print(f"[scan2d] point ({i+1},{j+1}) / ({v1.size},{v2.size}) targets=({target1:.10g} {u1}, {target2:.10g} {u2})")

            if j > 0 and coords_grid[i][j - 1] is not None:
                coords_init = np.asarray(coords_grid[i][j - 1], dtype=np.float64)
            elif i > 0 and coords_grid[i - 1][j] is not None:
                coords_init = np.asarray(coords_grid[i - 1][j], dtype=np.float64)
            else:
                coords_init = np.asarray(coords0, dtype=np.float64)

            res = optimize_cartesian(
                energy_grad,
                coords_init,
                settings=st.opt,
                constraints=(c1, c2),
                constraint_targets=(target1, target2),
            )

            coords_fin = np.asarray(res.coords_final_bohr, dtype=np.float64)
            achieved1 = float(c1.value(coords_fin))
            achieved2 = float(c2.value(coords_fin))

            pt = Scan2DPointResult(
                i=int(i),
                j=int(j),
                target1=target1,
                target2=target2,
                achieved1=achieved1,
                achieved2=achieved2,
                energy=float(res.energy_final),
                converged=bool(res.converged),
                message=str(res.message),
                n_steps=int(res.n_steps),
                n_eval=int(res.n_eval),
                coords_final_bohr=coords_fin,
            )
            points_grid[i][j] = pt
            coords_grid[i][j] = coords_fin

            if st.save_path is not None:
                pts_partial = [p for row in points_grid for p in row if p is not None]
                save_scan2d_result(
                    Scan2DResult(
                        constraint_types=(ct1, ct2),
                        indices1=tuple(indices1),
                        indices2=tuple(indices2),
                        units=(u1, u2),
                        values1=v1,
                        values2=v2,
                        points=tuple(pts_partial),
                    ),
                    st.save_path,
                )

    pts = tuple(points_grid[i][j] for i in range(int(v1.size)) for j in range(int(v2.size)))
    if any(p is None for p in pts):  # pragma: no cover
        raise RuntimeError("scan2d finished with missing points")

    return Scan2DResult(
        constraint_types=(ct1, ct2),
        indices1=tuple(indices1),
        indices2=tuple(indices2),
        units=(u1, u2),
        values1=v1,
        values2=v2,
        points=tuple(p for p in pts if p is not None),
    )


def distance_distance_scan2d(
    energy_grad: EnergyGradFn,
    coords0_bohr: np.ndarray,
    *,
    i1: int,
    j1: int,
    values1_bohr: Iterable[float] | np.ndarray,
    i2: int,
    j2: int,
    values2_bohr: Iterable[float] | np.ndarray,
    settings: Scan2DSettings | None = None,
) -> Scan2DResult:
    """Convenience wrapper for a 2D scan over two distances."""

    return scan2d(
        energy_grad,
        coords0_bohr,
        constraint1_type="distance",
        indices1=(int(i1), int(j1)),
        values1=values1_bohr,
        constraint2_type="distance",
        indices2=(int(i2), int(j2)),
        values2=values2_bohr,
        settings=settings,
    )


__all__ = [
    "ScanSettings",
    "ScanPointResult",
    "ScanResult",
    "Scan2DSettings",
    "Scan2DPointResult",
    "Scan2DResult",
    "distance_scan",
    "angle_scan",
    "scan2d",
    "distance_distance_scan2d",
    "save_scan_result",
    "load_scan_result",
    "save_scan2d_result",
    "load_scan2d_result",
]
