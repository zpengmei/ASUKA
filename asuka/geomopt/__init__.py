"""Geometry optimization utilities.

This subpackage provides a lightweight geometry optimization driver that
operates on an *energy+gradient callback*.

It is intended to be used with ASUKA gradient drivers (CASSCF, CASPT2, MRCI,
SOC-SI, ...) and any other code that can supply an energy+gradient callback.
"""

from __future__ import annotations

from .constraints import AngleConstraint, DistanceConstraint, InternalCoordinateConstraint
from .optimizer import GeomOptResult, GeomOptSettings, optimize_cartesian
from .scan import (
    Scan2DPointResult,
    Scan2DResult,
    Scan2DSettings,
    ScanPointResult,
    ScanResult,
    ScanSettings,
    angle_scan,
    distance_distance_scan2d,
    distance_scan,
    load_scan2d_result,
    load_scan_result,
    save_scan2d_result,
    save_scan_result,
    scan2d,
)

__all__ = [
    "AngleConstraint",
    "DistanceConstraint",
    "InternalCoordinateConstraint",
    "GeomOptResult",
    "GeomOptSettings",
    "optimize_cartesian",
    "ScanSettings",
    "ScanPointResult",
    "ScanResult",
    "distance_scan",
    "angle_scan",
    "save_scan_result",
    "load_scan_result",
    "Scan2DSettings",
    "Scan2DPointResult",
    "Scan2DResult",
    "scan2d",
    "distance_distance_scan2d",
    "save_scan2d_result",
    "load_scan2d_result",
]
