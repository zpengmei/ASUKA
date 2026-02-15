"""Single-reference configuration interaction (spin-adapted, CSF/DRT).

This package contains small wrappers around :mod:`asuka.cuguga` that implement
single-reference CI truncations in a *spin-adapted* CSF basis.

Currently implemented
---------------------
* CISD (configuration interaction with singles and doubles)
"""

from asuka.ci.cisd import CISDResult, CISDResultMulti, GUGACISDSolver, build_drt_cisd, cisd, cisd_kernel

__all__ = [
    "CISDResult",
    "CISDResultMulti",
    "GUGACISDSolver",
    "build_drt_cisd",
    "cisd",
    "cisd_kernel",
]
