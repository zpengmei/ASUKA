r"""Internally contracted CASPT2 (SS/MS/XMS) on GPU (DF, C1, FP64).

This package implements the 13-case OpenMolcas formalism for IC-CASPT2
with support for both CPU (NumPy + full ERIs) and GPU (CuPy + DF) backends.

Theory
------
IC-CASPT2 expands the first-order wavefunction in internally contracted
basis functions :math:`|\Phi_P\rangle = \hat{E}_{pqrs\ldots}|\Psi_0\rangle`,
classified into 13 cases (A–H±) by the orbital subspace (inactive/active/
virtual) of the excitation indices. The PT2 energy is:

.. math::

    E_{\text{PT2}} = \sum_{c=1}^{13} \sum_P T_P^{(c)} V_P^{(c)}

where :math:`V_P = \langle \Phi_P|\hat{H}|\Psi_0\rangle` and the amplitudes
:math:`T_P` solve :math:`(\hat{H}_0 - E_0) T = -V`.

Module Map
----------
- ``superindex.py``: Superindex infrastructure for 13 IC cases
- ``fock.py`` / ``fock_df.py``: Fock matrix construction (full ERI / DF)
- ``overlap.py``: S matrix + joint S/B diagonalisation
- ``hzero.py``: Dyall's :math:`\hat{H}_0` (B matrix) for all cases
- ``rhs.py`` / ``rhs_df.py``: RHS coupling vectors :math:`V_P`
- ``f3.py``: Fock-contracted 4-body quantity engine (DELTA3)
- ``sigma.py``: Sigma-vector operator + inter-case couplings
- ``solver.py``: PCG linear system solver
- ``shifts.py``: IPEA / imaginary / real level shifts
- ``energy.py``: SS-CASPT2 energy workflow
- ``multistate.py`` / ``hcoup.py``: MS-CASPT2 effective Hamiltonian
- ``xms.py`` / ``xms_utils.py``: XMS state rotation
- ``driver_asuka.py``: End-to-end ASUKA workflow drivers
- ``result.py``: Result dataclasses
- ``pt2lag.py`` / ``pt2lag_df2e.py``: Gradient Lagrangian intermediates
- ``sigder_native.py``: SIGDER OFFDIAG for CLagDX

Primary entry points for end-to-end ASUKA workflows are:
  - :func:`run_caspt2` (ASUKA CASCIResult/CASSCFResult)

See ``asuka/caspt2/README.md`` for detailed conventions and usage examples.
"""

from asuka.caspt2.driver_asuka import run_caspt2, run_caspt2_soc, run_caspt2_soc_multispin
from asuka.caspt2.energy import caspt2_energy_ss
from asuka.caspt2.fock import CASPT2Fock, build_caspt2_fock
from asuka.caspt2.gradient.driver import caspt2_gradient_from_casscf
from asuka.caspt2.multistate import build_heff, diagonalize_heff
from asuka.caspt2.overlap import SBDecomposition, sbdiag
from asuka.caspt2.result import (
    CASPT2EnergyResult,
    CASPT2GradResult,
    CASPT2Result,
    CASPT2SOCResult,
    CASPT2SOCResultMultiSpin,
)
from asuka.caspt2.superindex import CASOrbitals, SuperindexMap, build_superindex
from asuka.caspt2.xms import xms_rotate_states

__all__ = [
    "run_caspt2",
    "run_caspt2_soc",
    "run_caspt2_soc_multispin",
    "caspt2_gradient_from_casscf",
    "caspt2_energy_ss",
    "CASPT2Fock",
    "build_caspt2_fock",
    "build_heff",
    "diagonalize_heff",
    "SBDecomposition",
    "sbdiag",
    "CASPT2EnergyResult",
    "CASPT2GradResult",
    "CASPT2Result",
    "CASPT2SOCResult",
    "CASPT2SOCResultMultiSpin",
    "CASOrbitals",
    "SuperindexMap",
    "build_superindex",
    "xms_rotate_states",
]
