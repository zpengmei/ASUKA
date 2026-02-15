from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .basis_cart import BasisCartSoA
from .dense_cpu import build_active_eri_mat_dense_cpu, build_active_eri_packed_dense_cpu, schwarz_shellpairs_cpu
from .pair_tables_cpu import PairTablesCPU, build_pair_tables_cpu
from .shell_pairs import ShellPairs, build_shell_pairs_l_order


@dataclass
class CuERIActiveSpaceDenseCPUBuilder:
    """Builder for CPU dense active-space electron repulsion integrals (ERIs) using cuERI.

    This class facilitates the construction of dense active-space ERIs while caching basis
    preprocessing steps (e.g., shell pairs, pair tables, and Schwarz screening bounds).
    It is designed to be reused across repeated active-space rebuilds, such as in
    CASSCF macro-iterations, to optimize performance.

    Parameters
    ----------
    ao_basis : BasisCartSoA | None, optional
        The Cartesian packed AO basis set. Required for initialization.
    max_l : int | None, optional
        The maximum angular momentum to support. If None, it is automatically determined
        from `ao_basis.shell_l`.
    max_tile_bytes : int, default=268435456 (256 MB)
        The maximum memory size in bytes for a single tile during ERI evaluation.
    threads : int, default=0
        The number of OpenMP threads to use. If 0, the system default is used.
    sp : ShellPairs | None, optional
        Pre-computed shell pairs. Computed during initialization if not provided.
    pair_tables : PairTablesCPU | None, optional
        Pre-computed pair tables. Computed during initialization if not provided.
    sp_Q : np.ndarray | None, optional
        Pre-computed Schwarz screening values. Computed on demand if not provided.

    Notes
    -----
    - The builder requires a Cartesian packed AO basis (`BasisCartSoA`).
    - The `max_l` parameter acts as a policy limit; shells with higher angular momentum
      will cause an error.
    """

    ao_basis: BasisCartSoA | None = None

    max_l: int | None = None
    max_tile_bytes: int = 256 << 20
    threads: int = 0

    # Optional cached preprocessing artifacts.
    sp: ShellPairs | None = None
    pair_tables: PairTablesCPU | None = None
    sp_Q: np.ndarray | None = None

    def __post_init__(self) -> None:
        max_tile_bytes = int(self.max_tile_bytes)
        if max_tile_bytes <= 0:
            raise ValueError("max_tile_bytes must be > 0")

        threads = int(self.threads)
        if threads < 0:
            raise ValueError("threads must be >= 0")

        ao_basis = self.ao_basis
        if ao_basis is None:
            raise ValueError("ao_basis is required")

        shell_l = np.asarray(getattr(ao_basis, "shell_l"), dtype=np.int32).ravel()
        shell_l_max = int(np.max(shell_l)) if int(shell_l.size) else 0
        max_l_cfg = self.max_l
        if max_l_cfg is None:
            max_l = int(shell_l_max)
        else:
            max_l = int(max_l_cfg)
            if max_l < 0:
                raise ValueError("max_l must be >= 0")
        if shell_l_max > max_l:
            raise NotImplementedError("basis has shells with l > max_l")
        object.__setattr__(self, "max_l", int(max_l))

        sp = self.sp
        if sp is None:
            sp = build_shell_pairs_l_order(ao_basis)
            object.__setattr__(self, "sp", sp)

        pair_tables = self.pair_tables
        if pair_tables is None:
            pair_tables = build_pair_tables_cpu(ao_basis, sp, threads=threads)
            object.__setattr__(self, "pair_tables", pair_tables)

    def _ensure_sp_Q(self, *, threads: int) -> np.ndarray:
        sp_Q = self.sp_Q
        if sp_Q is not None:
            return np.asarray(sp_Q, dtype=np.float64)

        ao_basis = self.ao_basis
        sp = self.sp
        pair_tables = self.pair_tables
        if ao_basis is None or sp is None or pair_tables is None:  # pragma: no cover
            raise RuntimeError("internal error: builder is missing cached ao_basis/sp/pair_tables")

        sp_Q = schwarz_shellpairs_cpu(ao_basis, sp, pair_tables=pair_tables, max_l=int(self.max_l), threads=int(threads))
        object.__setattr__(self, "sp_Q", np.asarray(sp_Q, dtype=np.float64, order="C"))
        return sp_Q

    def build_eri_packed(
        self,
        C_active: np.ndarray,
        *,
        eps_ao: float = 0.0,
        eps_mo: float = 0.0,
        blas_nthreads: int | None = None,
        profile: dict | None = None,
    ) -> np.ndarray:
        """Compute packed lower-triangular active-space ERIs on the CPU.

        This method evaluates the active-space ERI tensor and returns it in a packed
        lower-triangular format corresponding to `(npair, npair)` where
        `npair = nmo * (nmo + 1) / 2`.

        Parameters
        ----------
        C_active : np.ndarray
            The active space MO coefficients. Shape: (nao, nmo).
        eps_ao : float, default=0.0
            Screening threshold for AO integrals.
        eps_mo : float, default=0.0
            Screening threshold for MO transformation.
        blas_nthreads : int | None, optional
            Number of BLAS threads to use for contractions.
        profile : dict | None, optional
            Dictionary to collect profiling data.

        Returns
        -------
        np.ndarray
            The packed ERI matrix. Shape: (npair, npair).
        """
        ao_basis = self.ao_basis
        sp = self.sp
        pair_tables = self.pair_tables
        if ao_basis is None or sp is None or pair_tables is None:  # pragma: no cover
            raise RuntimeError("internal error: builder is missing cached ao_basis/sp/pair_tables")

        eps_ao_f = float(eps_ao)
        eps_mo_f = float(eps_mo)
        sp_Q = None
        if eps_ao_f > 0.0 or eps_mo_f > 0.0:
            sp_Q = self._ensure_sp_Q(threads=int(self.threads))

        return build_active_eri_packed_dense_cpu(
            ao_basis,
            C_active,
            eps_ao=eps_ao_f,
            eps_mo=eps_mo_f,
            sp=sp,
            sp_Q=sp_Q,
            pair_tables=pair_tables,
            max_l=int(self.max_l),
            max_tile_bytes=int(self.max_tile_bytes),
            threads=int(self.threads),
            blas_nthreads=blas_nthreads,
            profile=profile,
        )

    def build_eri_mat(
        self,
        C_active: np.ndarray,
        *,
        eps_ao: float = 0.0,
        eps_mo: float = 0.0,
        blas_nthreads: int | None = None,
        profile: dict | None = None,
    ) -> np.ndarray:
        """Compute full square active-space ERI matrix on the CPU.

        This method evaluates the active-space ERI tensor and returns it as a full
        square matrix in ordered-pair layout `(nmo^2, nmo^2)`.

        Parameters
        ----------
        C_active : np.ndarray
            The active space MO coefficients. Shape: (nao, nmo).
        eps_ao : float, default=0.0
            Screening threshold for AO integrals.
        eps_mo : float, default=0.0
            Screening threshold for MO transformation.
        blas_nthreads : int | None, optional
            Number of BLAS threads to use for contractions.
        profile : dict | None, optional
            Dictionary to collect profiling data.

        Returns
        -------
        np.ndarray
            The full ERI matrix. Shape: (nmo*nmo, nmo*nmo).
        """
        ao_basis = self.ao_basis
        sp = self.sp
        pair_tables = self.pair_tables
        if ao_basis is None or sp is None or pair_tables is None:  # pragma: no cover
            raise RuntimeError("internal error: builder is missing cached ao_basis/sp/pair_tables")

        eps_ao_f = float(eps_ao)
        eps_mo_f = float(eps_mo)
        sp_Q = None
        if eps_ao_f > 0.0 or eps_mo_f > 0.0:
            sp_Q = self._ensure_sp_Q(threads=int(self.threads))

        return build_active_eri_mat_dense_cpu(
            ao_basis,
            C_active,
            eps_ao=eps_ao_f,
            eps_mo=eps_mo_f,
            sp=sp,
            sp_Q=sp_Q,
            pair_tables=pair_tables,
            max_l=int(self.max_l),
            max_tile_bytes=int(self.max_tile_bytes),
            threads=int(self.threads),
            blas_nthreads=blas_nthreads,
            profile=profile,
        )


__all__ = ["CuERIActiveSpaceDenseCPUBuilder"]
