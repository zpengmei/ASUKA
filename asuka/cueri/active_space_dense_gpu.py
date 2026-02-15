from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .basis_cart import BasisCartSoA
from .basis_utils import shell_nfunc_cart
from .cart import ncart
from .gpu import CUDA_MAX_L, sph_coeff_sph_to_cart_device
from .mol_basis import SphMapForCartBasis, pack_cart_shells_from_mol_with_sph_map
from .shell_pairs import ShellPairs, build_shell_pairs_l_order
from .stream import stream_ctx
from .tasks import (
    TaskList,
    build_tasks_screened,
    build_tasks_screened_sorted_q,
    group_tasks_by_class,
    group_tasks_by_spab,
    with_task_class_id,
)


def _build_ao2shell_and_local_cart(shell_ao_start_cart: np.ndarray, shell_l: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Build cart AO-row -> expanded-shell index and local cart component index."""

    starts = np.asarray(shell_ao_start_cart, dtype=np.int32).ravel()
    ls = np.asarray(shell_l, dtype=np.int32).ravel()
    if starts.shape != ls.shape:
        raise ValueError("shell_ao_start_cart/shell_l shape mismatch")
    if starts.size == 0:
        return np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.int32), 0

    shell_ncart = np.asarray([ncart(int(l)) for l in ls], dtype=np.int32)
    nao_cart = int(np.max(starts.astype(np.int64) + shell_ncart.astype(np.int64)))
    ao2shell = np.empty((nao_cart,), dtype=np.int32)
    ao2local = np.empty((nao_cart,), dtype=np.int32)
    for sh, (a0, nc) in enumerate(zip(starts, shell_ncart, strict=False)):
        a1 = int(a0) + int(nc)
        ao2shell[int(a0) : a1] = np.int32(sh)
        ao2local[int(a0) : a1] = np.arange(int(nc), dtype=np.int32)
    return ao2shell, ao2local, nao_cart


@dataclass
class DeviceActiveDense:
    norb: int
    eri_mat: Any  # cp.ndarray, shape (norb*norb, norb*norb)
    j_ps: Any | None  # cp.ndarray, shape (norb, norb) or None


@dataclass
class CuERIActiveSpaceDenseGPUBuilder:
    """Builder for GPU dense active-space electron repulsion integrals (ERIs) via cuERI.

    This class provides a cached, reusable interface for computing active-space ERI matrices
    on the GPU. It handles basis preprocessing, task scheduling, and kernel execution
    parameters, optimizing performance for repeated active-space rebuilds (e.g., in CASSCF
    algorithms).

    Parameters
    ----------
    mol : object | None, optional
        A PySCF-like molecule object used to derive basis information if `ao_basis` is absent.
    ao_basis : BasisCartSoA | None, optional
        The Cartesian packed AO basis set. One of `mol` or `ao_basis` must be provided.
    max_l : int, default=CUDA_MAX_L
        The maximum angular momentum supported by the compiled CUDA kernels.
    threads : int, default=256
        Block size (threads per block) for CUDA kernel execution.
    max_tile_bytes : int, default=268435456 (256 MB)
        Maximum size in bytes for temporary tile buffers on the GPU.
    eps_ao : float, default=0.0
        Screening threshold for AO integrals.
    sort_q : bool, default=True
        Whether to sort tasks by their Schwarz screening value for better load balancing.
    sp_Q : np.ndarray | None, optional
        Pre-computed Schwarz screening values.
    stream : cupy.cuda.Stream | None, optional
        The CUDA stream to use for asynchronous execution.
    algorithm : str, default='auto'
        Dispatch algorithm: 'auto', 'ab_group', or 'direct'.
    pair_block : int, default=0
        Block size for pair scheduling.
    mode : str, default='auto'
        Kernel execution mode: 'auto', 'warp', 'block', or 'multiblock'.
    work_small_max : int, default=512
        Workload threshold for "small" kernel dispatch paths.
    work_large_min : int, default=200000
        Workload threshold for "large" kernel dispatch paths.
    blocks_per_task : int, default=8
        Number of blocks to assign per task in certain dispatch modes.
    boys : str, default='ref'
        Algorithm for Boys function evaluation: 'ref' or 'fast'.
    ao_rep : str, default='auto'
        Representation of AO basis: 'auto', 'cart' (Cartesian), or 'sph' (Spherical).

    Notes
    -----
    - Internal integral evaluation is performed in Cartesian coordinates.
    - If spherical AOs are required, the builder handles the transformation on the GPU.
    - Requires a CUDA-enabled environment with the cuERI extension built.
    """

    # One of (mol) or (ao_basis) must be provided.
    mol: Any | None = None
    ao_basis: BasisCartSoA | None = None

    max_l: int = CUDA_MAX_L
    threads: int = 256
    max_tile_bytes: int = 256 << 20
    eps_ao: float = 0.0
    sort_q: bool = True
    sp_Q: np.ndarray | None = None
    stream: Any | None = None
    algorithm: str = "auto"
    pair_block: int = 0
    mode: str = "auto"
    work_small_max: int = 512
    work_large_min: int = 200_000
    blocks_per_task: int = 8
    boys: str = "ref"
    ao_rep: str = "auto"  # "auto"|"cart"|"sph"

    # Cached preprocessing artifacts.
    sp: ShellPairs | None = None
    Q_np: np.ndarray | None = None
    tasks: TaskList | None = None
    perm: np.ndarray | None = None
    class_ids: np.ndarray | None = None
    offsets: np.ndarray | None = None
    task_ab: np.ndarray | None = None
    task_cd: np.ndarray | None = None
    perm_ab: np.ndarray | None = None
    ab_offsets: np.ndarray | None = None
    task_cd_by_ab: np.ndarray | None = None
    shell_ao_start_sph: np.ndarray | None = None
    ao2shell_cart: np.ndarray | None = None
    ao2local_cart: np.ndarray | None = None

    # Cached device-side artifacts.
    dbasis: Any | None = None
    dsp: Any | None = None
    pair_tables: Any | None = None
    d_shell_ao_start_sph: Any | None = None
    d_ao2shell_cart: Any | None = None
    d_ao2local_cart: Any | None = None
    d_shell_l: Any | None = None
    device_id: int | None = None

    # Cached AO dimensions for runtime validation.
    nao_expected: int | None = None  # backwards-compatible alias of nao_expected_in
    nao_expected_in: int | None = None
    nao_expected_eval: int | None = None

    def __post_init__(self) -> None:
        max_l = int(self.max_l)
        if max_l < 0:
            raise ValueError("max_l must be >= 0")
        if max_l > CUDA_MAX_L:
            raise NotImplementedError(f"CuERIActiveSpaceDenseGPUBuilder currently supports only max_l<={CUDA_MAX_L}")

        threads = int(self.threads)
        if threads <= 0:
            raise ValueError("threads must be > 0")

        max_tile_bytes = int(self.max_tile_bytes)
        if max_tile_bytes <= 0:
            raise ValueError("max_tile_bytes must be > 0")

        eps_ao = float(self.eps_ao)
        if eps_ao < 0.0:
            raise ValueError("eps_ao must be >= 0")

        algorithm = str(self.algorithm).lower().replace("-", "_").strip()
        if algorithm not in ("auto", "ab_group", "direct"):
            raise ValueError("algorithm must be one of {'auto','ab_group','direct'}")
        object.__setattr__(self, "algorithm", algorithm)

        pair_block = int(self.pair_block)
        if pair_block < 0:
            raise ValueError("pair_block must be >= 0")
        object.__setattr__(self, "pair_block", pair_block)

        mode = str(self.mode).lower().strip()
        if mode not in ("auto", "warp", "block", "multiblock"):
            raise ValueError("mode must be one of {'auto','warp','block','multiblock'}")
        object.__setattr__(self, "mode", mode)

        work_small_max = int(self.work_small_max)
        if work_small_max < 0:
            raise ValueError("work_small_max must be >= 0")
        object.__setattr__(self, "work_small_max", work_small_max)

        work_large_min = int(self.work_large_min)
        if work_large_min < 0:
            raise ValueError("work_large_min must be >= 0")
        object.__setattr__(self, "work_large_min", work_large_min)

        blocks_per_task = int(self.blocks_per_task)
        if blocks_per_task <= 0:
            raise ValueError("blocks_per_task must be > 0")
        object.__setattr__(self, "blocks_per_task", blocks_per_task)

        boys = str(self.boys).lower().strip()
        if boys not in ("ref", "fast"):
            raise ValueError("boys must be one of {'ref','fast'}")
        object.__setattr__(self, "boys", boys)

        ao_rep = str(self.ao_rep).lower().strip()
        if ao_rep not in ("auto", "cart", "sph"):
            raise ValueError("ao_rep must be one of {'auto','cart','sph'}")
        object.__setattr__(self, "ao_rep", ao_rep)

        mol = self.mol
        ao_basis = self.ao_basis
        sph_map: SphMapForCartBasis | None = None
        if ao_basis is None:
            if mol is None:
                raise ValueError("mol or ao_basis must be provided")

            try:
                ao_basis, sph_map = pack_cart_shells_from_mol_with_sph_map(mol, expand_contractions=True)
            except Exception as e:  # pragma: no cover
                raise RuntimeError(
                    "asuka.cueri.pack_cart_shells_from_mol_with_sph_map is required for "
                    "CuERIActiveSpaceDenseGPUBuilder(mol=...)"
                ) from e
            object.__setattr__(self, "ao_basis", ao_basis)

        ao_rep_eff = ao_rep
        if ao_rep_eff == "auto":
            if mol is None:
                ao_rep_eff = "cart"
            else:
                ao_rep_eff = "cart" if bool(getattr(mol, "cart", False)) else "sph"

        if ao_rep_eff == "sph" and sph_map is None:
            if mol is None:
                raise ValueError("ao_rep='sph' requires mol=... so spherical AO offsets can be derived")
            try:
                _basis_ref, sph_map = pack_cart_shells_from_mol_with_sph_map(mol, expand_contractions=True)
            except Exception as e:  # pragma: no cover
                raise RuntimeError("failed to derive spherical AO offsets from mol for ao_rep='sph'") from e
        object.__setattr__(self, "ao_rep", ao_rep_eff)

        shell_l = np.asarray(getattr(ao_basis, "shell_l"), dtype=np.int32).ravel()
        if int(shell_l.size) and int(np.max(shell_l)) > max_l:
            raise NotImplementedError("basis has shells with l > max_l")

        ao_starts_cart = np.asarray(ao_basis.shell_ao_start, dtype=np.int32).ravel()
        ao_ends = ao_starts_cart.astype(np.int64) + np.asarray(shell_nfunc_cart(ao_basis), dtype=np.int64)
        nao_eval = int(ao_ends.max()) if ao_ends.size else 0
        object.__setattr__(self, "nao_expected_eval", nao_eval)

        if ao_rep_eff == "sph":
            if sph_map is None:  # pragma: no cover
                raise RuntimeError("internal error: missing spherical AO map for ao_rep='sph'")
            shell_ao_start_sph = np.asarray(sph_map.shell_ao_start_sph, dtype=np.int32).ravel()
            if shell_ao_start_sph.shape != shell_l.shape:
                raise ValueError("spherical shell offset map is incompatible with packed cart basis")
            ao2shell_cart, ao2local_cart, nao_cart_mapped = _build_ao2shell_and_local_cart(ao_starts_cart, shell_l)
            if int(nao_cart_mapped) != int(nao_eval):
                raise RuntimeError("internal error: cart AO mapping size mismatch")
            object.__setattr__(self, "shell_ao_start_sph", shell_ao_start_sph)
            object.__setattr__(self, "ao2shell_cart", ao2shell_cart)
            object.__setattr__(self, "ao2local_cart", ao2local_cart)
            nao_expected_in = int(sph_map.nao_sph)
        else:
            object.__setattr__(self, "shell_ao_start_sph", None)
            object.__setattr__(self, "ao2shell_cart", None)
            object.__setattr__(self, "ao2local_cart", None)
            nao_expected_in = int(nao_eval)

        object.__setattr__(self, "nao_expected_in", nao_expected_in)
        object.__setattr__(self, "nao_expected", nao_expected_in)

        sp = self.sp
        if sp is None:
            sp = build_shell_pairs_l_order(ao_basis)
            object.__setattr__(self, "sp", sp)

        self._prepare_tasks()
        self._prepare_device()

    def _prepare_tasks(self) -> None:
        basis = self.ao_basis
        sp = self.sp
        if basis is None or sp is None:  # pragma: no cover
            raise RuntimeError("internal error: builder is missing cached ao_basis/sp")

        nsp = int(np.asarray(sp.sp_A, dtype=np.int32).shape[0])
        if nsp == 0:
            object.__setattr__(self, "Q_np", np.zeros((0,), dtype=np.float64))
            object.__setattr__(self, "tasks", TaskList(np.zeros((0,), np.int32), np.zeros((0,), np.int32)))
            object.__setattr__(self, "perm", np.zeros((0,), np.int32))
            object.__setattr__(self, "class_ids", np.zeros((0,), np.int32))
            object.__setattr__(self, "offsets", np.zeros((1,), np.int32))
            object.__setattr__(self, "task_ab", np.zeros((0,), np.int32))
            object.__setattr__(self, "task_cd", np.zeros((0,), np.int32))
            object.__setattr__(self, "perm_ab", np.zeros((0,), np.int32))
            object.__setattr__(self, "ab_offsets", np.zeros((1,), np.int32))
            object.__setattr__(self, "task_cd_by_ab", np.zeros((0,), np.int32))
            return

        eps_ao = float(self.eps_ao)

        # Q_np (shell-pair Schwarz bounds).
        sp_Q = self.sp_Q
        if sp_Q is not None:
            Q_np = np.asarray(sp_Q, dtype=np.float64).ravel()
            if Q_np.shape != (nsp,):
                raise ValueError(f"sp_Q must have shape (nsp,), got {Q_np.shape} (nsp={nsp})")
        elif eps_ao > 0.0:
            try:
                import cupy as cp  # noqa: PLC0415
            except Exception as e:  # pragma: no cover
                raise RuntimeError("CuPy is required for GPU dense builder") from e
            from .screening import schwarz_shellpairs_device  # noqa: PLC0415

            with stream_ctx(self.stream):
                Q_dev = schwarz_shellpairs_device(
                    basis,
                    sp,
                    stream=None,
                    threads=int(self.threads),
                    max_tiles_bytes=int(self.max_tile_bytes),
                )
                Q_np = cp.asnumpy(Q_dev)
                cp.cuda.get_current_stream().synchronize()
        else:
            Q_np = np.ones((nsp,), dtype=np.float64)

        Q_np = np.asarray(Q_np, dtype=np.float64, order="C")
        object.__setattr__(self, "Q_np", Q_np)

        # Canonical screened task list.
        if eps_ao > 0.0 and bool(self.sort_q):
            tasks = build_tasks_screened_sorted_q(Q_np, eps=eps_ao)
        else:
            tasks = build_tasks_screened(Q_np, eps=eps_ao)
        if int(tasks.ntasks) == 0:
            object.__setattr__(self, "tasks", tasks)
            object.__setattr__(self, "perm", np.zeros((0,), np.int32))
            object.__setattr__(self, "class_ids", np.zeros((0,), np.int32))
            object.__setattr__(self, "offsets", np.zeros((1,), np.int32))
            object.__setattr__(self, "task_ab", np.zeros((0,), np.int32))
            object.__setattr__(self, "task_cd", np.zeros((0,), np.int32))
            object.__setattr__(self, "perm_ab", np.zeros((0,), np.int32))
            object.__setattr__(self, "ab_offsets", np.zeros((nsp + 1,), np.int32))
            object.__setattr__(self, "task_cd_by_ab", np.zeros((0,), np.int32))
            return

        shell_l = np.asarray(basis.shell_l, dtype=np.int32).ravel()
        tasks = with_task_class_id(tasks, sp, shell_l)
        assert tasks.task_class_id is not None
        perm, class_ids, offsets = group_tasks_by_class(tasks.task_class_id)

        object.__setattr__(self, "tasks", tasks)
        object.__setattr__(self, "perm", perm)
        object.__setattr__(self, "class_ids", class_ids)
        object.__setattr__(self, "offsets", offsets)
        object.__setattr__(self, "task_ab", np.asarray(tasks.task_spAB[perm], dtype=np.int32))
        object.__setattr__(self, "task_cd", np.asarray(tasks.task_spCD[perm], dtype=np.int32))
        perm_ab, ab_offsets = group_tasks_by_spab(tasks.task_spAB, nsp=nsp)
        object.__setattr__(self, "perm_ab", perm_ab)
        object.__setattr__(self, "ab_offsets", ab_offsets)
        object.__setattr__(self, "task_cd_by_ab", np.asarray(tasks.task_spCD[perm_ab], dtype=np.int32))

    def _prepare_device(self) -> None:
        try:
            import cupy as cp  # noqa: PLC0415
        except Exception as e:  # pragma: no cover
            raise RuntimeError("CuPy is required for GPU dense builder") from e

        from .gpu import build_pair_tables_ss_device, has_cuda_ext, to_device_basis_ss, to_device_shell_pairs  # noqa: PLC0415

        if not has_cuda_ext():
            raise RuntimeError("cuERI CUDA extension not available; build via `python -m asuka.cueri.build_cuda_ext`")

        basis = self.ao_basis
        sp = self.sp
        if basis is None or sp is None:  # pragma: no cover
            raise RuntimeError("internal error: builder is missing cached ao_basis/sp")

        d_shell_ao_start_sph = None
        d_ao2shell_cart = None
        d_ao2local_cart = None
        d_shell_l = None
        with stream_ctx(self.stream):
            object.__setattr__(self, "device_id", int(cp.cuda.runtime.getDevice()))
            dbasis = to_device_basis_ss(basis)
            dsp = to_device_shell_pairs(sp)
            pt = build_pair_tables_ss_device(dbasis, dsp, stream=None, threads=int(self.threads))
            if str(self.ao_rep) == "sph":
                shell_ao_start_sph = self.shell_ao_start_sph
                ao2shell_cart = self.ao2shell_cart
                ao2local_cart = self.ao2local_cart
                if shell_ao_start_sph is None or ao2shell_cart is None or ao2local_cart is None:  # pragma: no cover
                    raise RuntimeError("internal error: missing spherical/cart mapping arrays on builder")
                d_shell_ao_start_sph = cp.ascontiguousarray(cp.asarray(shell_ao_start_sph, dtype=cp.int32))
                d_ao2shell_cart = cp.ascontiguousarray(cp.asarray(ao2shell_cart, dtype=cp.int32))
                d_ao2local_cart = cp.ascontiguousarray(cp.asarray(ao2local_cart, dtype=cp.int32))
                d_shell_l = cp.ascontiguousarray(cp.asarray(basis.shell_l, dtype=cp.int32))

            # Synchronize before caching so later use on any stream is safe.
            cp.cuda.get_current_stream().synchronize()

        object.__setattr__(self, "dbasis", dbasis)
        object.__setattr__(self, "dsp", dsp)
        object.__setattr__(self, "pair_tables", pt)
        object.__setattr__(self, "d_shell_ao_start_sph", d_shell_ao_start_sph)
        object.__setattr__(self, "d_ao2shell_cart", d_ao2shell_cart)
        object.__setattr__(self, "d_ao2local_cart", d_ao2local_cart)
        object.__setattr__(self, "d_shell_l", d_shell_l)

    def allocate(self, norb: int, *, want_j_ps: bool = True) -> DeviceActiveDense:
        try:
            import cupy as cp  # noqa: PLC0415
        except Exception as e:  # pragma: no cover
            raise RuntimeError("CuPy is required for GPU dense builder") from e

        norb_i = int(norb)
        if norb_i <= 0:
            raise ValueError("norb must be > 0")
        n_pair = norb_i * norb_i
        eri_mat = cp.empty((n_pair, n_pair), dtype=cp.float64)
        j_ps = cp.empty((norb_i, norb_i), dtype=cp.float64) if want_j_ps else None
        return DeviceActiveDense(norb=norb_i, eri_mat=eri_mat, j_ps=j_ps)

    def build(self, C_active, *, out: DeviceActiveDense | None = None, profile: dict | None = None) -> DeviceActiveDense:
        """Compute the ordered-pair active-space ERI matrix on the GPU.

        This method evaluates the exact ERIs for the provided active orbitals and returns
        the result as an ordered-pair matrix (and optionally the J_ps matrix) on the GPU.

        Parameters
        ----------
        C_active : np.ndarray | cupy.ndarray
            Active-space MO coefficients with shape `(nao, norb)`.
        out : DeviceActiveDense | None, optional
            Pre-allocated output container. If None, a new one is allocated.
        profile : dict | None, optional
            Dictionary to collect timing and profiling metadata.

        Returns
        -------
        DeviceActiveDense
            A data structure containing the computed ERI matrix (`eri_mat`) and
            optionally `j_ps`.
        """

        try:
            import cupy as cp  # noqa: PLC0415
        except Exception as e:  # pragma: no cover
            raise RuntimeError("CuPy is required for GPU dense builder") from e

        from .dense import _build_active_eri_mat_dense_rys_from_cached  # noqa: PLC0415
        from .eri_utils import j_ps_from_eri_mat  # noqa: PLC0415

        if int(cp.cuda.runtime.getDevice()) != int(self.device_id):
            raise RuntimeError("CuERIActiveSpaceDenseGPUBuilder used on a different CUDA device")

        with stream_ctx(self.stream):
            C_active_dev = cp.asarray(C_active, dtype=cp.float64)
            C_active_dev = cp.ascontiguousarray(C_active_dev)
            if C_active_dev.ndim != 2:
                raise ValueError("C_active must have shape (nao, norb)")

            nao_in, norb = map(int, C_active_dev.shape)
            nao_expected_in = int(self.nao_expected_in or 0)
            if nao_expected_in != nao_in:
                raise ValueError(
                    f"C_active has nao={nao_in}, but builder expects nao={nao_expected_in} for ao_rep={self.ao_rep!r}"
                )

            C_eval = C_active_dev
            t_transform_s = 0.0
            if str(self.ao_rep) == "sph":
                d_ao2shell_cart = self.d_ao2shell_cart
                d_ao2local_cart = self.d_ao2local_cart
                d_shell_ao_start_sph = self.d_shell_ao_start_sph
                d_shell_l = self.d_shell_l
                if (
                    d_shell_l is None
                    or d_ao2shell_cart is None
                    or d_ao2local_cart is None
                    or d_shell_ao_start_sph is None
                ):  # pragma: no cover
                    raise RuntimeError("internal error: missing spherical mapping/device arrays on builder")
                if profile is not None:
                    cp.cuda.get_current_stream().synchronize()
                    t_conv0 = time.perf_counter()
                C_eval = sph_coeff_sph_to_cart_device(
                    C_active_dev,
                    ao2shell_cart=d_ao2shell_cart,
                    ao2local_cart=d_ao2local_cart,
                    shell_ao_start_sph=d_shell_ao_start_sph,
                    shell_l=d_shell_l,
                    out=None,
                    stream=None,
                    threads=int(self.threads),
                )
                if profile is not None:
                    cp.cuda.get_current_stream().synchronize()
                    t_transform_s = float(time.perf_counter() - float(t_conv0))
                nao_eval = int(self.nao_expected_eval or 0)
                if int(C_eval.shape[0]) != nao_eval:
                    raise RuntimeError("internal error: spherical->cart transform produced unexpected AO dimension")
            else:
                nao_eval = int(self.nao_expected_eval or 0)
                if nao_eval != nao_in:
                    raise RuntimeError("internal error: cart ao_rep expects nao_in == nao_eval")

            if out is None:
                out = self.allocate(norb, want_j_ps=True)
            else:
                if int(out.norb) != int(norb):
                    raise ValueError("out.norb mismatch")

            t0 = time.perf_counter() if profile is not None else None
            _build_active_eri_mat_dense_rys_from_cached(builder=self, C_active=C_eval, out_eri_mat=out.eri_mat, profile=profile)
            if out.j_ps is not None:
                out.j_ps[...] = j_ps_from_eri_mat(out.eri_mat, norb=norb)

            if profile is not None and t0 is not None:
                cp.cuda.get_current_stream().synchronize()
                prof = profile.setdefault("cueri_dense_rys_cached", {})
                prof["t_total_s"] = float(time.perf_counter() - float(t0))
                prof["norb"] = int(norb)
                prof["ao_rep"] = str(self.ao_rep)
                prof["nao_in"] = int(nao_in)
                prof["nao_eval"] = int(nao_eval)
                if str(self.ao_rep) == "sph":
                    prof["t_sph_to_cart_s"] = float(t_transform_s)

            return out

    def build_pu_wx_eri_mat(self, C_mo, C_act, *, out=None, profile: dict | None = None):
        """Compute mixed-index ERIs (pu|wx) as an ordered-pair matrix on the GPU.

        Evaluates integrals of the form (pu|wx) where p is a general MO, u is an
        active orbital, and w, x are active orbitals.

        Parameters
        ----------
        C_mo : np.ndarray | cupy.ndarray
            General MO coefficients defining index `p`. Shape: `(nao, nmo)`.
        C_act : np.ndarray | cupy.ndarray
            Active MO coefficients defining indices `u, w, x`. Shape: `(nao, ncas)`.
        out : cupy.ndarray | None, optional
            Pre-allocated output array of shape `(nmo*ncas, ncas*ncas)`.
        profile : dict | None, optional
            Dictionary for profiling data.

        Returns
        -------
        cupy.ndarray
            The computed ERI matrix with shape `(nmo*ncas, ncas*ncas)`.
            Row index `pu = p*ncas + u`, Column index `wx = w*ncas + x`.
        """

        try:
            import cupy as cp  # noqa: PLC0415
        except Exception as e:  # pragma: no cover
            raise RuntimeError("CuPy is required for GPU dense builder") from e

        from .dense import _build_pu_wx_eri_mat_dense_rys_from_cached  # noqa: PLC0415

        if int(cp.cuda.runtime.getDevice()) != int(self.device_id):
            raise RuntimeError("CuERIActiveSpaceDenseGPUBuilder used on a different CUDA device")

        with stream_ctx(self.stream):
            C_mo_dev = cp.asarray(C_mo, dtype=cp.float64)
            C_act_dev = cp.asarray(C_act, dtype=cp.float64)
            C_mo_dev = cp.ascontiguousarray(C_mo_dev)
            C_act_dev = cp.ascontiguousarray(C_act_dev)
            if C_mo_dev.ndim != 2 or C_act_dev.ndim != 2:
                raise ValueError("C_mo/C_act must have shape (nao, nmo)/(nao, ncas)")
            nao_in, nmo = map(int, C_mo_dev.shape)
            nao_in2, ncas = map(int, C_act_dev.shape)
            if nao_in2 != nao_in:
                raise ValueError("C_mo/C_act nao mismatch")

            nao_expected_in = int(self.nao_expected_in or 0)
            if nao_expected_in != nao_in:
                raise ValueError(
                    f"C_mo/C_act have nao={nao_in}, but builder expects nao={nao_expected_in} for ao_rep={self.ao_rep!r}"
                )

            C_mo_eval = C_mo_dev
            C_act_eval = C_act_dev
            t_transform_s = 0.0
            if str(self.ao_rep) == "sph":
                d_shell_l = self.d_shell_l
                d_ao2shell_cart = self.d_ao2shell_cart
                d_ao2local_cart = self.d_ao2local_cart
                d_shell_ao_start_sph = self.d_shell_ao_start_sph
                if (
                    d_shell_l is None
                    or d_ao2shell_cart is None
                    or d_ao2local_cart is None
                    or d_shell_ao_start_sph is None
                ):  # pragma: no cover
                    raise RuntimeError("internal error: missing spherical mapping/device arrays on builder")
                if profile is not None:
                    cp.cuda.get_current_stream().synchronize()
                    t_conv0 = time.perf_counter()
                C_mo_eval = sph_coeff_sph_to_cart_device(
                    C_mo_dev,
                    ao2shell_cart=d_ao2shell_cart,
                    ao2local_cart=d_ao2local_cart,
                    shell_ao_start_sph=d_shell_ao_start_sph,
                    shell_l=d_shell_l,
                    out=None,
                    stream=None,
                    threads=int(self.threads),
                )
                C_act_eval = sph_coeff_sph_to_cart_device(
                    C_act_dev,
                    ao2shell_cart=d_ao2shell_cart,
                    ao2local_cart=d_ao2local_cart,
                    shell_ao_start_sph=d_shell_ao_start_sph,
                    shell_l=d_shell_l,
                    out=None,
                    stream=None,
                    threads=int(self.threads),
                )
                if profile is not None:
                    cp.cuda.get_current_stream().synchronize()
                    t_transform_s = float(time.perf_counter() - float(t_conv0))
                nao_eval = int(self.nao_expected_eval or 0)
                if int(C_mo_eval.shape[0]) != nao_eval or int(C_act_eval.shape[0]) != nao_eval:
                    raise RuntimeError("internal error: spherical->cart transform produced unexpected AO dimension")
            else:
                nao_eval = int(self.nao_expected_eval or 0)
                if nao_eval != nao_in:
                    raise RuntimeError("internal error: cart ao_rep expects nao_in == nao_eval")

            n_pair_left = int(nmo) * int(ncas)
            n_pair_right = int(ncas) * int(ncas)
            if out is None:
                out = cp.empty((n_pair_left, n_pair_right), dtype=cp.float64)
            else:
                if getattr(out, "shape", None) != (n_pair_left, n_pair_right):
                    raise ValueError(f"out must have shape {(n_pair_left, n_pair_right)}, got {getattr(out, 'shape', None)}")
                out = cp.asarray(out, dtype=cp.float64)

            t0 = time.perf_counter() if profile is not None else None
            _build_pu_wx_eri_mat_dense_rys_from_cached(
                builder=self,
                C_mo=C_mo_eval,
                C_act=C_act_eval,
                out_eri_puwx=out,
                profile=profile,
            )

            if profile is not None and t0 is not None:
                cp.cuda.get_current_stream().synchronize()
                prof = profile.setdefault("cueri_pu_wx_dense_rys_cached", {})
                prof["t_total_s"] = float(time.perf_counter() - float(t0))
                prof["ao_rep"] = str(self.ao_rep)
                prof["nao_in"] = int(nao_in)
                prof["nao_eval"] = int(nao_eval)
                if str(self.ao_rep) == "sph":
                    prof["t_sph_to_cart_s"] = float(t_transform_s)

            return out

    def build_pq_uv_eri_mat(self, C_mo, C_act, *, out=None, profile: dict | None = None):
        """Compute (pq|uv) ERIs where p,q are general MOs and u,v are active.

        Parameters
        ----------
        C_mo : np.ndarray | cupy.ndarray
            General MO coefficients defining indices `p, q`. Shape: `(nao, nmo)`.
        C_act : np.ndarray | cupy.ndarray
            Active MO coefficients defining indices `u, v`. Shape: `(nao, ncas)`.
        out : cupy.ndarray | None, optional
            Pre-allocated output array of shape `(nmo*nmo, ncas*ncas)`.
        profile : dict | None, optional
            Dictionary for profiling data.

        Returns
        -------
        cupy.ndarray
            The computed ERI matrix with shape `(nmo*nmo, ncas*ncas)`.
            Row index `pq = p*nmo + q`, Column index `uv = u*ncas + v`.
        """

        try:
            import cupy as cp  # noqa: PLC0415
        except Exception as e:  # pragma: no cover
            raise RuntimeError("CuPy is required for GPU dense builder") from e

        from .dense import _build_pq_uv_eri_mat_dense_rys_from_cached  # noqa: PLC0415

        if int(cp.cuda.runtime.getDevice()) != int(self.device_id):
            raise RuntimeError("CuERIActiveSpaceDenseGPUBuilder used on a different CUDA device")

        with stream_ctx(self.stream):
            C_mo_dev = cp.ascontiguousarray(cp.asarray(C_mo, dtype=cp.float64))
            C_act_dev = cp.ascontiguousarray(cp.asarray(C_act, dtype=cp.float64))
            if C_mo_dev.ndim != 2 or C_act_dev.ndim != 2:
                raise ValueError("C_mo/C_act must have shape (nao, nmo)/(nao, ncas)")
            nao_in, nmo = map(int, C_mo_dev.shape)
            nao_in2, ncas = map(int, C_act_dev.shape)
            if nao_in2 != nao_in:
                raise ValueError("C_mo/C_act nao mismatch")

            nao_expected_in = int(self.nao_expected_in or 0)
            if nao_expected_in != nao_in:
                raise ValueError(
                    f"C_mo/C_act have nao={nao_in}, but builder expects nao={nao_expected_in}"
                )

            C_mo_eval = C_mo_dev
            C_act_eval = C_act_dev
            t_transform_s = 0.0
            if str(self.ao_rep) == "sph":
                d_shell_l = self.d_shell_l
                d_ao2shell_cart = self.d_ao2shell_cart
                d_ao2local_cart = self.d_ao2local_cart
                d_shell_ao_start_sph = self.d_shell_ao_start_sph
                if any(x is None for x in (d_shell_l, d_ao2shell_cart, d_ao2local_cart, d_shell_ao_start_sph)):
                    raise RuntimeError("missing spherical mapping/device arrays")
                if profile is not None:
                    cp.cuda.get_current_stream().synchronize()
                    t_conv0 = time.perf_counter()
                C_mo_eval = sph_coeff_sph_to_cart_device(
                    C_mo_dev, ao2shell_cart=d_ao2shell_cart, ao2local_cart=d_ao2local_cart,
                    shell_ao_start_sph=d_shell_ao_start_sph, shell_l=d_shell_l,
                    out=None, stream=None, threads=int(self.threads),
                )
                C_act_eval = sph_coeff_sph_to_cart_device(
                    C_act_dev, ao2shell_cart=d_ao2shell_cart, ao2local_cart=d_ao2local_cart,
                    shell_ao_start_sph=d_shell_ao_start_sph, shell_l=d_shell_l,
                    out=None, stream=None, threads=int(self.threads),
                )
                if profile is not None:
                    cp.cuda.get_current_stream().synchronize()
                    t_transform_s = float(time.perf_counter() - float(t_conv0))
                nao_eval = int(self.nao_expected_eval or 0)
                if int(C_mo_eval.shape[0]) != nao_eval or int(C_act_eval.shape[0]) != nao_eval:
                    raise RuntimeError("spherical->cart transform produced unexpected AO dimension")
            else:
                nao_eval = int(self.nao_expected_eval or 0)
                if nao_eval != nao_in:
                    raise RuntimeError("cart ao_rep expects nao_in == nao_eval")

            n_pair_left = int(nmo) * int(nmo)
            n_pair_right = int(ncas) * int(ncas)
            if out is None:
                out = cp.empty((n_pair_left, n_pair_right), dtype=cp.float64)
            else:
                if getattr(out, "shape", None) != (n_pair_left, n_pair_right):
                    raise ValueError(f"out must have shape {(n_pair_left, n_pair_right)}")
                out = cp.asarray(out, dtype=cp.float64)

            t0 = time.perf_counter() if profile is not None else None
            _build_pq_uv_eri_mat_dense_rys_from_cached(
                builder=self, C_mo=C_mo_eval, C_act=C_act_eval,
                out_eri_pquv=out, profile=profile,
            )

            if profile is not None and t0 is not None:
                cp.cuda.get_current_stream().synchronize()
                prof = profile.setdefault("cueri_pq_uv_dense_rys_cached", {})
                prof["t_total_s"] = float(time.perf_counter() - float(t0))
                prof["ao_rep"] = str(self.ao_rep)
                prof["nao_in"] = int(nao_in)
                prof["nao_eval"] = int(nao_eval)
                if str(self.ao_rep) == "sph":
                    prof["t_sph_to_cart_s"] = float(t_transform_s)

            return out


__all__ = ["CuERIActiveSpaceDenseGPUBuilder", "DeviceActiveDense"]
