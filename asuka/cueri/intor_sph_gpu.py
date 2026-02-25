from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from .cart import ncart
from .eri_utils import npair
from .gpu import (
    CUDA_MAX_L,
    build_pair_tables_ss_device,
    cart2sph_eri_tiles_device,
    has_cuda_ext,
    scatter_eri_tiles_sph_s4_inplace_device,
    scatter_eri_tiles_sph_s8_inplace_device,
    to_device_basis_ss,
    to_device_shell_pairs,
)
from .mol_basis import SphMapForCartBasis, pack_cart_shells_from_mol_with_sph_map
from .shell_pairs import ShellPairs, build_shell_pairs_l_order
from .tasks import TaskList, build_tasks_screened_sorted_q, decode_eri_class_id
from .tile_eval import iter_tile_batches_spd


def _build_full_tasks_lower_triangle(nsp: int) -> TaskList:
    """All shell-pair quartets (AB,CD) with CD<=AB."""

    if nsp < 0:
        raise ValueError("nsp must be >= 0")
    if nsp == 0:
        return TaskList(task_spAB=np.zeros((0,), dtype=np.int32), task_spCD=np.zeros((0,), dtype=np.int32))

    ab, cd = np.tril_indices(int(nsp), k=0)
    return TaskList(task_spAB=np.asarray(ab, dtype=np.int32), task_spCD=np.asarray(cd, dtype=np.int32))


@dataclass
class CuERISphERIEngine:
    """PySCF-like AO ERI backend producing spherical AO integrals on the GPU.

    This engine evaluates contracted ERIs in Cartesian representation using existing
    cuERI CUDA kernels, then applies a native GPU Cartesian->spherical transform
    per tile, and finally scatters into packed AO layouts.

    Supported output layouts
    - aosym='s8': 8-fold symmetry packed vector over AO pairs-of-pairs.
    - aosym='s4': 4-fold symmetry square matrix over packed AO pairs.
    """

    mol: Any
    ao_basis: Any
    sph_map: SphMapForCartBasis
    shell_pairs: ShellPairs

    # Device cached objects
    dbasis: Any
    dsp: Any
    pt: Any

    # Optional cached rigorous Schwarz bounds (host-side, aligned to shell_pairs).
    Q_np: np.ndarray | None = None

    @classmethod
    def from_mol(
        cls,
        mol: Any,
        *,
        expand_contractions: bool = True,
        stream=None,
        threads: int = 256,
        max_l: int | None = None,
    ) -> "CuERISphERIEngine":
        if not has_cuda_ext():
            raise RuntimeError("cuERI CUDA extension not available; build via `python -m asuka.cueri.build_cuda_ext`")

        ao_basis, sph_map = pack_cart_shells_from_mol_with_sph_map(mol, expand_contractions=bool(expand_contractions))
        shell_l = np.asarray(ao_basis.shell_l, dtype=np.int32).ravel()
        lmax_basis = int(shell_l.max()) if int(shell_l.size) else 0

        lmax_allowed = int(CUDA_MAX_L if max_l is None else max_l)
        if lmax_basis > lmax_allowed:
            raise NotImplementedError(f"basis has shells with l={lmax_basis}, but CUDA backend supports l<={lmax_allowed}")

        sp = build_shell_pairs_l_order(ao_basis)
        dbasis = to_device_basis_ss(ao_basis)
        dsp = to_device_shell_pairs(sp)
        pt = build_pair_tables_ss_device(dbasis, dsp, stream=stream, threads=threads)
        return cls(mol=mol, ao_basis=ao_basis, sph_map=sph_map, shell_pairs=sp, dbasis=dbasis, dsp=dsp, pt=pt)

    @property
    def nao_sph(self) -> int:
        return int(self.sph_map.nao_sph)

    def int2e(
        self,
        *,
        aosym: Literal["s8", "s4"] = "s8",
        eps_ao: float = 0.0,
        tasks: TaskList | None = None,
        max_tiles_bytes: int = 256 << 20,
        stream=None,
        threads: int = 256,
        mode: str = "auto",
        work_small_max: int = 512,
        work_large_min: int = 200_000,
        blocks_per_task: int = 8,
        boys: str = "ref",
    ):
        """Compute (mu nu|lambda sigma) in the spherical AO basis."""

        import cupy as cp

        aosym = str(aosym).lower().strip()
        if aosym not in ("s8", "s4"):
            raise ValueError("aosym must be one of {'s8','s4'}")

        shell_l = np.asarray(self.ao_basis.shell_l, dtype=np.int32).ravel()
        nsp = int(np.asarray(self.shell_pairs.sp_A, dtype=np.int32).shape[0])

        if tasks is None:
            eps_ao_f = float(eps_ao)
            if eps_ao_f > 0.0:
                if self.Q_np is None:
                    self.Q_np = self._compute_schwarz_Q_np(
                        max_tiles_bytes=max_tiles_bytes,
                        stream=stream,
                        threads=threads,
                        mode=mode,
                        work_small_max=work_small_max,
                        work_large_min=work_large_min,
                        blocks_per_task=blocks_per_task,
                        boys=boys,
                    )
                tasks = build_tasks_screened_sorted_q(self.Q_np, eps_ao_f)
            else:
                tasks = _build_full_tasks_lower_triangle(nsp)

        nao_sph = int(self.sph_map.nao_sph)
        nao_pair = int(npair(nao_sph))

        if aosym == "s8":
            out_len = int(npair(nao_pair))
            out = cp.zeros((out_len,), dtype=cp.float64)
        else:
            out = cp.zeros((nao_pair, nao_pair), dtype=cp.float64)

        # Pre-upload spherical AO shell offsets once; reused across batches.
        shell_ao_start_sph = cp.asarray(self.sph_map.shell_ao_start_sph, dtype=cp.int32)

        for batch in iter_tile_batches_spd(
            tasks,
            shell_l=shell_l,
            shell_pairs=self.shell_pairs,
            dbasis=self.dbasis,
            dsp=self.dsp,
            pt=self.pt,
            stream=stream,
            threads=threads,
            mode=mode,
            work_small_max=work_small_max,
            work_large_min=work_large_min,
            blocks_per_task=blocks_per_task,
            max_tile_bytes=int(max_tiles_bytes),
            boys=boys,
        ):
            if int(batch.task_spAB.shape[0]) == 0:
                continue

            la, lb, lc, ld = decode_eri_class_id(int(batch.kernel_class_id))

            nAB_cart = int(ncart(la)) * int(ncart(lb))
            nCD_cart = int(ncart(lc)) * int(ncart(ld))
            if tuple(batch.tiles.shape[1:]) != (nAB_cart, nCD_cart):
                raise RuntimeError(
                    f"internal error: unexpected tile shape {tuple(batch.tiles.shape)} for (la,lb,lc,ld)=({la},{lb},{lc},{ld})"
                )

            tile_sph = cart2sph_eri_tiles_device(
                batch.tiles,
                la=la,
                lb=lb,
                lc=lc,
                ld=ld,
                stream=stream,
                threads=threads,
            )

            nA_sph = 2 * la + 1
            nB_sph = 2 * lb + 1
            nC_sph = 2 * lc + 1
            nD_sph = 2 * ld + 1

            batch_tasks = TaskList(task_spAB=batch.task_spAB, task_spCD=batch.task_spCD)

            if aosym == "s8":
                scatter_eri_tiles_sph_s8_inplace_device(
                    batch_tasks,
                    self.dsp,
                    shell_ao_start_sph=shell_ao_start_sph,
                    nao_sph=nao_sph,
                    nA=nA_sph,
                    nB=nB_sph,
                    nC=nC_sph,
                    nD=nD_sph,
                    tile_vals=tile_sph,
                    out_s8=out,
                    stream=stream,
                    threads=threads,
                )
            else:
                scatter_eri_tiles_sph_s4_inplace_device(
                    batch_tasks,
                    self.dsp,
                    shell_ao_start_sph=shell_ao_start_sph,
                    nao_sph=nao_sph,
                    nA=nA_sph,
                    nB=nB_sph,
                    nC=nC_sph,
                    nD=nD_sph,
                    tile_vals=tile_sph,
                    out_s4=out,
                    stream=stream,
                    threads=threads,
                )

        return out

    def _compute_schwarz_Q_np(
        self,
        *,
        max_tiles_bytes: int,
        stream,
        threads: int,
        mode: str,
        work_small_max: int,
        work_large_min: int,
        blocks_per_task: int,
        boys: str,
    ) -> np.ndarray:
        """Compute rigorous Schwarz bounds Q_AB for each shell pair AB."""

        import cupy as cp

        nsp = int(np.asarray(self.shell_pairs.sp_A, dtype=np.int32).shape[0])
        if nsp == 0:
            return np.zeros((0,), dtype=np.float64)

        idx = np.arange(nsp, dtype=np.int32)
        diag_tasks = TaskList(task_spAB=idx, task_spCD=idx)

        Q_dev = cp.empty((nsp,), dtype=cp.float64)
        shell_l = np.asarray(self.ao_basis.shell_l, dtype=np.int32).ravel()

        for batch in iter_tile_batches_spd(
            diag_tasks,
            shell_l=shell_l,
            shell_pairs=self.shell_pairs,
            dbasis=self.dbasis,
            dsp=self.dsp,
            pt=self.pt,
            stream=stream,
            threads=threads,
            mode=mode,
            work_small_max=work_small_max,
            work_large_min=work_large_min,
            blocks_per_task=blocks_per_task,
            max_tile_bytes=int(max_tiles_bytes),
            boys=boys,
        ):
            if int(batch.task_spAB.shape[0]) == 0:
                continue
            tile = batch.tiles
            diag = cp.diagonal(tile, axis1=1, axis2=2)
            q = cp.sqrt(cp.maximum(cp.max(diag, axis=1), 0.0))
            Q_dev[cp.asarray(batch.task_spAB, dtype=cp.int32)] = q

        return cp.asnumpy(Q_dev)


def int2e_sph_device(
    mol: Any,
    *,
    aosym: Literal["s8", "s4"] = "s8",
    eps_ao: float = 0.0,
    max_tiles_bytes: int = 256 << 20,
    expand_contractions: bool = True,
    stream=None,
    threads: int = 256,
    mode: str = "auto",
    work_small_max: int = 512,
    work_large_min: int = 200_000,
    blocks_per_task: int = 8,
    boys: str = "ref",
):
    """Convenience wrapper: build a CuERISphERIEngine and compute spherical int2e."""

    eng = CuERISphERIEngine.from_mol(
        mol,
        expand_contractions=expand_contractions,
        stream=stream,
        threads=threads,
    )
    return eng.int2e(
        aosym=aosym,
        eps_ao=eps_ao,
        max_tiles_bytes=max_tiles_bytes,
        stream=stream,
        threads=threads,
        mode=mode,
        work_small_max=work_small_max,
        work_large_min=work_large_min,
        blocks_per_task=blocks_per_task,
        boys=boys,
    )


__all__ = ["CuERISphERIEngine", "int2e_sph_device"]
