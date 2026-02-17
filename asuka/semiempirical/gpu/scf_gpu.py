"""CuPy/CUDA AM1 SCF driver with fused NDDO Fock kernels."""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from asuka.nddo_core import (
    build_ao_offsets,
    build_core_hamiltonian_from_pair_terms,
    build_onecenter_eris,
    build_pair_ri_payload,
    build_pair_list,
    compute_all_multipole_params,
    core_core_repulsion_from_gamma_ss,
    nao_for_Z,
    valence_electrons,
)
from asuka.semiempirical.overlap import build_overlap_matrix
from asuka.semiempirical.params import MethodParams

from .kernels import (
    build_pair_buckets,
    build_wblocks_from_ri_device,
    ensure_kernel_source_available,
    get_fock_kernels,
    pack_onecenter_eris,
)
from .runtime import _import_cupy

_WBLOCK_BYTES_PER_PAIR = 256 * 8  # packed W block [256] float64
_AUTO_WBLOCK_MAX_BYTES = 512 * 1024 * 1024


def _require_cuda_runtime():
    cp = _import_cupy()
    if cp is None:
        raise RuntimeError(
            "CuPy is required for device='cuda'. Install ASUKA with CUDA extras "
            "(for example: pip install -e '.[cuda]')."
        )
    try:
        ndev = int(cp.cuda.runtime.getDeviceCount())
    except Exception as exc:
        raise RuntimeError("Unable to query CUDA devices via CuPy runtime") from exc
    if ndev < 1:
        raise RuntimeError("No CUDA device is visible to CuPy (device='cuda' unavailable)")
    return cp


def _resolve_cuda_fock_mode(fock_mode: str, npairs: int) -> str:
    """Resolve CUDA fock mode, selecting W mode when memory budget allows."""
    mode = str(fock_mode).strip().lower()
    if mode in ("ri", "w"):
        return mode
    if mode != "auto":
        raise ValueError("fock_mode must be 'ri', 'w', or 'auto'")
    wbytes = int(npairs) * _WBLOCK_BYTES_PER_PAIR
    return "w" if wbytes <= _AUTO_WBLOCK_MAX_BYTES else "ri"


def _build_density(cp, C, nocc: int):
    Cocc = C[:, :nocc]
    return 2.0 * (Cocc @ Cocc.T)


def _build_fock_gpu(
    cp,
    kernels,
    H_d,
    P_d,
    ao_off_d,
    nao_atom_d,
    onecenter_d,
    pair_i_d,
    pair_j_d,
    Wblocks_d,
    ri_d,
    ta_d,
    tb_d,
    ri_bucket_dev,
    fock_mode: str,
    nat: int,
    nao_total: int,
    npairs: int,
):
    F_d = H_d.copy()

    onecenter_kernel = kernels["onecenter"]
    twocenter_kernel = kernels["twocenter"]
    twocenter_ri_kernel = kernels["twocenter_ri"]
    twocenter_ri_11 = kernels["twocenter_ri_11"]
    twocenter_ri_14 = kernels["twocenter_ri_14"]
    twocenter_ri_41 = kernels["twocenter_ri_41"]
    twocenter_ri_44 = kernels["twocenter_ri_44"]

    onecenter_kernel(
        (nat,),
        (32,),
        (
            ao_off_d,
            nao_atom_d,
            onecenter_d,
            P_d,
            F_d,
            np.int32(nao_total),
            np.int32(nat),
        ),
    )

    if fock_mode == "ri":
        if npairs == 0:
            return F_d
        launched = False
        for key, kernel in (
            ("11", twocenter_ri_11),
            ("14", twocenter_ri_14),
            ("41", twocenter_ri_41),
            ("44", twocenter_ri_44),
        ):
            b = ri_bucket_dev[key]
            n = int(b["count"])
            if n == 0:
                continue
            launched = True
            kernel(
                (n,),
                (64,),
                (
                    b["pair_i"],
                    b["pair_j"],
                    ao_off_d,
                    b["ri"],
                    b["ta"],
                    b["tb"],
                    P_d,
                    F_d,
                    np.int32(nao_total),
                    np.int32(n),
                ),
            )
        if not launched:
            twocenter_ri_kernel(
                (npairs,),
                (64,),
                (
                    pair_i_d,
                    pair_j_d,
                    ao_off_d,
                    nao_atom_d,
                    ri_d,
                    ta_d,
                    tb_d,
                    P_d,
                    F_d,
                    np.int32(nao_total),
                    np.int32(npairs),
                ),
            )
    else:
        if npairs == 0:
            return F_d
        twocenter_kernel(
            (npairs,),
            (64,),
            (
                pair_i_d,
                pair_j_d,
                ao_off_d,
                nao_atom_d,
                Wblocks_d,
                P_d,
                F_d,
                np.int32(nao_total),
                np.int32(npairs),
            ),
        )

    return F_d


def am1_scf_cuda(
    atomic_numbers: Sequence[int],
    coords_bohr: np.ndarray,
    params: MethodParams,
    charge: int = 0,
    max_iter: int = 200,
    conv_tol: float = 1e-8,
    comm_tol: float = 1e-6,
    diis: bool = True,
    diis_start: int = 2,
    diis_size: int = 8,
    fock_mode: str = "ri",
):
    """Run AM1 SCF with CuPy-managed CUDA kernels for Fock assembly.

    Parameters
    ----------
    comm_tol
        Commutator convergence threshold on ``max(abs(FP-PF))``.
    fock_mode
        Two-center build mode: ``\"ri\"`` reconstructs pair tensors from rotational
        invariants in-kernel; ``\"w\"`` materializes full packed W blocks once on
        device from RI payload; ``\"auto\"`` selects ``\"w\"`` unless packed-W
        memory exceeds a conservative threshold.
    """
    cp = _require_cuda_runtime()
    ensure_kernel_source_available()
    fock_mode = str(fock_mode).strip().lower()
    if fock_mode not in ("ri", "w", "auto"):
        raise ValueError("fock_mode must be 'ri', 'w', or 'auto'")

    atomic_numbers = list(atomic_numbers)
    coords_bohr = np.asarray(coords_bohr, dtype=float)
    elem_params = params.elements
    nat = len(atomic_numbers)

    for Z in atomic_numbers:
        if Z not in elem_params:
            raise ValueError(f"No parameters for element Z={Z}")

    offsets = build_ao_offsets(atomic_numbers)
    nao_total = int(offsets[-1])
    nao_atom = np.asarray([nao_for_Z(Z) for Z in atomic_numbers], dtype=np.uint8)

    n_elec = sum(valence_electrons(Z) for Z in atomic_numbers) - charge
    if n_elec % 2 != 0:
        raise ValueError(f"Odd number of electrons ({n_elec}), open-shell not supported")
    nocc = n_elec // 2

    pair_i, pair_j, _, pair_r = build_pair_list(coords_bohr)
    npairs = int(len(pair_i))
    fock_mode_run = _resolve_cuda_fock_mode(fock_mode, npairs)

    mp_params = compute_all_multipole_params(elem_params)
    S = build_overlap_matrix(atomic_numbers, coords_bohr, elem_params)
    ri_pack, ta_pack, tb_pack, vaa_pack, vbb_pack, gamma_ss = build_pair_ri_payload(
        atomic_numbers=atomic_numbers,
        coords_bohr=coords_bohr,
        pair_i=pair_i,
        pair_j=pair_j,
        mp_params=mp_params,
    )

    H = build_core_hamiltonian_from_pair_terms(
        atomic_numbers=atomic_numbers,
        S=S,
        pair_i=pair_i,
        pair_j=pair_j,
        vaa_pack=vaa_pack,
        vbb_pack=vbb_pack,
        elem_params=elem_params,
    )

    E_core = core_core_repulsion_from_gamma_ss(
        atomic_numbers=atomic_numbers,
        pair_i=pair_i,
        pair_j=pair_j,
        pair_r=pair_r,
        gamma_ss=gamma_ss,
        elem_params=elem_params,
    )

    onecenter_eris: List[np.ndarray] = []
    for Z in atomic_numbers:
        ep = elem_params[Z]
        nao = nao_for_Z(Z)
        onecenter_eris.append(build_onecenter_eris(nao, ep.gss, ep.gsp, ep.gpp, ep.gp2, ep.hsp))

    # Host packing for fixed-size CUDA kernels.
    onecenter_pack = pack_onecenter_eris(atomic_numbers, onecenter_eris)
    ri_bucket_dev = None

    # Device mirrors
    ao_off_d = cp.asarray(offsets.astype(np.int32))
    nao_atom_d = cp.asarray(nao_atom)
    pair_i_d = cp.asarray(pair_i.astype(np.int32))
    pair_j_d = cp.asarray(pair_j.astype(np.int32))
    onecenter_d = cp.asarray(onecenter_pack)
    ri_d = cp.asarray(ri_pack)
    ta_d = cp.asarray(ta_pack)
    tb_d = cp.asarray(tb_pack)
    H_d = cp.asarray(H)
    kernels = get_fock_kernels()

    if fock_mode_run == "w":
        Wblocks_d = build_wblocks_from_ri_device(
            cp=cp,
            kernels=kernels,
            pair_i_d=pair_i_d,
            pair_j_d=pair_j_d,
            ao_off_d=ao_off_d,
            nao_atom_d=nao_atom_d,
            ri_d=ri_d,
            ta_d=ta_d,
            tb_d=tb_d,
            npairs=npairs,
        )
    else:
        Wblocks_d = cp.empty((0,), dtype=cp.float64)

    if fock_mode_run == "ri":
        buckets = build_pair_buckets(atomic_numbers, pair_i, pair_j)
        ri_bucket_dev = {}
        for key in ("11", "14", "41", "44"):
            idx = buckets[key]
            if len(idx) == 0:
                ri_bucket_dev[key] = {
                    "count": 0,
                    "pair_i": cp.empty((0,), dtype=cp.int32),
                    "pair_j": cp.empty((0,), dtype=cp.int32),
                    "ri": cp.empty((0, 22), dtype=cp.float64),
                    "ta": cp.empty((0, 16), dtype=cp.float64),
                    "tb": cp.empty((0, 16), dtype=cp.float64),
                }
            else:
                ri_bucket_dev[key] = {
                    "count": int(len(idx)),
                    "pair_i": cp.asarray(pair_i[idx].astype(np.int32)),
                    "pair_j": cp.asarray(pair_j[idx].astype(np.int32)),
                    "ri": cp.asarray(ri_pack[idx, :]),
                    "ta": cp.asarray(ta_pack[idx, :]),
                    "tb": cp.asarray(tb_pack[idx, :]),
                }
    else:
        ri_bucket_dev = {
            "11": {"count": 0},
            "14": {"count": 0},
            "41": {"count": 0},
            "44": {"count": 0},
        }

    # Initial guess from H
    eps_d, C_d = cp.linalg.eigh(H_d)
    P_d = _build_density(cp, C_d, nocc)

    E_old = 0.0
    converged = False
    diis_F_list = []
    diis_err_host_list = []

    for it in range(max_iter):
        F_d = _build_fock_gpu(
            cp,
            kernels,
            H_d,
            P_d,
            ao_off_d,
            nao_atom_d,
            onecenter_d,
            pair_i_d,
            pair_j_d,
            Wblocks_d,
            ri_d,
            ta_d,
            tb_d,
            ri_bucket_dev,
            fock_mode_run,
            nat,
            nao_total,
            npairs,
        )
        err_d = F_d @ P_d - P_d @ F_d
        err_max = float(cp.max(cp.abs(err_d)).item())

        E_el = float((0.5 * cp.sum(P_d * (H_d + F_d))).item())
        E_total = E_el + E_core

        dE = abs(E_total - E_old)
        if it > 0 and dE < conv_tol and err_max < comm_tol:
            converged = True
            break
        E_old = E_total

        if diis and it >= diis_start:
            diis_F_list.append(F_d.copy())
            diis_err_host_list.append(cp.asnumpy(err_d).ravel())

            if len(diis_F_list) > diis_size:
                diis_F_list.pop(0)
                diis_err_host_list.pop(0)

            if len(diis_F_list) >= 2:
                m = len(diis_F_list)
                Bmat = np.empty((m + 1, m + 1), dtype=float)
                for i in range(m):
                    for j in range(m):
                        Bmat[i, j] = float(np.dot(diis_err_host_list[i], diis_err_host_list[j]))
                Bmat[:m, m] = -1.0
                Bmat[m, :m] = -1.0
                Bmat[m, m] = 0.0
                rhs = np.zeros(m + 1, dtype=float)
                rhs[m] = -1.0
                try:
                    coeff = np.linalg.solve(Bmat, rhs)[:m]
                    F_mix = cp.zeros_like(F_d)
                    for c, Fi in zip(coeff, diis_F_list):
                        F_mix += float(c) * Fi
                    F_d = F_mix
                except np.linalg.LinAlgError:
                    pass

        eps_d, C_d = cp.linalg.eigh(F_d)
        P_d = _build_density(cp, C_d, nocc)

    from asuka.semiempirical.scf import SCFResult

    eps = cp.asnumpy(eps_d)
    C = cp.asnumpy(C_d)
    P = cp.asnumpy(P_d)
    F = cp.asnumpy(F_d)

    return SCFResult(
        converged=converged,
        n_iter=it + 1,
        energy_electronic=E_el,
        energy_core=E_core,
        energy_total=E_total,
        eps=eps,
        C=C,
        P=P,
        F=F,
        H=H,
        nocc=nocc,
    )
