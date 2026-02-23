from __future__ import annotations

"""ASUKA-native driver helpers for MRCI.

This module mirrors :mod:`asuka.caspt2.driver_asuka` by providing end-to-end
MRCISD entry points starting from ASUKA frontend SCF outputs and ASUKA CASCI/CASSCF
results.

Phase-1 scope
-------------
- Uncontracted MRCISD only (``method='mrcisd'``).
- DF backend only via `scf_out.df_B` (AO DF factors from the frontend).
- Frozen-core folding is done in AO basis using DF J/K from `asuka.hf.df_jk`,
  matching the standard closed-shell `VHF = J - 0.5 K` convention.
"""

from typing import Any, Literal, Sequence

import numpy as np

from asuka.integrals.df_integrals import DFMOIntegrals, DeviceDFMOIntegrals
from asuka.mcscf.state_average import ci_as_list
from asuka.mrci.common import compute_cas_reference_energy_df, embed_ci_with_docc_prefix
from asuka.mrci.mrcisd import MRCISDResult, MRCISDResultMulti, mrcisd_kernel, mrcisd_plus_q
from asuka.mrci.result import MRCIResult, MRCIStatesResult, MRCISOCResult
from asuka.soc.si import SOCIntegrals, SpinFreeState, soc_state_interaction


Method = Literal["mrcisd"]
DFIntegralsBackend = Literal["df_B"]


def _nelecas_total(nelecas: int | tuple[int, int]) -> int:
    if isinstance(nelecas, (int, np.integer)):
        return int(nelecas)
    if isinstance(nelecas, (tuple, list)) and len(nelecas) == 2:
        return int(nelecas[0]) + int(nelecas[1])
    raise ValueError("nelecas must be an int or a length-2 tuple/list")


def _resolve_scf_out_from_ref(ref: Any, *, scf_out: Any | None) -> Any:
    scf_out_use = scf_out
    if scf_out_use is None:
        scf_out_use = getattr(ref, "scf_out", None)
    if scf_out_use is None and hasattr(ref, "casci"):
        scf_out_use = getattr(getattr(ref, "casci"), "scf_out", None)
    if scf_out_use is None:
        raise ValueError("scf_out is required (missing on ref; pass scf_out explicitly)")
    return scf_out_use


def _infer_states_from_ref(ref: Any, *, states: Sequence[int] | None) -> list[int]:
    if states is None:
        ci = getattr(ref, "ci")
        if isinstance(ci, (list, tuple)):
            return list(range(len(ci)))
        return [0]
    out = [int(s) for s in states]
    if not out:
        raise ValueError("states must be non-empty when provided")
    if len(set(out)) != len(out):
        raise ValueError("states must not contain duplicates")
    if any(s < 0 for s in out):
        raise ValueError("states must be non-negative")
    ci = getattr(ref, "ci")
    if isinstance(ci, (list, tuple)):
        n = int(len(ci))
        if any(s >= n for s in out):
            raise ValueError("state index out of range for ref.ci")
    else:
        if any(s != 0 for s in out):
            raise ValueError("state != 0 but ref.ci is not a list/tuple")
    return out


def _get_ref_twos(ref: Any, scf_out: Any, *, twos: int | None) -> int:
    if twos is not None:
        return int(twos)
    val = getattr(getattr(scf_out, "mol", None), "spin", None)
    if val is None:
        val = getattr(getattr(ref, "mol", None), "spin", 0)
    return int(val)


def _maybe_asnumpy(x: Any) -> np.ndarray:
    try:
        import cupy as cp  # noqa: PLC0415

        if isinstance(x, cp.ndarray):  # type: ignore[attr-defined]
            return np.asarray(cp.asnumpy(x), dtype=np.float64)
    except Exception:
        pass
    return np.asarray(x, dtype=np.float64)


def _frozen_core_h1e_ecore_df(
    *,
    scf_out: Any,
    mo_core: Any,
    mo_corr: Any,
) -> tuple[np.ndarray, float]:
    """Return (h1e_corr, ecore) using AO DF J/K for the frozen-core density.

    This follows the standard closed-shell folding convention:
      - dm_core = 2 * C_core C_core^T (closed-shell core)
      - vhf_core = J(dm_core) - 0.5 K(dm_core)
      - h_eff_ao = hcore + vhf_core
      - h1e_corr = C_corr^T h_eff_ao C_corr
      - ecore = E_nuc + Tr(dm_core hcore) + 0.5 Tr(dm_core vhf_core)
    """

    from asuka.hf import df_jk  # noqa: PLC0415

    B = getattr(scf_out, "df_B", None)
    if B is None:
        raise ValueError("scf_out.df_B is required for DF MRCI driver")
    hcore = getattr(getattr(scf_out, "int1e", None), "hcore", None)
    if hcore is None:
        raise ValueError("scf_out.int1e.hcore is missing")

    e_nuc = float(getattr(getattr(scf_out, "mol", None), "energy_nuc")())

    # Choose xp based on inputs.
    try:
        import cupy as cp  # noqa: PLC0415

        use_gpu = isinstance(B, cp.ndarray) or isinstance(mo_core, cp.ndarray) or isinstance(mo_corr, cp.ndarray)  # type: ignore[attr-defined]
        xp = cp if use_gpu else np
    except Exception:
        xp = np

    B = xp.asarray(B, dtype=xp.float64)
    hcore = xp.asarray(hcore, dtype=xp.float64)
    c_core = xp.asarray(mo_core, dtype=xp.float64)
    c_corr = xp.asarray(mo_corr, dtype=xp.float64)

    ncore = int(getattr(c_core, "shape", (0, 0))[1])
    if ncore == 0:
        h_eff_ao = hcore
        h1e_corr = c_corr.T @ h_eff_ao @ c_corr
        return np.asarray(_maybe_asnumpy(h1e_corr), dtype=np.float64, order="C"), float(e_nuc)

    dm_core = 2.0 * (c_core @ c_core.T)

    # DF J/K requires BQ layout.
    BQ = xp.transpose(B, (2, 0, 1))
    try:
        BQ = xp.ascontiguousarray(BQ)
    except Exception:
        pass

    j = df_jk.df_J_from_BQ_D(BQ, dm_core)
    k = df_jk.df_K_from_BQ_D(BQ, dm_core)
    vhf_core = j - 0.5 * k

    h_eff_ao = hcore + vhf_core
    h1e_corr = c_corr.T @ h_eff_ao @ c_corr

    e1 = xp.einsum("ij,ji->", dm_core, hcore, optimize=True)
    e2 = 0.5 * xp.einsum("ij,ji->", dm_core, vhf_core, optimize=True)

    if xp is np:
        ecore = float(e_nuc + float(e1) + float(e2))
        return np.asarray(h1e_corr, dtype=np.float64, order="C"), ecore

    ecore = float(e_nuc + float(np.asarray(xp.asnumpy(e1))) + float(np.asarray(xp.asnumpy(e2))))
    return np.asarray(_maybe_asnumpy(h1e_corr), dtype=np.float64, order="C"), ecore


def _dfmo_integrals_from_df_B(
    B_ao: Any,
    C: Any,
    *,
    device: Literal["cpu", "cuda"],
) -> DFMOIntegrals | DeviceDFMOIntegrals:
    """Build DF MO integrals from AO DF factors and MO coefficients."""

    B_ao = np.asarray(B_ao) if device == "cpu" else B_ao
    C = np.asarray(C) if device == "cpu" else C

    if device == "cpu":
        B_ao = np.asarray(B_ao, dtype=np.float64, order="C")
        C = np.asarray(C, dtype=np.float64, order="C")
        if B_ao.ndim != 3:
            raise ValueError("B_ao must have shape (nao, nao, naux)")
        nao0, nao1, naux = map(int, B_ao.shape)
        if nao0 != nao1:
            raise ValueError("B_ao must have shape (nao, nao, naux)")
        if C.ndim != 2 or int(C.shape[0]) != nao0:
            raise ValueError("C has incompatible shape for B_ao")
        norb = int(C.shape[1])
        if norb <= 0:
            raise ValueError("C must have norb > 0")

        tmp = np.tensordot(B_ao, C, axes=([1], [0]))  # (nao, naux, norb)
        tmp = np.transpose(tmp, (0, 2, 1))  # (nao, norb, naux)
        l_pqQ = np.tensordot(C.T, tmp, axes=([1], [0]))  # (norb, norb, naux)
        l_full = np.asarray(l_pqQ.reshape(norb * norb, naux), dtype=np.float64, order="C")
        pair_norm = np.linalg.norm(l_full, axis=1)
        j_ps = np.einsum("pql,qsl->ps", l_pqQ, l_pqQ, optimize=True)
        return DFMOIntegrals(
            norb=int(norb),
            l_full=l_full,
            j_ps=np.asarray(j_ps, dtype=np.float64, order="C"),
            pair_norm=np.asarray(pair_norm, dtype=np.float64, order="C"),
        )

    # CUDA: build GPU-resident arrays (CuPy) if available.
    try:
        import cupy as cp  # noqa: PLC0415
    except Exception as e:  # pragma: no cover
        raise RuntimeError("device='cuda' requires CuPy") from e

    B_ao_d = cp.asarray(B_ao, dtype=cp.float64)
    C_d = cp.asarray(C, dtype=cp.float64)
    if B_ao_d.ndim != 3:
        raise ValueError("B_ao must have shape (nao, nao, naux)")
    nao0, nao1, naux = map(int, B_ao_d.shape)
    if nao0 != nao1:
        raise ValueError("B_ao must have shape (nao, nao, naux)")
    if C_d.ndim != 2 or int(C_d.shape[0]) != nao0:
        raise ValueError("C has incompatible shape for B_ao")
    norb = int(C_d.shape[1])
    if norb <= 0:
        raise ValueError("C must have norb > 0")

    tmp = cp.tensordot(B_ao_d, C_d, axes=([1], [0]))  # (nao, naux, norb)
    tmp = cp.transpose(tmp, (0, 2, 1))  # (nao, norb, naux)
    l_pqQ = cp.tensordot(C_d.T, tmp, axes=([1], [0]))  # (norb, norb, naux)

    l_pqQ = cp.ascontiguousarray(l_pqQ, dtype=cp.float64)
    l_full = l_pqQ.reshape(norb * norb, naux)
    pair_norm = cp.linalg.norm(l_full, axis=1)
    j_ps = cp.einsum("pql,qsl->ps", l_pqQ, l_pqQ, optimize=True)

    return DeviceDFMOIntegrals(
        norb=int(norb),
        l_full=cp.ascontiguousarray(l_full, dtype=cp.float64),
        j_ps=cp.ascontiguousarray(j_ps, dtype=cp.float64),
        pair_norm=cp.ascontiguousarray(pair_norm, dtype=cp.float64),
        eri_mat=None,
    )


def mrci_states_from_ref(
    ref: Any,
    *,
    scf_out: Any | None = None,
    method: Method = "mrcisd",
    states: Sequence[int] | None = None,
    nroots: int | None = None,
    n_virt: int | None = None,
    twos: int | None = None,
    max_virt_e: int = 2,
    correlate_inactive: int = 0,
    orbsym: Sequence[int] | None = None,
    wfnsym: int | None = None,
    integrals_backend: DFIntegralsBackend = "df_B",
    # --- uncontracted MRCISD knobs ---
    hop_backend: str | None = None,
    tol: float = 1e-10,
    max_cycle: int = 400,
    max_space: int = 30,
    max_memory_mb: float = 4000.0,
    contract_nthreads: int = 1,
    contract_blas_nthreads: int | None = None,
    precompute_epq: bool = True,
    return_integrals: bool = False,
) -> MRCIStatesResult:
    """Run multi-root MRCISD on top of an ASUKA CASCI/CASSCF result."""

    method_s = str(method).strip().lower()
    if method_s != "mrcisd":
        raise NotImplementedError("ASUKA-native MRCI driver currently supports only method='mrcisd'")

    integrals_backend_s = str(integrals_backend).strip().lower()
    if integrals_backend_s != "df_b":
        raise ValueError("integrals_backend must be 'df_B' for ASUKA-native MRCI")

    scf_out_use = _resolve_scf_out_from_ref(ref, scf_out=scf_out)

    mo = getattr(ref, "mo_coeff", None)
    if mo is None:
        raise ValueError("ref.mo_coeff is required")
    mo = _maybe_asnumpy(mo)

    ncore_ref = int(getattr(ref, "ncore", 0))
    n_act_ref = int(getattr(ref, "ncas", 0))
    if n_act_ref <= 0:
        raise ValueError("ref.ncas must be positive")

    correlate_inactive_i = int(correlate_inactive)
    if correlate_inactive_i < 0 or correlate_inactive_i > ncore_ref:
        raise ValueError("correlate_inactive must satisfy 0 <= correlate_inactive <= ref.ncore")
    ncore_frozen = ncore_ref - correlate_inactive_i
    n_act_int = n_act_ref + correlate_inactive_i

    nmo = int(mo.shape[1])
    nvirt_all = nmo - ncore_ref - n_act_ref
    if nvirt_all < 0:
        raise RuntimeError("invalid orbital partition: ncore+ncas > nmo")

    if n_virt is None:
        n_virt = nvirt_all
    n_virt = int(n_virt)
    if n_virt < 0 or n_virt > nvirt_all:
        raise ValueError("n_virt must satisfy 0 <= n_virt <= (nmo-ncore-ncas)")

    states_list = _infer_states_from_ref(ref, states=states)
    nroots_i = len(states_list) if nroots is None else int(nroots)
    if nroots_i != len(states_list):
        raise ValueError(
            f"mrci_states_from_ref currently requires nroots == len(states); got nroots={nroots_i} and "
            f"len(states)={len(states_list)}"
        )

    # Orbital partitions in the reference MO basis.
    mo_core_frozen = mo[:, :ncore_frozen]
    mo_core_corr = mo[:, ncore_frozen:ncore_ref]
    mo_act = mo[:, ncore_ref : ncore_ref + n_act_ref]
    mo_virt = mo[:, ncore_ref + n_act_ref : ncore_ref + n_act_ref + n_virt]
    mo_corr = np.hstack([mo_core_corr, mo_act, mo_virt])

    # Frozen-core 1e integrals in the correlated space.
    h1e_corr, ecore = _frozen_core_h1e_ecore_df(
        scf_out=scf_out_use,
        mo_core=mo_core_frozen,
        mo_corr=mo_corr,
    )

    # DF integrals in the correlated MO basis.
    B_ao = getattr(scf_out_use, "df_B", None)
    if B_ao is None:
        raise ValueError("scf_out.df_B is required (DF factors missing)")

    want_cuda_ints = str(hop_backend).strip().lower() == "cuda"
    if want_cuda_ints:
        eri_payload = _dfmo_integrals_from_df_B(B_ao, mo_corr, device="cuda")
        if not isinstance(eri_payload, DeviceDFMOIntegrals):
            raise RuntimeError("internal error: expected DeviceDFMOIntegrals for device='cuda'")
        l_full = eri_payload.l_full
        df_ints_ret = eri_payload if bool(return_integrals) else None
    else:
        eri_payload = _dfmo_integrals_from_df_B(_maybe_asnumpy(B_ao), mo_corr, device="cpu")
        if not isinstance(eri_payload, DFMOIntegrals):
            raise RuntimeError("internal error: expected DFMOIntegrals for device='cpu'")
        l_full = eri_payload.l_full
        df_ints_ret = None

    # Active-space electron count and spin.
    nelec_act = _nelecas_total(getattr(ref, "nelecas"))
    nelec_corr = int(nelec_act) + 2 * correlate_inactive_i
    twos_i = _get_ref_twos(ref, scf_out_use, twos=twos)

    # CI vectors for requested reference states.
    ci_obj = getattr(ref, "ci")
    if isinstance(ci_obj, (list, tuple)):
        nroots_ref = int(len(ci_obj))
    else:
        nroots_ref = int(getattr(ref, "nroots", 1))
    ci_all = ci_as_list(ci_obj, nroots=max(1, nroots_ref))
    ci_act_list = [np.asarray(ci_all[s], dtype=np.float64).ravel() for s in states_list]

    from asuka.cuguga.drt import build_drt  # noqa: PLC0415

    drt_act = build_drt(norb=int(n_act_ref), nelec=int(nelec_act), twos_target=int(twos_i))
    ncsf_act = int(drt_act.ncsf)
    bad = [int(states_list[i]) for i, c_act in enumerate(ci_act_list) if int(np.asarray(c_act).size) != ncsf_act]
    if bad:
        raise ValueError(f"reference CI vector length mismatch with active DRT ncsf={ncsf_act}; bad states={bad}")

    ci_cas_list = []
    for ci_act in ci_act_list:
        if correlate_inactive_i > 0:
            ci_cas_list.append(
                embed_ci_with_docc_prefix(
                    ci_act=ci_act,
                    n_docc=correlate_inactive_i,
                    n_act=n_act_ref,
                    nelec_act=nelec_act,
                    twos=twos_i,
                    orbsym_act=None,
                    orbsym_full=None,
                    wfnsym=wfnsym,
                )
            )
        else:
            ci_cas_list.append(np.asarray(ci_act, dtype=np.float64))

    # Reference energies in the frozen-core convention (one per reference state).
    e_ref_list: list[float] = []
    for ci_cas in ci_cas_list:
        e_ref_list.append(
            compute_cas_reference_energy_df(
                h1e_corr=h1e_corr,
                l_full=l_full,
                ecore=float(ecore),
                ci_cas=np.asarray(ci_cas, dtype=np.float64),
                n_act=n_act_int,
                nelec=nelec_corr,
                twos=twos_i,
                orbsym_act=None,
                wfnsym=wfnsym,
            )
        )

    nthreads = int(contract_nthreads)
    if nthreads < 1:
        nthreads = 1

    mrci = mrcisd_kernel(
        h1e=h1e_corr,
        eri=eri_payload,
        n_act=n_act_int,
        n_virt=n_virt,
        nelec=nelec_corr,
        twos=twos_i,
        ci_cas=ci_cas_list,
        nroots=nroots_i,
        ecore=float(ecore),
        orbsym_act=None,
        orbsym_corr=None,
        wfnsym=wfnsym,
        max_virt_e=max_virt_e,
        hop_backend=hop_backend,
        tol=tol,
        max_cycle=max_cycle,
        max_space=max_space,
        max_memory=max_memory_mb,
        contract_nthreads=nthreads,
        contract_blas_nthreads=contract_blas_nthreads,
        precompute_epq=precompute_epq,
    )
    if not isinstance(mrci, MRCISDResultMulti):
        raise RuntimeError("internal error: expected a multi-root MRCISD result")

    return MRCIStatesResult(
        method="mrcisd",
        states=list(states_list),
        nroots=int(nroots_i),
        e_ref=np.asarray(e_ref_list, dtype=np.float64),
        mrci=mrci,
        ecore=float(ecore),
        ncore=int(ncore_frozen),
        n_act=int(n_act_int),
        n_virt=int(n_virt),
        nelec=int(nelec_corr),
        twos=int(twos_i),
        df_integrals=df_ints_ret,
    )


def mrci_from_ref(
    ref: Any,
    *,
    scf_out: Any | None = None,
    method: Method = "mrcisd",
    state: int = 0,
    n_virt: int | None = None,
    twos: int | None = None,
    max_virt_e: int = 2,
    correlate_inactive: int = 0,
    wfnsym: int | None = None,
    integrals_backend: DFIntegralsBackend = "df_B",
    # --- uncontracted MRCISD knobs ---
    hop_backend: str | None = None,
    tol: float = 1e-10,
    max_cycle: int = 400,
    max_space: int = 30,
    max_memory_mb: float = 4000.0,
    contract_nthreads: int = 1,
    contract_blas_nthreads: int | None = None,
    precompute_epq: bool = True,
    # --- optional +Q ---
    plus_q: bool = False,
    plus_q_model: str = "fixed",
    plus_q_min_ref: float = 1e-8,
    # --- misc ---
    return_integrals: bool = False,
) -> MRCIResult:
    """Run single-root MRCISD on top of an ASUKA CASCI/CASSCF result."""

    method_s = str(method).strip().lower()
    if method_s != "mrcisd":
        raise NotImplementedError("ASUKA-native MRCI driver currently supports only method='mrcisd'")

    integrals_backend_s = str(integrals_backend).strip().lower()
    if integrals_backend_s != "df_b":
        raise ValueError("integrals_backend must be 'df_B' for ASUKA-native MRCI")

    scf_out_use = _resolve_scf_out_from_ref(ref, scf_out=scf_out)

    mo = getattr(ref, "mo_coeff", None)
    if mo is None:
        raise ValueError("ref.mo_coeff is required")
    mo = _maybe_asnumpy(mo)

    ncore_ref = int(getattr(ref, "ncore", 0))
    n_act_ref = int(getattr(ref, "ncas", 0))
    if n_act_ref <= 0:
        raise ValueError("ref.ncas must be positive")

    correlate_inactive_i = int(correlate_inactive)
    if correlate_inactive_i < 0 or correlate_inactive_i > ncore_ref:
        raise ValueError("correlate_inactive must satisfy 0 <= correlate_inactive <= ref.ncore")
    ncore_frozen = ncore_ref - correlate_inactive_i
    n_act_int = n_act_ref + correlate_inactive_i

    nmo = int(mo.shape[1])
    nvirt_all = nmo - ncore_ref - n_act_ref
    if nvirt_all < 0:
        raise RuntimeError("invalid orbital partition: ncore+ncas > nmo")

    if n_virt is None:
        n_virt = nvirt_all
    n_virt = int(n_virt)
    if n_virt < 0 or n_virt > nvirt_all:
        raise ValueError("n_virt must satisfy 0 <= n_virt <= (nmo-ncore-ncas)")

    state_i = int(state)
    if state_i < 0:
        raise ValueError("state must be >= 0")

    # Orbital partitions in the reference MO basis.
    mo_core_frozen = mo[:, :ncore_frozen]
    mo_core_corr = mo[:, ncore_frozen:ncore_ref]
    mo_act = mo[:, ncore_ref : ncore_ref + n_act_ref]
    mo_virt = mo[:, ncore_ref + n_act_ref : ncore_ref + n_act_ref + n_virt]
    mo_corr = np.hstack([mo_core_corr, mo_act, mo_virt])

    h1e_corr, ecore = _frozen_core_h1e_ecore_df(
        scf_out=scf_out_use,
        mo_core=mo_core_frozen,
        mo_corr=mo_corr,
    )

    B_ao = getattr(scf_out_use, "df_B", None)
    if B_ao is None:
        raise ValueError("scf_out.df_B is required (DF factors missing)")

    want_cuda_ints = str(hop_backend).strip().lower() == "cuda"
    if want_cuda_ints:
        eri_payload = _dfmo_integrals_from_df_B(B_ao, mo_corr, device="cuda")
        if not isinstance(eri_payload, DeviceDFMOIntegrals):
            raise RuntimeError("internal error: expected DeviceDFMOIntegrals for device='cuda'")
        l_full = eri_payload.l_full
        df_ints_ret = eri_payload if bool(return_integrals) else None
    else:
        eri_payload = _dfmo_integrals_from_df_B(_maybe_asnumpy(B_ao), mo_corr, device="cpu")
        if not isinstance(eri_payload, DFMOIntegrals):
            raise RuntimeError("internal error: expected DFMOIntegrals for device='cpu'")
        l_full = eri_payload.l_full
        df_ints_ret = None

    nelec_act = _nelecas_total(getattr(ref, "nelecas"))
    nelec_corr = int(nelec_act) + 2 * correlate_inactive_i
    twos_i = _get_ref_twos(ref, scf_out_use, twos=twos)

    ci_obj = getattr(ref, "ci")
    if isinstance(ci_obj, (list, tuple)):
        nroots_ref = int(len(ci_obj))
    else:
        nroots_ref = int(getattr(ref, "nroots", 1))
    ci_all = ci_as_list(ci_obj, nroots=max(1, nroots_ref))
    if state_i >= len(ci_all):
        raise ValueError("state index out of range for ref.ci")
    ci_act = np.asarray(ci_all[state_i], dtype=np.float64).ravel()

    from asuka.cuguga.drt import build_drt  # noqa: PLC0415

    drt_act = build_drt(norb=int(n_act_ref), nelec=int(nelec_act), twos_target=int(twos_i))
    if int(ci_act.size) != int(drt_act.ncsf):
        raise ValueError("reference CI vector length mismatch with active DRT")

    ci_cas = ci_act
    if correlate_inactive_i > 0:
        ci_cas = embed_ci_with_docc_prefix(
            ci_act=ci_act,
            n_docc=correlate_inactive_i,
            n_act=n_act_ref,
            nelec_act=nelec_act,
            twos=twos_i,
            orbsym_act=None,
            orbsym_full=None,
            wfnsym=wfnsym,
        )

    e_ref = compute_cas_reference_energy_df(
        h1e_corr=h1e_corr,
        l_full=l_full,
        ecore=float(ecore),
        ci_cas=np.asarray(ci_cas, dtype=np.float64),
        n_act=n_act_int,
        nelec=nelec_corr,
        twos=twos_i,
        orbsym_act=None,
        wfnsym=wfnsym,
    )

    nthreads = int(contract_nthreads)
    if nthreads < 1:
        nthreads = 1

    res = mrcisd_kernel(
        h1e=h1e_corr,
        eri=eri_payload,
        n_act=n_act_int,
        n_virt=n_virt,
        nelec=nelec_corr,
        twos=twos_i,
        ci_cas=np.asarray(ci_cas, dtype=np.float64),
        nroots=1,
        ecore=float(ecore),
        orbsym_act=None,
        orbsym_corr=None,
        wfnsym=wfnsym,
        max_virt_e=max_virt_e,
        hop_backend=hop_backend,
        tol=tol,
        max_cycle=max_cycle,
        max_space=max_space,
        max_memory=max_memory_mb,
        contract_nthreads=nthreads,
        contract_blas_nthreads=contract_blas_nthreads,
        precompute_epq=precompute_epq,
    )
    if not isinstance(res, MRCISDResult):
        raise RuntimeError("internal error: expected a single-root MRCISD result")

    e_tot = float(res.e_mrci)
    e_corr = float(e_tot - float(e_ref))

    e_plus_q = None
    q_diag = None
    if bool(plus_q):
        e_plus_q, q_diag = mrcisd_plus_q(
            e_mrci=e_tot,
            e_ref=float(e_ref),
            ci_mrci=res.ci,
            ci_ref0=res.ci_ref0,
            ref_idx=res.ref_idx,
            model=str(plus_q_model),
            min_ref=float(plus_q_min_ref),
        )
        if e_plus_q is not None:
            e_plus_q = float(e_plus_q)

    return MRCIResult(
        method="mrcisd",
        e_ref=float(e_ref),
        e_tot=float(e_tot),
        e_corr=float(e_corr),
        e_tot_plus_q=e_plus_q,
        plus_q_diag=q_diag,
        result=res,
        df_integrals=df_ints_ret,
    )


def mrci_states_from_ref_soc(
    ref: Any,
    *,
    scf_out: Any | None = None,
    soc_integrals: SOCIntegrals | None = None,
    soc_method: Literal["integrals", "amfi"] = "amfi",
    # --- AMFI (internal SOC integrals) ---
    amfi_scale: float = 1.0,
    amfi_include_mean_field: bool = True,
    amfi_atoms: Sequence[int] | None = None,
    amfi_rme_scale: float = 4.0,
    amfi_phase: complex = 1j,
    # --- MRCI args forwarded ---
    method: Method = "mrcisd",
    states: Sequence[int] | None = None,
    nroots: int | None = None,
    n_virt: int | None = None,
    twos: int | None = None,
    max_virt_e: int = 2,
    correlate_inactive: int = 0,
    orbsym: Sequence[int] | None = None,
    wfnsym: int | None = None,
    integrals_backend: DFIntegralsBackend = "df_B",
    hop_backend: str | None = None,
    tol: float = 1e-10,
    max_cycle: int = 400,
    max_space: int = 30,
    max_memory_mb: float = 4000.0,
    contract_nthreads: int = 1,
    contract_blas_nthreads: int | None = None,
    precompute_epq: bool = True,
    return_integrals: bool = False,
    # --- SOC args ---
    soc_backend: str = "auto",
    soc_block_nops: int = 8,
    soc_symmetrize: bool = True,
    soc_cuda_threads: int = 128,
    soc_cuda_sync: bool = True,
    soc_cuda_fallback_to_cpu: bool = True,
    soc_cuda_gm_strategy: str = "auto",
    soc_cuda_gm_direct_max_nb_nk: int = 256,
    return_h_si: bool = False,
) -> MRCISOCResult:
    """Run ASUKA-native MRCISD, then perform SOC-SI on the correlated roots."""

    scf_out_use = _resolve_scf_out_from_ref(ref, scf_out=scf_out)

    mrci = mrci_states_from_ref(
        ref,
        scf_out=scf_out_use,
        method=method,
        states=states,
        nroots=nroots,
        n_virt=n_virt,
        twos=twos,
        max_virt_e=max_virt_e,
        correlate_inactive=correlate_inactive,
        orbsym=orbsym,
        wfnsym=wfnsym,
        integrals_backend=integrals_backend,
        hop_backend=hop_backend,
        tol=tol,
        max_cycle=max_cycle,
        max_space=max_space,
        max_memory_mb=max_memory_mb,
        contract_nthreads=contract_nthreads,
        contract_blas_nthreads=contract_blas_nthreads,
        precompute_epq=precompute_epq,
        return_integrals=return_integrals,
    )

    soc_method_s = str(soc_method).strip().lower()
    if soc_integrals is None:
        if soc_method_s != "amfi":
            raise ValueError("soc_integrals is None; set soc_method='amfi' to build internal AMFI SOC integrals")
        from asuka.soc.amfi import build_amfi_soc_integrals_from_scf_out  # noqa: PLC0415

        mo = getattr(ref, "mo_coeff", None)
        if mo is None:
            raise ValueError("ref.mo_coeff is required to build AMFI SOC integrals")
        mo = _maybe_asnumpy(mo)

        ncore_ref = int(getattr(ref, "ncore", 0))
        n_act_ref = int(getattr(ref, "ncas", 0))
        correlate_inactive_i = int(correlate_inactive)
        if correlate_inactive_i < 0 or correlate_inactive_i > ncore_ref:
            raise ValueError("correlate_inactive must satisfy 0 <= correlate_inactive <= ref.ncore")
        ncore_frozen = ncore_ref - correlate_inactive_i
        nmo = int(mo.shape[1])
        nvirt_all = nmo - ncore_ref - n_act_ref
        if nvirt_all < 0:
            raise RuntimeError("invalid orbital partition: ncore+ncas > nmo")

        n_virt_use = nvirt_all if n_virt is None else int(n_virt)
        if n_virt_use < 0 or n_virt_use > nvirt_all:
            raise ValueError("n_virt must satisfy 0 <= n_virt <= (nmo-ncore-ncas)")

        # Orbital ordering in MRCI kernels: [core_corr][active][virt]
        mo_core_corr = mo[:, ncore_frozen:ncore_ref]
        mo_act = mo[:, ncore_ref : ncore_ref + n_act_ref]
        mo_virt = mo[:, ncore_ref + n_act_ref : ncore_ref + n_act_ref + n_virt_use]
        mo_corr = np.hstack([mo_core_corr, mo_act, mo_virt])

        soc_integrals_use = build_amfi_soc_integrals_from_scf_out(
            scf_out_use,
            mo_coeff=mo_corr,
            rme_scale=float(amfi_rme_scale),
            phase=complex(amfi_phase),
            scale=float(amfi_scale),
            include_mean_field=bool(amfi_include_mean_field),
            atoms=amfi_atoms,
        )
    else:
        if not isinstance(soc_integrals, SOCIntegrals):
            raise TypeError("soc_integrals must be a asuka.soc.SOCIntegrals")
        soc_integrals_use = soc_integrals

    states_sf: list[SpinFreeState] = []
    for i in range(int(mrci.nroots)):
        states_sf.append(
            SpinFreeState(
                twos=int(mrci.twos),
                energy=float(np.asarray(mrci.mrci.e_mrci, dtype=np.float64).ravel()[i]),
                drt=mrci.mrci.drt,
                ci=np.asarray(mrci.mrci.ci[i], dtype=np.float64).ravel(),
            )
        )

    e_so, c_so, basis = soc_state_interaction(
        states_sf,
        soc_integrals_use,
        include_diag=True,
        block_nops=int(soc_block_nops),
        symmetrize=bool(soc_symmetrize),
        backend=str(soc_backend),
        cuda_threads=int(soc_cuda_threads),
        cuda_sync=bool(soc_cuda_sync),
        cuda_fallback_to_cpu=bool(soc_cuda_fallback_to_cpu),
        cuda_gm_strategy=str(soc_cuda_gm_strategy),
        cuda_gm_direct_max_nb_nk=int(soc_cuda_gm_direct_max_nb_nk),
    )

    h_si = None
    if bool(return_h_si):
        c = np.asarray(c_so, dtype=np.complex128)
        e = np.asarray(e_so, dtype=np.float64).ravel()
        h_si = np.asarray(c @ np.diag(e.astype(np.complex128)) @ c.conj().T, dtype=np.complex128)

    return MRCISOCResult(
        mrci=mrci,
        spinfree_states=list(states_sf),
        so_energies=np.asarray(e_so, dtype=np.float64),
        so_vectors=np.asarray(c_so, dtype=np.complex128),
        so_basis=list(basis),
        h_si=h_si,
    )


__all__ = [
    "mrci_from_ref",
    "mrci_states_from_ref",
    "mrci_states_from_ref_soc",
]
