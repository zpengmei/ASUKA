"""Top-level CASPT2 gradient dispatcher."""

from __future__ import annotations

import os
from typing import Any, Literal

import numpy as np

from asuka.caspt2.gradient.fd import fd_nuclear_gradient_from_casscf
from asuka.caspt2.gradient.native_grad import caspt2_ss_gradient_native
from asuka.caspt2.gradient.slag import build_state_lagrangian
from asuka.caspt2.result import CASPT2GradResult


_VALID_PARITY_PROFILES: tuple[str, ...] = ("default", "molcas_ss_strict")


def caspt2_gradient_from_casscf(
    scf_out: Any,
    casscf: Any,
    *,
    method: str = "SS",
    iroot: int = 0,
    pt2_backend: str = "ic",
    df_backend: str = "cpu",
    int1e_contract_backend: str = "cpu",
    parity_profile: str = "default",
    gradient_backend: Literal["auto", "analytic", "fd"] = "auto",
    verbose: int = 0,
) -> Any:
    """Compute CASPT2 nuclear gradient from CASSCF reference.

    Parameters
    ----------
    scf_out : Any
        ASUKA SCF result (provides DF tensors, AO basis, hcore).
    casscf : Any
        ASUKA CASSCF result (mo_coeff, ci, ncore, ncas, nelecas).
    method : str
        CASPT2 variant: "SS" (single-state), "MS", "XMS".
    iroot : int
        Target state index.
    pt2_backend : str
        CASPT2 energy backend selector forwarded to the FD driver and used by
        the SS analytic gradient to source IC-equivalent amplitudes when using
        the SST backend (``\"sst\"``, ``\"sst-ic\"``, ``\"sst-full\"``).
    df_backend : str
        Backend for DF derivative contraction ("cpu" or "cuda").
    int1e_contract_backend : str
        Backend for 1e derivative contractions.
    parity_profile : str
        Gradient convention profile.
    gradient_backend : {"auto", "analytic", "fd"}
        Gradient evaluation backend. ``auto`` uses analytic SS and FD MS/XMS.
    verbose : int
        Verbosity level.

    Returns
    -------
    CASPT2GradResult
        The computed gradient.
    """
    method_u = str(method).upper().strip()
    parity_u = str(parity_profile).strip().lower()
    pt2_backend_u = str(pt2_backend).strip().lower()
    grad_backend_u = str(
        os.environ.get("ASUKA_CASPT2_GRAD_BACKEND", gradient_backend)
    ).strip().lower()

    # In SS, the "analytic" backend includes the optional FD correction patch
    # (enabled by default). Treat "auto" as "analytic" so callers get the
    # most reliable result without having to pick a backend explicitly.
    if method_u == "SS" and grad_backend_u == "auto":
        grad_backend_u = "analytic"

    if parity_u not in _VALID_PARITY_PROFILES:
        raise ValueError(
            f"Unknown parity_profile='{parity_profile}'. "
            f"Expected one of: {', '.join(_VALID_PARITY_PROFILES)}"
        )
    if grad_backend_u not in {"auto", "analytic", "fd"}:
        raise ValueError(
            f"Unknown gradient_backend='{gradient_backend}'. "
            "Expected one of: auto, analytic, fd"
        )

    if grad_backend_u in {"fd", "auto"}:
        try:
            fd_step_bohr = float(os.environ.get("ASUKA_CASPT2_MSXMS_GRAD_FD_STEP_BOHR", "1.0e-4"))
        except Exception:
            fd_step_bohr = 1.0e-4
        ss_analytic = None
        if method_u == "SS":
            try:
                ss_analytic = caspt2_ss_gradient_native(
                    scf_out,
                    casscf,
                    iroot=int(iroot),
                    pt2_backend=str(pt2_backend_u),
                    df_backend=str(df_backend),
                    int1e_contract_backend=str(int1e_contract_backend),
                    parity_profile=parity_u,
                    verbose=int(verbose),
                )
                if grad_backend_u == "auto" and ss_analytic is not None:
                    return ss_analytic
            except Exception:
                ss_analytic = None
        elif method_u in {"MS", "XMS"} and grad_backend_u == "auto" and pt2_backend_u not in {"sst", "sst-ic", "sst-full"}:
            try:
                from asuka.caspt2.gradient.ms_grad import caspt2_ms_gradient_native
                return caspt2_ms_gradient_native(
                    scf_out,
                    casscf,
                    method=method_u,
                    iroot=int(iroot),
                    df_backend=str(df_backend),
                    int1e_contract_backend=str(int1e_contract_backend),
                    verbose=int(verbose),
                )
            except Exception:
                pass  # fall through to FD
        base_res, grad_fd, fd_points = fd_nuclear_gradient_from_casscf(
            scf_out,
            casscf,
            method=method_u,
            iroot=int(iroot),
            verbose=int(verbose),
            step_bohr=float(fd_step_bohr),
            caspt2_kwargs={"pt2_backend": str(pt2_backend_u)},
        )
        e_tot = np.asarray(getattr(base_res, "e_tot"), dtype=np.float64).ravel()
        e_ref = np.asarray(getattr(base_res, "e_ref"), dtype=np.float64).ravel()
        e_pt2 = np.asarray(getattr(base_res, "e_pt2"), dtype=np.float64).ravel()
        nstates = int(e_tot.size)
        if int(iroot) < 0 or int(iroot) >= int(nstates):
            raise ValueError(f"iroot={int(iroot)} out of range for {method_u} with nstates={nstates}")
        breakdown = dict(getattr(base_res, "breakdown", {}) or {})
        if ss_analytic is not None:
            breakdown["grad_analytic"] = np.asarray(getattr(ss_analytic, "grad"), dtype=np.float64)
            for key, val in dict(getattr(ss_analytic, "breakdown", {}) or {}).items():
                breakdown.setdefault(key, val)
        breakdown["grad_source"] = "fd"
        breakdown["fd_points"] = list(fd_points)

        slag = None
        clag = None
        olag = None
        wlag = None
        dpt2_1rdm = None
        convergence_flags = {
            "gradient_backend": "fd",
            "energy_backend": "native",
            "fd_step_bohr": float(fd_step_bohr),
        }
        if ss_analytic is not None:
            clag = getattr(ss_analytic, "clag", None)
            olag = getattr(ss_analytic, "olag", None)
            wlag = getattr(ss_analytic, "wlag", None)
            dpt2_1rdm = getattr(ss_analytic, "dpt2_1rdm", None)
            convergence_flags.update(dict(getattr(ss_analytic, "convergence_flags", {}) or {}))
        if method_u in {"MS", "XMS"}:
            ueff = np.asarray(getattr(base_res, "ueff"), dtype=np.float64)
            if ueff.shape != (nstates, nstates):
                raise ValueError(f"{method_u} CASPT2Result.ueff shape mismatch: {ueff.shape}")
            u0 = None
            if method_u == "XMS":
                if "u0" not in breakdown:
                    raise ValueError("XMS CASPT2Result.breakdown['u0'] is missing")
                u0 = np.asarray(breakdown.get("u0"), dtype=np.float64)
            slag = build_state_lagrangian(
                int(nstates),
                ueff,
                int(iroot),
                is_xms=bool(method_u == "XMS"),
                u0=u0,
            )
            breakdown["ueff"] = np.asarray(ueff, dtype=np.float64)
            breakdown["slag"] = np.asarray(slag, dtype=np.float64)
            if getattr(base_res, "heff", None) is not None:
                breakdown["heff"] = np.asarray(getattr(base_res, "heff"), dtype=np.float64)

        return CASPT2GradResult(
            e_tot=float(e_tot[int(iroot)]),
            e_ref=float(e_ref[int(iroot)]),
            e_pt2=float(e_pt2[int(iroot)]),
            grad=np.asarray(grad_fd, dtype=np.float64),
            method=str(method_u),
            iroot=int(iroot),
            nstates=int(nstates),
            clag=None if clag is None else np.asarray(clag, dtype=np.float64),
            olag=None if olag is None else np.asarray(olag, dtype=np.float64),
            slag=None if slag is None else np.asarray(slag, dtype=np.float64),
            wlag=None if wlag is None else np.asarray(wlag, dtype=np.float64),
            dpt2_1rdm=None if dpt2_1rdm is None else np.asarray(dpt2_1rdm, dtype=np.float64),
            convergence_flags=convergence_flags,
            breakdown=breakdown,
        )

    if method_u == "SS" and grad_backend_u == "analytic":
        out = caspt2_ss_gradient_native(
            scf_out,
            casscf,
            iroot=int(iroot),
            pt2_backend=str(pt2_backend_u),
            df_backend=str(df_backend),
            int1e_contract_backend=str(int1e_contract_backend),
            parity_profile=parity_u,
            verbose=int(verbose),
        )
        if str(
            dict(getattr(out, "convergence_flags", {}) or {}).get(
                "pt2_correction_backend", ""
            )
        ).strip().lower() == "fd":
            return out
        use_fd_patch = str(
            os.environ.get("ASUKA_CASPT2_SS_ANALYTIC_FD_PATCH", "1")
        ).strip().lower() not in {"0", "false", "off", "no"}
        if not bool(use_fd_patch):
            return out

        try:
            fd_step_bohr = float(os.environ.get("ASUKA_CASPT2_MSXMS_GRAD_FD_STEP_BOHR", "1.0e-4"))
        except Exception:
            fd_step_bohr = 1.0e-4
        _fd_res, grad_fd, fd_points = fd_nuclear_gradient_from_casscf(
            scf_out,
            casscf,
            method=method_u,
            iroot=int(iroot),
            verbose=int(verbose),
            step_bohr=float(fd_step_bohr),
            caspt2_kwargs={"pt2_backend": str(pt2_backend_u)},
        )
        grad_ref = np.asarray((dict(getattr(out, "breakdown", {}) or {}).get("grad_ref")), dtype=np.float64)
        grad_analytic = np.asarray(getattr(out, "grad"), dtype=np.float64)
        grad_fd = np.asarray(grad_fd, dtype=np.float64)
        grad_pt2_analytic = np.asarray((dict(getattr(out, "breakdown", {}) or {}).get("grad_pt2")), dtype=np.float64)
        grad_pt2_fd = np.asarray(grad_fd - grad_ref, dtype=np.float64)
        breakdown = dict(getattr(out, "breakdown", {}) or {})
        breakdown["grad_source"] = "analytic+fd_patch"
        breakdown["fd_points"] = list(fd_points)
        breakdown["grad_analytic"] = np.asarray(grad_analytic, dtype=np.float64)
        breakdown["grad_pt2_analytic"] = np.asarray(grad_pt2_analytic, dtype=np.float64)
        breakdown["grad_pt2_fd_patch"] = np.asarray(grad_pt2_fd, dtype=np.float64)
        breakdown["grad_pt2_fd_patch_delta"] = np.asarray(grad_pt2_fd - grad_pt2_analytic, dtype=np.float64)
        breakdown["grad_total_fd_patch_delta"] = np.asarray(grad_fd - grad_analytic, dtype=np.float64)
        breakdown["grad_pt2"] = np.asarray(grad_pt2_fd, dtype=np.float64)
        breakdown["grad_total_rebuilt"] = np.asarray(grad_ref + grad_pt2_fd, dtype=np.float64)
        convergence_flags = dict(getattr(out, "convergence_flags", {}) or {})
        convergence_flags["pt2_correction_backend"] = "fd"
        convergence_flags["fd_step_bohr"] = float(fd_step_bohr)
        return CASPT2GradResult(
            e_tot=float(getattr(out, "e_tot")),
            e_ref=float(getattr(out, "e_ref")),
            e_pt2=float(getattr(out, "e_pt2")),
            grad=np.asarray(grad_fd, dtype=np.float64),
            method=str(getattr(out, "method", "SS")),
            iroot=int(getattr(out, "iroot", iroot)),
            nstates=getattr(out, "nstates", None),
            clag=None if getattr(out, "clag", None) is None else np.asarray(getattr(out, "clag"), dtype=np.float64),
            olag=None if getattr(out, "olag", None) is None else np.asarray(getattr(out, "olag"), dtype=np.float64),
            slag=None if getattr(out, "slag", None) is None else np.asarray(getattr(out, "slag"), dtype=np.float64),
            wlag=None if getattr(out, "wlag", None) is None else np.asarray(getattr(out, "wlag"), dtype=np.float64),
            dpt2_1rdm=None if getattr(out, "dpt2_1rdm", None) is None else np.asarray(getattr(out, "dpt2_1rdm"), dtype=np.float64),
            dpt2_2rdm=None if getattr(out, "dpt2_2rdm", None) is None else np.asarray(getattr(out, "dpt2_2rdm"), dtype=np.float64),
            convergence_flags=convergence_flags,
            breakdown=breakdown,
        )
    if method_u in {"MS", "XMS"} and grad_backend_u == "analytic":
        from asuka.caspt2.gradient.ms_grad import caspt2_ms_gradient_native
        return caspt2_ms_gradient_native(
            scf_out,
            casscf,
            method=method_u,
            iroot=int(iroot),
            df_backend=str(df_backend),
            int1e_contract_backend=str(int1e_contract_backend),
            verbose=int(verbose),
        )
    raise ValueError(f"Unknown CASPT2 gradient method '{method_u}'")
