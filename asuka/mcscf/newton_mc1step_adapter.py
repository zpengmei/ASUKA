from __future__ import annotations

from functools import reduce
from typing import Any

import numpy as np


def _to_numpy_f64(a: Any) -> np.ndarray:
    try:
        import cupy as cp  # type: ignore
    except Exception:
        cp = None  # type: ignore
    if cp is not None and isinstance(a, cp.ndarray):  # type: ignore[attr-defined]
        a = cp.asnumpy(a)
    return np.asarray(a, dtype=np.float64)


def _get_xp(*arrays: Any):
    try:
        import cupy as cp  # type: ignore
    except Exception:
        cp = None  # type: ignore
    if cp is not None:
        for a in arrays:
            if isinstance(a, cp.ndarray):  # type: ignore[attr-defined]
                return cp, True
    return np, False


def _to_xp_f64(a: Any, xp: Any):
    if xp is np:
        return _to_numpy_f64(a)
    return xp.asarray(a, dtype=xp.float64)


def _scalar_float(a: Any) -> float:
    if hasattr(a, "item"):
        try:
            return float(a.item())
        except Exception:
            pass
    return float(np.asarray(a).reshape(()))


def _norm_f64(a: Any) -> float:
    xp, _ = _get_xp(a)
    arr = xp.asarray(a, dtype=xp.float64).ravel()
    return _scalar_float(xp.linalg.norm(arr))


class NewtonMC1StepAdapterMixin:
    """Shared mc1step-compatible helpers for ASUKA Newton adapters.

    The adapters remain lightweight and only implement the subset of the
    PySCF CASSCF interface that ASUKA's internal 1step driver needs.
    """

    max_stepsize: float = 0.02
    ah_level_shift: float = 1e-8
    ah_conv_tol: float = 1e-12
    ah_max_cycle: int = 30
    ah_lindep: float = 1e-14
    ah_start_tol: float = 2.5
    ah_start_cycle: int = 3
    max_cycle_micro: int = 4
    kf_interval: int = 4
    ah_grad_trust_region: float = 3.0
    kf_trust_region: float = 3.0
    small_rot_tol: float = 1e-4
    conv_tol: float = 1e-8
    max_memory: int = 4000
    ci_response_space: int = 4
    ci_grad_trust_region: float = 3.0
    scale_restoration: float = 0.5
    with_dep4: bool = False
    verbose: int = 0
    callback: Any = None
    chkfile: str | None = None
    canonicalization: bool = False
    sorting_mo_energy: bool = False
    natorb: bool = False
    _max_stepsize: float | None = None

    def rotate_orb_cc(
        self,
        mo: Any,
        fcivec: Any,
        fcasdm1: Any,
        fcasdm2: Any,
        eris: Any,
        x0_guess: Any = None,
        conv_tol_grad: float = 1e-4,
        max_stepsize: float | None = None,
        verbose: Any = None,
    ):
        from asuka.mcscf.newton_casscf import davidson_cc  # noqa: PLC0415

        if max_stepsize is None:
            max_stepsize = float(self.max_stepsize)

        u = 1
        g_orb, gorb_update, h_op, h_diag = self.gen_g_hop(mo, u, fcasdm1(), fcasdm2(), eris)
        xp, _on_gpu = _get_xp(g_orb, mo, h_diag, x0_guess)
        g_kf = xp.asarray(g_orb, dtype=xp.float64).ravel()
        norm_gkf = norm_gorb = _norm_f64(g_kf)

        if norm_gorb < float(conv_tol_grad) * 0.3:
            u = self.update_rotate_matrix(g_kf * 0.0)
            yield u, g_kf, 1, x0_guess
            return

        def precond(x, e):
            hdiagd = xp.asarray(h_diag, dtype=xp.float64).ravel() - (float(e) - float(self.ah_level_shift))
            hdiagd = xp.where(xp.abs(hdiagd) < 1e-8, xp.asarray(1e-8, dtype=xp.float64), hdiagd)
            x = xp.asarray(x, dtype=xp.float64).ravel() / hdiagd
            norm_x = _norm_f64(x)
            if norm_x > 0.0:
                x *= 1.0 / norm_x
            return x

        jkcount = 0
        if x0_guess is None:
            x0_guess = g_kf.copy()
        imic = 0
        dr = xp.zeros_like(g_kf)
        ikf = 0

        def g_op():
            return xp.asarray(g_kf if ikf == 0 else g_orb_cur, dtype=xp.float64).ravel()

        problem_size = int(g_kf.size)
        g_orb_cur = xp.asarray(g_kf, dtype=xp.float64).ravel()
        for ah_end, ihop, w, dxi, hdxi, residual, seig in davidson_cc(
            h_op,
            g_op,
            precond,
            x0_guess,
            tol=float(self.ah_conv_tol),
            max_cycle=int(self.ah_max_cycle),
            lindep=float(self.ah_lindep),
            verbose=None,
        ):
            norm_residual = _norm_f64(residual)
            accept = (
                bool(ah_end)
                or int(ihop) == int(self.ah_max_cycle)
                or ((norm_residual < float(self.ah_start_tol)) and (int(ihop) >= int(self.ah_start_cycle)))
                or (float(seig) < float(self.ah_lindep))
            )
            if not accept:
                continue

            imic += 1
            dxi = xp.asarray(dxi, dtype=xp.float64).ravel()
            hdxi = xp.asarray(hdxi, dtype=xp.float64).ravel()
            dxmax = _scalar_float(xp.max(xp.abs(dxi))) if dxi.size else 0.0
            if dxmax > float(max_stepsize):
                scale = float(max_stepsize) / dxmax
                dxi *= scale
                hdxi *= scale

            g_orb_cur = g_orb_cur + hdxi
            dr = dr + dxi
            norm_gorb = _norm_f64(g_orb_cur)

            ikf += 1
            if ikf > 1 and norm_gorb > norm_gkf * float(self.ah_grad_trust_region):
                g_orb_cur = g_orb_cur - hdxi
                dr = dr - dxi
                break
            if norm_gorb < float(conv_tol_grad) * 0.3:
                break

            need_keyframe = (
                ikf >= max(int(self.kf_interval), int(max(0.0, -np.log(_norm_f64(dr) + 1e-7))))
                or norm_gorb < norm_gkf / float(self.kf_trust_region)
            )
            if need_keyframe:
                ikf = 0
                u = self.update_rotate_matrix(dr, u)
                yield u, xp.asarray(g_kf, dtype=xp.float64).ravel(), int(ihop) + int(jkcount), dxi
                g_kf1 = xp.asarray(gorb_update(u, fcivec()), dtype=xp.float64).ravel()
                jkcount += 1
                norm_gkf1 = _norm_f64(g_kf1)
                norm_dg = _norm_f64(g_kf1 - g_orb_cur)
                if (
                    norm_dg > norm_gorb * float(self.ah_grad_trust_region)
                    and norm_gkf1 > norm_gkf
                    and norm_gkf1 > norm_gkf * float(self.ah_grad_trust_region)
                ):
                    dr = -dxi * (1.0 - float(self.scale_restoration))
                    g_kf = g_kf1
                    break
                g_orb_cur = g_kf = g_kf1
                norm_gorb = norm_gkf = norm_gkf1
                dr[:] = 0.0
                x0_guess = xp.asarray(dxi, dtype=xp.float64).ravel()

        u = self.update_rotate_matrix(dr, u)
        r_last = dxi if "dxi" in locals() else x0_guess
        yield u, xp.asarray(g_kf, dtype=xp.float64).ravel(), int(jkcount), xp.asarray(r_last, dtype=xp.float64).ravel()

    def casci(self, mo_coeff: Any, ci0: Any = None, eris: Any = None, verbose: Any = None, envs: Any = None):
        runner = getattr(self, "_asuka_casci_runner", None)
        if runner is None:
            raise RuntimeError("adapter is missing _asuka_casci_runner")
        casci_out = runner(mo_coeff, ci0=ci0, eris=eris, verbose=verbose, envs=envs)
        self.e_tot = float(np.asarray(casci_out.e_tot, dtype=np.float64).ravel()[0])
        self.e_cas = float(self.e_tot - float(casci_out.ecore))
        self.ci = casci_out.ci
        self.mo_coeff = mo_coeff
        return self.e_tot, self.e_cas, self.ci

    def dump_chk(self, envs: Any) -> None:
        return None

    def rotate_mo(self, mo: Any, u: Any, log: Any = None) -> np.ndarray:
        xp, _on_gpu = _get_xp(mo, u)
        mo_xp = _to_xp_f64(mo, xp)
        if np.isscalar(u):
            if float(u) == 1.0:
                u_xp = xp.eye(int(mo_xp.shape[1]), dtype=xp.float64)
            else:
                raise ValueError("scalar orbital rotation is only supported for u=1")
        else:
            u_xp = _to_xp_f64(u, xp)
        mo_new = mo_xp @ u_xp
        if log is not None and getattr(log, "verbose", 0) >= 5:
            ncore = int(self.ncore)
            ncas = int(self.ncas)
            nocc = ncore + ncas
            s_act = _to_numpy_f64(mo_new[:, ncore:nocc]).T @ _to_numpy_f64(mo_xp[:, ncore:nocc])
            try:
                sv = np.linalg.svd(s_act, compute_uv=False)
                log.debug("Active-space overlap to prior step, SVD = %s", sv)
            except Exception:
                pass
        return xp.ascontiguousarray(_to_xp_f64(mo_new, xp))

    def micro_cycle_scheduler(self, envs: dict[str, Any]) -> int:
        norm_ddm = float(envs.get("norm_ddm", 1.0))
        if norm_ddm <= 0.0:
            return int(self.max_cycle_micro)
        return max(int(self.max_cycle_micro), int(self.max_cycle_micro - 1 - np.log(norm_ddm)))

    def max_stepsize_scheduler(self, envs: dict[str, Any]) -> float:
        cur = envs.get("max_stepsize", None)
        if cur is None:
            cur = float(self.max_stepsize)
        else:
            cur = float(cur)
        de = envs.get("de", None)
        if de is not None and float(de) > -float(self.conv_tol):
            cur *= 0.3
        else:
            cur = float(np.sqrt(max(float(self.max_stepsize) * cur, 1e-16)))
        self._max_stepsize = float(cur)
        return float(cur)

    def gen_g_hop(self, mo: Any, u: Any, fcasdm1: Any, fcasdm2: Any, eris: Any):
        from asuka.mcscf.newton_casscf import gen_g_hop_orbital  # noqa: PLC0415

        mo_u = self.rotate_mo(mo, u)
        ci_cur = getattr(self, "_mc1step_ci_current", None)
        if ci_cur is None:
            raise RuntimeError("mc1step orbital Hessian requires _mc1step_ci_current")
        g, h_op, h_diag, gorb_update_full = gen_g_hop_orbital(
            self,
            mo_u,
            ci_cur,
            eris,
            weights=getattr(self, "weights", None),
            strict_weights=False,
        )
        xp, _ = _get_xp(g, mo_u, ci_cur)
        def _gorb_update_only(u_rot: Any, ci_new: Any) -> Any:
            g_new, _h_op_unused, _h_diag_unused = gorb_update_full(u_rot, ci_new)
            xp_g, _ = _get_xp(g_new, mo_u, ci_new)
            return xp_g.asarray(g_new, dtype=xp_g.float64).ravel()

        return xp.asarray(g, dtype=xp.float64).ravel(), _gorb_update_only, h_op, h_diag

    def solve_approx_ci(self, h1: Any, h2: Any, ci0: Any, ecore: float, e_cas: float, envs: dict[str, Any]):
        xp, _on_gpu = _get_xp(h1, h2, ci0)
        if bool(getattr(self, "_asuka_force_cpu", False)):
            xp = np
            _on_gpu = False
        h1_solver = h1 if xp is not np else _to_numpy_f64(h1)
        h2_solver = h2 if xp is not np else _to_numpy_f64(h2)
        ci0_solver = ci0 if xp is not np else _to_numpy_f64(ci0)
        ncas = int(self.ncas)
        nelecas = self.nelecas
        if "norm_gorb" in envs:
            tol = max(float(self.conv_tol), float(envs["norm_gorb"]) ** 2 * 0.1)
        else:
            tol = None
        if getattr(self.fcisolver, "approx_kernel", None):
            e, ci1 = self.fcisolver.approx_kernel(
                h1_solver,
                h2_solver,
                ncas,
                nelecas,
                ecore=float(ecore),
                ci0=ci0_solver,
                tol=tol,
                max_memory=int(getattr(self, "max_memory", 4000)),
                return_cupy=bool(_on_gpu),
            )
            return ci1, None
        if not (getattr(self.fcisolver, "contract_2e", None) and getattr(self.fcisolver, "absorb_h1e", None)):
            e, ci1 = self.fcisolver.kernel(
                h1_solver,
                h2_solver,
                ncas,
                nelecas,
                ecore=float(ecore),
                ci0=ci0_solver,
                tol=tol,
                max_memory=int(getattr(self, "max_memory", 4000)),
                max_cycle=int(getattr(self, "ci_response_space", 4)),
                return_cupy=bool(_on_gpu),
            )
            return ci1, None

        h2eff = self.fcisolver.absorb_h1e(h1_solver, h2_solver, ncas, nelecas, 0.5)

        def contract_2e(c):
            hc = self.fcisolver.contract_2e(h2eff, c, ncas, nelecas)
            return _to_xp_f64(hc, xp).ravel()

        ci0_xp = _to_xp_f64(ci0, xp).ravel()
        hc = contract_2e(ci0_xp)
        g = hc - (float(e_cas) - float(ecore)) * ci0_xp
        e, ci1 = self.fcisolver.kernel(
            h1_solver,
            h2_solver,
            ncas,
            nelecas,
            ecore=float(ecore),
            ci0=ci0_solver,
            tol=tol,
            max_memory=int(getattr(self, "max_memory", 4000)),
            max_cycle=int(getattr(self, "ci_response_space", 4)),
            return_cupy=bool(_on_gpu),
        )
        return ci1, g

    def _build_provider_microstep_operator(
        self,
        *,
        provider: Any,
        ua: Any,
        ra: Any,
        ddm: Any,
        h1e_mo: Any,
        vhf_c: Any,
        C_act_ref: Any,
        xp: Any,
    ) -> tuple[Any, Any]:
        """Provider-backed micro-step Hamiltonian assembly.

        This is the fused operator path for the `1step` micro-iteration: it
        builds the CI-response Hamiltonian directly from the provider seam
        without materialized `ppaa/papa`.
        """
        ncas = int(self.ncas)
        ua_xp = _to_xp_f64(ua, xp)
        ra_xp = _to_xp_f64(ra, xp)
        vhf_c_xp = _to_xp_f64(vhf_c, xp)
        C_act_xp = _to_xp_f64(C_act_ref, xp)

        Jddm, Kddm = provider.jk(ddm, want_J=True, want_K=True)
        if Jddm is None or Kddm is None:  # pragma: no cover
            raise RuntimeError("provider.jk returned None while J/K were requested")
        jk = ua_xp.T @ vhf_c_xp @ ua_xp
        jk += ua_xp.T @ (xp.asarray(Jddm, dtype=xp.float64) - 0.5 * xp.asarray(Kddm, dtype=xp.float64)) @ ua_xp
        h1 = ua_xp.T @ h1e_mo @ ua_xp + jk

        aa11 = xp.asarray(provider.build_pq_uv(ua_xp, ua_xp), dtype=xp.float64).reshape((ncas,) * 4)
        aaaa = xp.asarray(provider.build_pq_uv(C_act_xp, C_act_xp), dtype=xp.float64).reshape((ncas,) * 4)
        aa11 = aa11 + aa11.transpose(2, 3, 0, 1) - aaaa

        a11a = xp.asarray(provider.build_pu_qv(ra_xp, ua_xp), dtype=xp.float64).reshape((ncas,) * 4).transpose(0, 1, 3, 2)
        a11a = a11a + a11a.transpose(1, 0, 2, 3)
        a11a = a11a + a11a.transpose(0, 1, 3, 2)
        return h1, aa11 + a11a

    def update_casdm(self, mo: Any, u: Any, fcivec: Any, e_cas: float, eris: Any, envs: dict[str, Any] | None = None):
        envs = {} if envs is None else envs
        xp, _on_gpu = _get_xp(mo, u, getattr(eris, "ppaa", None), getattr(eris, "papa", None), getattr(eris, "vhf_c", None))
        if bool(getattr(self, "_asuka_force_cpu", False)):
            xp = np
            _on_gpu = False
        ncas = int(self.ncas)
        ncore = int(self.ncore)
        nocc = ncore + ncas
        mo_xp = _to_xp_f64(mo, xp)
        u_xp = _to_xp_f64(u, xp)
        rmat = u_xp - xp.eye(int(mo_xp.shape[1]), dtype=xp.float64)
        uc = u_xp[:, :ncore]
        ua = xp.ascontiguousarray(u_xp[:, ncore:nocc])
        ra = xp.ascontiguousarray(rmat[:, ncore:nocc])
        hcore_xp = _to_xp_f64(self.get_hcore(), xp)
        h1e_mo = mo_xp.T @ hcore_xp @ mo_xp
        ddm = (uc @ uc.T) * 2.0
        if ncore:
            ddm[np.diag_indices(ncore)] -= 2.0
        jk = ua.T @ _to_xp_f64(eris.vhf_c, xp) @ ua
        provider = getattr(eris, "eri_provider", None)
        mo_ref = getattr(eris, "mo_coeff", None)
        C_act_ref = getattr(eris, "C_act", None)
        if provider is not None and mo_ref is not None and C_act_ref is not None:
            h1, h2 = self._build_provider_microstep_operator(
                provider=provider,
                ua=ua,
                ra=ra,
                ddm=ddm,
                h1e_mo=h1e_mo,
                vhf_c=eris.vhf_c,
                C_act_ref=C_act_ref,
                xp=xp,
            )
        else:
            p1aa = xp.empty((int(mo_xp.shape[1]), ncas, ncas * ncas), dtype=xp.float64)
            paa1 = xp.empty((int(mo_xp.shape[1]), ncas * ncas, ncas), dtype=xp.float64)
            ppaa = _to_xp_f64(eris.ppaa, xp)
            papa = _to_xp_f64(eris.papa, xp)
            for i in range(int(mo_xp.shape[1])):
                jbuf = ppaa[i]
                kbuf = papa[i]
                jk += xp.einsum("quv,q->uv", jbuf, ddm[i], optimize=True)
                jk -= 0.5 * xp.einsum("uqv,q->uv", kbuf, ddm[i], optimize=True)
                p1aa[i] = ua.T @ jbuf.reshape(int(mo_xp.shape[1]), -1)
                paa1[i] = kbuf.transpose(0, 2, 1).reshape(-1, int(mo_xp.shape[1])) @ ra
            h1 = ua.T @ h1e_mo @ ua + jk
            aa11 = (ua.T @ p1aa.reshape(int(mo_xp.shape[1]), -1)).reshape((ncas,) * 4)
            aaaa = ppaa[ncore:nocc, ncore:nocc, :, :]
            aa11 = aa11 + aa11.transpose(2, 3, 0, 1) - aaaa
            a11a = (ra.T @ paa1.reshape(int(mo_xp.shape[1]), -1)).reshape((ncas,) * 4)
            a11a = a11a + a11a.transpose(1, 0, 2, 3)
            a11a = a11a + a11a.transpose(0, 1, 3, 2)
            h2 = aa11 + a11a
        ecore = float((xp.einsum("pq,pq->", h1e_mo, ddm) + xp.einsum("pq,pq->", _to_xp_f64(eris.vhf_c, xp), ddm)).item())
        ci1, g = self.solve_approx_ci(h1, h2, fcivec, ecore, float(e_cas), envs)
        if g is not None:
            fcivec_xp = _to_xp_f64(fcivec, xp).ravel()
            ci1_xp = _to_xp_f64(ci1, xp).ravel()
            g_xp = _to_xp_f64(g, xp).ravel()
            ovlp = float((xp.dot(fcivec_xp, ci1_xp)).item())
            norm_g = float(xp.linalg.norm(g_xp).item())
            if 1.0 - abs(ovlp) > norm_g * float(getattr(self, "ci_grad_trust_region", 3.0)):
                ci1 = fcivec_xp + g_xp
                ci1 *= 1.0 / max(float(xp.linalg.norm(ci1).item()), 1e-16)
        rdm_kwargs: dict[str, Any] = {}
        if xp is not np:
            rdm_kwargs["rdm_backend"] = "cuda"
            rdm_kwargs["return_cupy"] = True
            rdm_kwargs["strict_gpu"] = True
        casdm1, casdm2 = self.fcisolver.make_rdm12(ci1, ncas, self.nelecas, **rdm_kwargs)
        return _to_xp_f64(casdm1, xp), _to_xp_f64(casdm2, xp), (None if g is None else _to_xp_f64(g, xp)), _to_xp_f64(ci1, xp)
