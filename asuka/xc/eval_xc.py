"""Native GPU XC evaluator for Minnesota meta-GGA functionals.

Implements M06-L, M06, M06-2X, MN15 using forward-mode automatic
differentiation on CuPy arrays.  No libxc dependency, no GPU-CPU
round-trips — all computation stays on GPU.

Convention (matching libxc unpolarized output):
  exc      = XC energy per electron
  vrho     = d(rho * exc) / d(rho)
  vsigma   = d(rho * exc) / d(sigma)
  vtau     = d(rho * exc) / d(tau)
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from .functional import FunctionalSpec
from . import minnesota as P

# ── Physical constants ─────────────────────────────────────────────────
_CBRT2 = 2.0 ** (1.0 / 3.0)  # 1.2599210498948732
_CBRT2_SQ = _CBRT2 * _CBRT2  # 2^(2/3)
_K_FACTOR_C = 0.3 * (6.0 * math.pi**2) ** (2.0 / 3.0)
_X_FACTOR_C = 0.375 * (3.0 / math.pi) ** (1.0 / 3.0) * 4.0 ** (2.0 / 3.0)
_RS_FACTOR = (3.0 / (4.0 * math.pi)) ** (1.0 / 3.0)
_X2S = 1.0 / (2.0 * (6.0 * math.pi**2) ** (1.0 / 3.0))
_CX = -0.75 * (3.0 / math.pi) ** (1.0 / 3.0)  # LDA exchange prefactor
_PBE_KAPPA = 0.804
_PBE_MU = 0.2195149727645171
_PBE_GAMMA = (1.0 - math.log(2.0)) / math.pi**2
_PBE_BETA = 0.06672455060314922
_RHO_CUTOFF = 1e-20
_TAU_CUTOFF = 1e-30


# ── Forward-mode AD tangent class ──────────────────────────────────────
class _T:
    """Tangent number tracking d/d(rho), d/d(sigma), d/d(tau)."""
    __slots__ = ("v", "dr", "ds", "dt")

    def __init__(self, v, dr, ds, dt):
        self.v = v
        self.dr = dr
        self.ds = ds
        self.dt = dt

    # arithmetic
    def __add__(self, o):
        if isinstance(o, _T):
            return _T(self.v + o.v, self.dr + o.dr, self.ds + o.ds, self.dt + o.dt)
        return _T(self.v + o, self.dr, self.ds, self.dt)

    __radd__ = __add__

    def __neg__(self):
        return _T(-self.v, -self.dr, -self.ds, -self.dt)

    def __sub__(self, o):
        if isinstance(o, _T):
            return _T(self.v - o.v, self.dr - o.dr, self.ds - o.ds, self.dt - o.dt)
        return _T(self.v - o, self.dr, self.ds, self.dt)

    def __rsub__(self, o):
        return _T(o - self.v, -self.dr, -self.ds, -self.dt)

    def __mul__(self, o):
        if isinstance(o, _T):
            return _T(
                self.v * o.v,
                self.dr * o.v + self.v * o.dr,
                self.ds * o.v + self.v * o.ds,
                self.dt * o.v + self.v * o.dt,
            )
        return _T(self.v * o, self.dr * o, self.ds * o, self.dt * o)

    __rmul__ = __mul__

    def __truediv__(self, o):
        if isinstance(o, _T):
            inv = 1.0 / o.v
            inv2 = inv * inv
            return _T(
                self.v * inv,
                (self.dr * o.v - self.v * o.dr) * inv2,
                (self.ds * o.v - self.v * o.ds) * inv2,
                (self.dt * o.v - self.v * o.dt) * inv2,
            )
        inv = 1.0 / o
        return _T(self.v * inv, self.dr * inv, self.ds * inv, self.dt * inv)

    def __rtruediv__(self, o):
        inv = 1.0 / self.v
        inv2 = inv * inv
        return _T(o * inv, -o * self.dr * inv2, -o * self.ds * inv2, -o * self.dt * inv2)

    def __pow__(self, n):
        vn = self.v ** n
        dvn = n * self.v ** (n - 1)
        return _T(vn, dvn * self.dr, dvn * self.ds, dvn * self.dt)


def _t_sqrt(x):
    xp = _xp(x.v)
    v = xp.sqrt(x.v)
    d = 0.5 / v
    return _T(v, d * x.dr, d * x.ds, d * x.dt)


def _t_log(x):
    xp = _xp(x.v)
    d = 1.0 / x.v
    return _T(xp.log(x.v), d * x.dr, d * x.ds, d * x.dt)


def _t_exp(x):
    xp = _xp(x.v)
    v = xp.exp(x.v)
    return _T(v, v * x.dr, v * x.ds, v * x.dt)


def _t_const(val, like):
    """Create a constant _T (zero derivatives) shaped like *like*."""
    xp = _xp(like)
    z = xp.zeros_like(like)
    if not isinstance(val, type(like)):
        val = xp.full_like(like, val)
    return _T(val, z, z, z)


# ── Array-module helper ────────────────────────────────────────────────
def _xp(arr):
    tp = type(arr).__module__
    if tp.startswith("cupy"):
        import cupy
        return cupy
    return np


# ── Building blocks ────────────────────────────────────────────────────

def _pw_g(k, rs_t):
    """Perdew-Wang LDA correlation component g(k, rs) as Tangent."""
    A = P.PW_A[k]
    a1 = P.PW_ALPHA1[k]
    b1 = P.PW_BETA1[k]
    b2 = P.PW_BETA2[k]
    b3 = P.PW_BETA3[k]
    b4 = P.PW_BETA4[k]
    rs12 = _t_sqrt(rs_t)
    G = b1 * rs12 + b2 * rs_t + b3 * rs_t * rs12 + b4 * rs_t * rs_t
    return -2.0 * A * (1.0 + a1 * rs_t) * _t_log(1.0 + 1.0 / (2.0 * A * G))


def _pw_ec_unpol(rs_t):
    """PW-LDA correlation for unpolarized gas: f_pw(rs, z=0) = g(0, rs)."""
    return _pw_g(0, rs_t)


def _stoll_decomp(rs_t):
    """Stoll decomposition for z=0. Returns (par, perp) as Tangents."""
    ec0 = _pw_g(0, rs_t)
    ec1 = _pw_g(1, rs_t * _CBRT2)
    return ec1 * 0.5, ec0 - ec1


def _series_w(coeffs, ts_t):
    """Sum a[i]*w^i for i=0..n-1 where w = (K - ts)/(K + ts)."""
    w = (_K_FACTOR_C - ts_t) / (_K_FACTOR_C + ts_t)
    result = _t_const(coeffs[0], ts_t.v)
    wi = _t_const(1.0, ts_t.v)
    for i in range(1, len(coeffs)):
        wi = wi * w
        result = result + coeffs[i] * wi
    return result


def _gvt4(alpha, dd, xs2_t, z_t):
    """GVT4 function. xs2_t = x^2, z_t = 2*(ts-K)."""
    g = 1.0 + alpha * (xs2_t + z_t)
    g2 = g * g
    g3 = g2 * g
    A = dd[0] / g
    B = (dd[1] * xs2_t + dd[2] * z_t) / g2
    C = (dd[3] * xs2_t * xs2_t + dd[4] * xs2_t * z_t + dd[5] * z_t * z_t) / g3
    return A + B + C


def _pbe_Fx(xs2_t):
    """PBE exchange enhancement factor F(xs2) where xs2 = x_sigma^2."""
    s2 = _X2S * _X2S * xs2_t
    denom = _PBE_KAPPA + _PBE_MU * s2
    return 1.0 + _PBE_KAPPA - _PBE_KAPPA * _PBE_KAPPA / denom


def _b97_g(gamma, cc, xs2_arg):
    """B97 g-function. cc has 5 coefficients. xs2_arg = x^2 passed to u."""
    u = gamma * xs2_arg / (1.0 + gamma * xs2_arg)
    result = _t_const(cc[0], xs2_arg.v)
    ui = _t_const(1.0, xs2_arg.v)
    for i in range(1, 5):
        ui = ui * u
        result = result + cc[i] * ui
    return result


def _fermi_D(xs2_t, ts_t):
    """Fermi hole curvature D = 1 - xs2/(8*ts)."""
    return 1.0 - xs2_t / (8.0 * ts_t)


def _fermi_D_corrected(xs2_t, ts_t, cnst):
    """Corrected Fermi D = D * (1 - exp(-4*ts^2/cnst^2))."""
    D = _fermi_D(xs2_t, ts_t)
    cnst2 = cnst * cnst
    correction = 1.0 - _t_exp(-4.0 * ts_t * ts_t / cnst2)
    return D * correction


def _pbe_ec_unpol(rs_t, xt2_t):
    """PBE correlation for unpolarized: f_pw + fH."""
    ec = _pw_ec_unpol(rs_t)
    # t = xt / (4 * 2^(1/3) * phi * sqrt(rs)), phi=1 for z=0
    rs12 = _t_sqrt(rs_t)
    t2 = xt2_t / (16.0 * _CBRT2_SQ * rs_t)  # t^2 = xt^2 / (16 * 2^(2/3) * rs)
    # A = beta / (gamma * (exp(-ec/(gamma)) - 1))
    exp_arg = -1.0 * ec / _PBE_GAMMA
    exp_val = _t_exp(exp_arg)
    A = _PBE_BETA / (_PBE_GAMMA * (exp_val - 1.0))
    # f1 = t^2 + A*t^4
    f1 = t2 + A * t2 * t2
    # f2 = beta*f1 / (gamma*(1 + A*f1))
    f2 = _PBE_BETA * f1 / (_PBE_GAMMA * (1.0 + A * f1))
    fH = _PBE_GAMMA * _t_log(1.0 + f2)
    return ec + fH


# ── M06 exchange ───────────────────────────────────────────────────────

def _m06_exchange(rho_t, sigma_t, tau_t, xa, xd, alpha):
    """M06 family exchange (for unpolarized). Returns E_x = rho * exc_x."""
    xp = _xp(rho_t.v)
    # Per-spin reduced variables
    xs2 = sigma_t * _CBRT2_SQ / rho_t ** (8.0 / 3.0)
    ts = tau_t * _CBRT2_SQ / rho_t ** (5.0 / 3.0)
    # Enhancement factor
    F_pbe = _pbe_Fx(xs2)
    S_w = _series_w(xa, ts)
    z_gvt4 = 2.0 * (ts - _K_FACTOR_C)
    G = _gvt4(alpha, xd, xs2, z_gvt4)
    F_total = F_pbe * S_w + G
    # E_x = CX * rho^(4/3) * F_total  (energy per unit volume)
    E_x = _CX * rho_t ** (4.0 / 3.0) * F_total
    return E_x


# ── M06 correlation ───────────────────────────────────────────────────

def _m06_correlation(rho_t, sigma_t, tau_t, css, cab, dss, dab,
                     gamma_ss, gamma_ab, alpha_ss, alpha_ab, fermi_cnst):
    """M06 family correlation (unpolarized). Returns E_c = rho * exc_c."""
    xp = _xp(rho_t.v)
    rs = _RS_FACTOR / rho_t ** (1.0 / 3.0)
    xs2 = sigma_t * _CBRT2_SQ / rho_t ** (8.0 / 3.0)
    ts = tau_t * _CBRT2_SQ / rho_t ** (5.0 / 3.0)

    par, perp = _stoll_decomp(rs)
    z_ss = 2.0 * (ts - _K_FACTOR_C)

    # M05 part: parallel
    g_ss = _b97_g(gamma_ss, css, xs2)
    D_corr = _fermi_D_corrected(xs2, ts, fermi_cnst)
    m05_par = 2.0 * par * g_ss * D_corr

    # M05 part: perpendicular (xs_ab^2 = 2*xs2 for z=0)
    g_ab = _b97_g(gamma_ab, cab, 2.0 * xs2)
    m05_perp = perp * g_ab

    # VSXC part: parallel
    gvt4_ss = _gvt4(alpha_ss, dss, xs2, z_ss)
    D_plain = _fermi_D(xs2, ts)
    vsxc_par = 2.0 * par * gvt4_ss * D_plain

    # VSXC part: perpendicular (xs_ab^2 = 2*xs2, z_ab = 4*(ts-K))
    z_ab = 4.0 * (ts - _K_FACTOR_C)
    gvt4_ab = _gvt4(alpha_ab, dab, 2.0 * xs2, z_ab)
    vsxc_perp = perp * gvt4_ab

    E_c = rho_t * (m05_par + m05_perp + vsxc_par + vsxc_perp)
    return E_c


# ── MN15 exchange ──────────────────────────────────────────────────────

def _mn15_exchange(rho_t, sigma_t, tau_t, cc):
    """MN15 exchange (MN12-type, unpolarized). Returns E_x = rho * exc_x."""
    xp = _xp(rho_t.v)
    xs2 = sigma_t * _CBRT2_SQ / rho_t ** (8.0 / 3.0)
    ts = tau_t * _CBRT2_SQ / rho_t ** (5.0 / 3.0)

    # ux = gamma*xs2 / (1 + gamma*xs2)
    gamma = P.MN15_X_GAMMA
    ux = gamma * xs2 / (1.0 + gamma * xs2)

    # vx = 1 / (1 + 2^(1/3)*0.4 / rho^(1/3))
    rho13 = rho_t ** (1.0 / 3.0)
    vx = 1.0 / (1.0 + 0.4 * _CBRT2 / rho13)

    # w = (K - ts) / (K + ts)
    w = (_K_FACTOR_C - ts) / (_K_FACTOR_C + ts)

    # Build 10 polynomials in w, then combine with powers of ux, vx
    # cc is 0-indexed (40 coefficients)
    c = cc  # alias

    def _wpoly(start, n):
        """Polynomial in w: c[start] + c[start+1]*w + ... + c[start+n-1]*w^(n-1)."""
        result = _t_const(c[start], w.v)
        wi = _t_const(1.0, w.v)
        for i in range(1, n):
            wi = wi * w
            result = result + c[start + i] * wi
        return result

    p1 = _wpoly(0, 6)   # CC000..CC005
    p2 = _wpoly(6, 5)   # CC010..CC014
    p3 = _wpoly(11, 4)  # CC020..CC023
    p4 = _wpoly(15, 3)  # CC030..CC032
    p5 = _wpoly(18, 5)  # CC100..CC104
    p6 = _wpoly(23, 4)  # CC110..CC113
    p7 = _wpoly(27, 3)  # CC120..CC122
    p8 = _wpoly(30, 4)  # CC200..CC203
    p9 = _wpoly(34, 3)  # CC210..CC212
    p10 = _wpoly(37, 3) # CC300..CC302

    ux2 = ux * ux
    ux3 = ux2 * ux
    vx2 = vx * vx
    vx3 = vx2 * vx

    F = (p1 + p2 * ux + p3 * ux2 + p4 * ux3
         + p5 * vx + p6 * ux * vx + p7 * ux2 * vx
         + p8 * vx2 + p9 * ux * vx2 + p10 * vx3)

    E_x = _CX * rho_t ** (4.0 / 3.0) * F
    return E_x


# ── MN15 correlation ──────────────────────────────────────────────────

def _mn15_correlation(rho_t, sigma_t, tau_t, m08a, m08b):
    """MN15 correlation (M08-type, unpolarized). Returns E_c = rho * exc_c."""
    xp = _xp(rho_t.v)
    ts = tau_t * _CBRT2_SQ / rho_t ** (5.0 / 3.0)
    rs = _RS_FACTOR / rho_t ** (1.0 / 3.0)
    # xt^2 = sigma / rho^(8/3)  (total reduced gradient squared)
    xt2 = sigma_t / rho_t ** (8.0 / 3.0)

    S_a = _series_w(m08a, ts)
    S_b = _series_w(m08b, ts)

    ec_pw = _pw_ec_unpol(rs)
    ec_pbe = _pbe_ec_unpol(rs, xt2)

    # m08_f = S_a * ec_pw + S_b * (ec_pbe - ec_pw)
    exc_c = S_a * ec_pw + S_b * (ec_pbe - ec_pw)
    return rho_t * exc_c


# ── Parameter dispatch ─────────────────────────────────────────────────

def _get_xc_parts(name):
    """Return (exchange_fn, correlation_fn) callables for the named functional."""
    name = name.lower()

    if name == "m06-l":
        def ex(r, s, t): return _m06_exchange(r, s, t, P.M06L_X_A, P.M06L_X_D, P.M06_X_ALPHA)
        def ec(r, s, t): return _m06_correlation(r, s, t,
            P.M06L_C_CSS, P.M06L_C_CAB, P.M06L_C_DSS, P.M06L_C_DAB,
            P.M06_C_GAMMA_SS, P.M06_C_GAMMA_AB,
            P.M06_C_ALPHA_SS, P.M06_C_ALPHA_AB, P.M06_C_FERMI_D_CNST)
        return ex, ec

    if name == "m06":
        def ex(r, s, t): return _m06_exchange(r, s, t, P.M06_X_A, P.M06_X_D, P.M06_X_ALPHA)
        def ec(r, s, t): return _m06_correlation(r, s, t,
            P.M06_C_CSS, P.M06_C_CAB, P.M06_C_DSS, P.M06_C_DAB,
            P.M06_C_GAMMA_SS, P.M06_C_GAMMA_AB,
            P.M06_C_ALPHA_SS, P.M06_C_ALPHA_AB, P.M06_C_FERMI_D_CNST)
        return ex, ec

    if name == "m06-2x":
        def ex(r, s, t): return _m06_exchange(r, s, t, P.M06_2X_X_A, P.M06_2X_X_D, 0.0)
        def ec(r, s, t): return _m06_correlation(r, s, t,
            P.M06_2X_C_CSS, P.M06_2X_C_CAB, P.M06_2X_C_DSS, P.M06_2X_C_DAB,
            P.M06_C_GAMMA_SS, P.M06_C_GAMMA_AB,
            P.M06_C_ALPHA_SS, P.M06_C_ALPHA_AB, P.M06_C_FERMI_D_CNST)
        return ex, ec

    if name == "mn15":
        def ex(r, s, t): return _mn15_exchange(r, s, t, P.MN15_X_CC)
        def ec(r, s, t): return _mn15_correlation(r, s, t, P.MN15_C_A, P.MN15_C_B)
        return ex, ec

    raise ValueError(f"Unknown functional: {name!r}")


# ── Main entry point ───────────────────────────────────────────────────

def eval_xc(
    spec: FunctionalSpec,
    rho: Any,
    sigma: Any,
    tau: Any,
    *,
    deriv: int = 1,
) -> tuple[Any, Any, Any, Any]:
    """Evaluate XC energy density and potentials on a grid.

    Parameters
    ----------
    spec : FunctionalSpec
    rho, sigma, tau : arrays (npt,) — total density, |grad rho|^2, KE density
    deriv : int — only 1 is supported

    Returns
    -------
    exc, vrho, vsigma, vtau : arrays (npt,)
    """
    if deriv != 1:
        raise NotImplementedError("Only deriv=1 is supported")

    xp = _xp(rho)
    rho_safe = xp.maximum(rho, _RHO_CUTOFF)
    sigma_safe = xp.maximum(sigma, 0.0)
    tau_safe = xp.maximum(tau, _TAU_CUTOFF)

    z = xp.zeros_like(rho_safe)
    one = xp.ones_like(rho_safe)

    rho_t = _T(rho_safe, one, z, z)
    sigma_t = _T(sigma_safe, z, one, z)
    tau_t = _T(tau_safe, z, z, one)

    ex_fn, ec_fn = _get_xc_parts(spec.name)

    # E_x = rho * exc_x  (energy per unit volume)
    E_x = ex_fn(rho_t, sigma_t, tau_t)
    E_c = ec_fn(rho_t, sigma_t, tau_t)
    E = E_x + E_c

    # Mask out low-density regions
    mask = rho > _RHO_CUTOFF
    exc = xp.where(mask, E.v / rho_safe, 0.0)
    vrho = xp.where(mask, E.dr, 0.0)
    vsigma = xp.where(mask, E.ds, 0.0)
    vtau = xp.where(mask, E.dt, 0.0)

    return exc, vrho, vsigma, vtau


__all__ = ["eval_xc"]
