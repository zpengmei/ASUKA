"""Native GPU XC evaluator for Minnesota meta-GGA functionals.

Implements M06-L, M06, M06-2X, MN15 using forward-mode automatic
differentiation on CuPy arrays. No GPU-CPU
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


# ── Forward-mode AD tangent class (spin-polarized) ──────────────────────
class _T7:
    """Tangent number tracking derivatives for spin-polarized meta-GGA inputs.

    Independent variables (derivative vector ordering):
      0 rho_a
      1 rho_b
      2 sigma_aa
      3 sigma_ab
      4 sigma_bb
      5 tau_a
      6 tau_b
    """

    __slots__ = ("v", "d")

    def __init__(self, v, d):
        self.v = v
        self.d = d

    def __add__(self, o):
        if isinstance(o, _T7):
            return _T7(self.v + o.v, self.d + o.d)
        return _T7(self.v + o, self.d)

    __radd__ = __add__

    def __neg__(self):
        return _T7(-self.v, -self.d)

    def __sub__(self, o):
        if isinstance(o, _T7):
            return _T7(self.v - o.v, self.d - o.d)
        return _T7(self.v - o, self.d)

    def __rsub__(self, o):
        return _T7(o - self.v, -self.d)

    def __mul__(self, o):
        if isinstance(o, _T7):
            return _T7(self.v * o.v, self.d * o.v + self.v * o.d)
        return _T7(self.v * o, self.d * o)

    __rmul__ = __mul__

    def __truediv__(self, o):
        if isinstance(o, _T7):
            inv = 1.0 / o.v
            inv2 = inv * inv
            return _T7(self.v * inv, (self.d * o.v - self.v * o.d) * inv2)
        inv = 1.0 / o
        return _T7(self.v * inv, self.d * inv)

    def __rtruediv__(self, o):
        inv = 1.0 / self.v
        inv2 = inv * inv
        return _T7(o * inv, -o * self.d * inv2)

    def __pow__(self, n):
        vn = self.v ** n
        dvn = n * self.v ** (n - 1)
        return _T7(vn, dvn * self.d)


def _t7_sqrt(x: _T7) -> _T7:
    xp = _xp(x.v)
    v = xp.sqrt(x.v)
    d = x.d * (0.5 / v)
    return _T7(v, d)


def _t7_log(x: _T7) -> _T7:
    xp = _xp(x.v)
    d = x.d * (1.0 / x.v)
    return _T7(xp.log(x.v), d)


def _t7_exp(x: _T7) -> _T7:
    xp = _xp(x.v)
    v = xp.exp(x.v)
    return _T7(v, v * x.d)


def _t7_const(val: float, like) -> _T7:
    """Create a constant _T7 (zero derivatives) shaped like *like*."""
    xp = _xp(like)
    v = xp.full_like(like, val, dtype=like.dtype)
    d = xp.zeros((7,) + tuple(like.shape), dtype=like.dtype)
    return _T7(v, d)


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
    """Fermi hole curvature D = max(0, 1 - xs2/(8*ts)).

    Clamped to [0, 1] to handle unphysical inputs where tau < tau_W
    (von Weizsäcker bound). libxc applies the same clamp.
    """
    xp = _xp(ts_t.v)
    D = 1.0 - xs2_t / (8.0 * ts_t)
    # Clamp: replace negative values with zero (preserving zero derivative at boundary).
    D_val = xp.maximum(D.v, 0.0)
    mask = D.v > 0.0
    D_dr = xp.where(mask, D.dr, 0.0)
    D_ds = xp.where(mask, D.ds, 0.0)
    D_dt = xp.where(mask, D.dt, 0.0)
    return _T(D_val, D_dr, D_ds, D_dt)


def _fermi_D_corrected(xs2_t, ts_t, cnst):
    """Corrected Fermi D = max(0, D) * (1 - exp(-4*ts^2/cnst^2))."""
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


# ── Spin-polarized (PW92 / PBE) building blocks ────────────────────────

_FZ_DENOM = 2.0 ** (4.0 / 3.0) - 2.0


def _pw_g7(k: int, rs_t: _T7) -> _T7:
    """Perdew-Wang correlation component g(k, rs) as _T7."""
    A = P.PW_A[k]
    a1 = P.PW_ALPHA1[k]
    b1 = P.PW_BETA1[k]
    b2 = P.PW_BETA2[k]
    b3 = P.PW_BETA3[k]
    b4 = P.PW_BETA4[k]
    rs12 = _t7_sqrt(rs_t)
    G = b1 * rs12 + b2 * rs_t + b3 * rs_t * rs12 + b4 * rs_t * rs_t
    return -2.0 * A * (1.0 + a1 * rs_t) * _t7_log(1.0 + 1.0 / (2.0 * A * G))


def _pw_ec_pw92_spin(rs_t: _T7, zeta_t: _T7) -> _T7:
    """PW92 LSDA correlation energy per electron for arbitrary spin polarization.

    Uses the standard PW92 spin interpolation with the spin-stiffness term.
    """
    xp = _xp(rs_t.v)
    ec0 = _pw_g7(0, rs_t)
    ec1 = _pw_g7(1, rs_t)
    ec2 = _pw_g7(2, rs_t)

    one = _t7_const(1.0, rs_t.v)
    two = _t7_const(2.0, rs_t.v)
    z = zeta_t
    fz = ((one + z) ** (4.0 / 3.0) + (one - z) ** (4.0 / 3.0) - two) / _FZ_DENOM
    z4 = z * z
    z4 = z4 * z4  # zeta^4
    # Libxc PW92 spin interpolation:
    #   ec = ec0 + f(zeta) * [ (ec1-ec0)*zeta^4 - ec2*(1-zeta^4)/f''(0) ].
    return ec0 + fz * ((ec1 - ec0) * z4 - ec2 * (one - z4) / P.PW_FZ20)


def _phi_zeta7(zeta_t: _T7) -> _T7:
    """PBE phi(zeta) spin-scaling factor."""
    one = _t7_const(1.0, zeta_t.v)
    return 0.5 * ((one + zeta_t) ** (2.0 / 3.0) + (one - zeta_t) ** (2.0 / 3.0))


def _pbe_ec_spin7(rs_t: _T7, zeta_t: _T7, xt2_t: _T7) -> _T7:
    """PBE correlation for arbitrary spin polarization: ec_pw92 + H."""
    ec = _pw_ec_pw92_spin(rs_t, zeta_t)
    phi = _phi_zeta7(zeta_t)
    # t^2 = xt^2 / (16 * 2^(2/3) * rs * phi^2)
    t2 = xt2_t / (16.0 * _CBRT2_SQ * rs_t * phi * phi)

    phi3 = phi * phi * phi
    gamma_phi = _PBE_GAMMA * phi3
    exp_arg = -1.0 * ec / gamma_phi
    exp_val = _t7_exp(exp_arg)
    beta_over_gamma = _PBE_BETA / _PBE_GAMMA
    # Note: PBE spin formula keeps beta/gamma (no phi^3 factor); phi^3 only
    # enters via the gamma*phi^3 prefactor and the exponent argument.
    A = beta_over_gamma / (exp_val - 1.0)
    f1 = t2 + A * t2 * t2
    f2 = beta_over_gamma * f1 / (1.0 + A * f1)
    H = gamma_phi * _t7_log(1.0 + f2)
    return ec + H


def _series_w7(coeffs, ts_t: _T7) -> _T7:
    """Sum a[i]*w^i for i=0..n-1 where w = (K - ts)/(K + ts)."""
    w = (_K_FACTOR_C - ts_t) / (_K_FACTOR_C + ts_t)
    result = _t7_const(coeffs[0], ts_t.v)
    wi = _t7_const(1.0, ts_t.v)
    for i in range(1, len(coeffs)):
        wi = wi * w
        result = result + coeffs[i] * wi
    return result


def _gvt4_7(alpha: float, dd, xs2_t: _T7, z_t: _T7) -> _T7:
    """GVT4 function (spin-polarized tangent)."""
    g = 1.0 + alpha * (xs2_t + z_t)
    g2 = g * g
    g3 = g2 * g
    A = dd[0] / g
    B = (dd[1] * xs2_t + dd[2] * z_t) / g2
    C = (dd[3] * xs2_t * xs2_t + dd[4] * xs2_t * z_t + dd[5] * z_t * z_t) / g3
    return A + B + C


def _pbe_Fx7(xs2_t: _T7) -> _T7:
    """PBE exchange enhancement factor for _T7."""
    s2 = _X2S * _X2S * xs2_t
    denom = _PBE_KAPPA + _PBE_MU * s2
    return 1.0 + _PBE_KAPPA - _PBE_KAPPA * _PBE_KAPPA / denom


def _b97_g7(gamma: float, cc, xs2_arg: _T7) -> _T7:
    """B97 g-function for _T7."""
    u = gamma * xs2_arg / (1.0 + gamma * xs2_arg)
    result = _t7_const(cc[0], xs2_arg.v)
    ui = _t7_const(1.0, xs2_arg.v)
    for i in range(1, 5):
        ui = ui * u
        result = result + cc[i] * ui
    return result


def _fermi_D7(xs2_t: _T7, ts_t: _T7) -> _T7:
    """Fermi hole curvature D = max(0, 1 - xs2/(8*ts)) for _T7."""
    xp = _xp(ts_t.v)
    D = 1.0 - xs2_t / (8.0 * ts_t)
    D_val = xp.maximum(D.v, 0.0)
    mask = D.v > 0.0
    D_d = xp.where(mask[None, ...], D.d, 0.0)
    return _T7(D_val, D_d)


def _fermi_D_corrected7(xs2_t: _T7, ts_t: _T7, cnst: float) -> _T7:
    """Corrected Fermi D = max(0, D) * (1 - exp(-4*ts^2/cnst^2))."""
    D = _fermi_D7(xs2_t, ts_t)
    cnst2 = cnst * cnst
    correction = 1.0 - _t7_exp(-4.0 * ts_t * ts_t / cnst2)
    return D * correction


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


# ── Spin-polarized Minnesota XC (full UKS) ──────────────────────────────

def _m06_exchange7(rho_t: _T7, sigma_t: _T7, tau_t: _T7, xa, xd, alpha: float) -> _T7:
    """M06 family exchange (unpolarized kernel) evaluated on _T7."""
    xs2 = sigma_t * _CBRT2_SQ / rho_t ** (8.0 / 3.0)
    ts = tau_t * _CBRT2_SQ / rho_t ** (5.0 / 3.0)
    F_pbe = _pbe_Fx7(xs2)
    S_w = _series_w7(xa, ts)
    z_gvt4 = 2.0 * (ts - _K_FACTOR_C)
    G = _gvt4_7(alpha, xd, xs2, z_gvt4)
    F_total = F_pbe * S_w + G
    return _CX * rho_t ** (4.0 / 3.0) * F_total


def _mn15_exchange7(rho_t: _T7, sigma_t: _T7, tau_t: _T7, cc) -> _T7:
    """MN15 exchange (unpolarized kernel) evaluated on _T7."""
    xs2 = sigma_t * _CBRT2_SQ / rho_t ** (8.0 / 3.0)
    ts = tau_t * _CBRT2_SQ / rho_t ** (5.0 / 3.0)

    gamma = P.MN15_X_GAMMA
    ux = gamma * xs2 / (1.0 + gamma * xs2)
    rho13 = rho_t ** (1.0 / 3.0)
    vx = 1.0 / (1.0 + 0.4 * _CBRT2 / rho13)
    w = (_K_FACTOR_C - ts) / (_K_FACTOR_C + ts)

    c = cc

    def _wpoly(start: int, n: int) -> _T7:
        result = _t7_const(c[start], w.v)
        wi = _t7_const(1.0, w.v)
        for i in range(1, n):
            wi = wi * w
            result = result + c[start + i] * wi
        return result

    p1 = _wpoly(0, 6)
    p2 = _wpoly(6, 5)
    p3 = _wpoly(11, 4)
    p4 = _wpoly(15, 3)
    p5 = _wpoly(18, 5)
    p6 = _wpoly(23, 4)
    p7 = _wpoly(27, 3)
    p8 = _wpoly(30, 4)
    p9 = _wpoly(34, 3)
    p10 = _wpoly(37, 3)

    ux2 = ux * ux
    ux3 = ux2 * ux
    vx2 = vx * vx
    vx3 = vx2 * vx

    F = (p1 + p2 * ux + p3 * ux2 + p4 * ux3
         + p5 * vx + p6 * ux * vx + p7 * ux2 * vx
         + p8 * vx2 + p9 * ux * vx2 + p10 * vx3)
    return _CX * rho_t ** (4.0 / 3.0) * F


def _m06_correlation_spin7(
    rho_a: _T7,
    sigma_aa: _T7,
    tau_a: _T7,
    rho_b: _T7,
    sigma_bb: _T7,
    tau_b: _T7,
    sigma_ab: _T7,
    *,
    css,
    cab,
    dss,
    dab,
    gamma_ss: float,
    gamma_ab: float,
    alpha_ss: float,
    alpha_ab: float,
    fermi_cnst: float,
) -> _T7:
    """M06 family correlation for arbitrary spin polarization (full UKS form)."""
    rho = rho_a + rho_b
    rs = _RS_FACTOR / rho ** (1.0 / 3.0)
    zeta = (rho_a - rho_b) / rho
    ec_lsda = _pw_ec_pw92_spin(rs, zeta)

    rs_a = _RS_FACTOR / rho_a ** (1.0 / 3.0)
    rs_b = _RS_FACTOR / rho_b ** (1.0 / 3.0)
    ec_pol_a = _pw_g7(1, rs_a)
    ec_pol_b = _pw_g7(1, rs_b)

    wa = rho_a / rho
    wb = rho_b / rho
    perp = ec_lsda - wa * ec_pol_a - wb * ec_pol_b

    # Same-spin reduced variables (x_sigma^2, t_sigma) and opposite-spin invariants.
    xs2_a = sigma_aa / rho_a ** (8.0 / 3.0)
    xs2_b = sigma_bb / rho_b ** (8.0 / 3.0)
    ts_a = tau_a / rho_a ** (5.0 / 3.0)
    ts_b = tau_b / rho_b ** (5.0 / 3.0)
    # Opposite-spin reduced gradient: libxc Minnesota convention uses a simple
    # sum of the per-spin reduced gradients (no sigma_ab coupling).
    # For zeta=0, xs2_a == xs2_b == xs2 -> xs2_ab == 2*xs2.
    xs2_ab = xs2_a + xs2_b
    ts_ab = ts_a + ts_b

    # M05 part: same-spin
    g_ss_a = _b97_g7(gamma_ss, css, xs2_a)
    g_ss_b = _b97_g7(gamma_ss, css, xs2_b)
    D_corr_a = _fermi_D_corrected7(xs2_a, ts_a, fermi_cnst)
    D_corr_b = _fermi_D_corrected7(xs2_b, ts_b, fermi_cnst)
    m05_par_a = ec_pol_a * g_ss_a * D_corr_a
    m05_par_b = ec_pol_b * g_ss_b * D_corr_b

    # M05 part: opposite-spin
    g_ab = _b97_g7(gamma_ab, cab, xs2_ab)
    m05_perp = perp * g_ab

    # VSXC part: same-spin
    z_ss_a = 2.0 * (ts_a - _K_FACTOR_C)
    z_ss_b = 2.0 * (ts_b - _K_FACTOR_C)
    gvt4_ss_a = _gvt4_7(alpha_ss, dss, xs2_a, z_ss_a)
    gvt4_ss_b = _gvt4_7(alpha_ss, dss, xs2_b, z_ss_b)
    D_plain_a = _fermi_D7(xs2_a, ts_a)
    D_plain_b = _fermi_D7(xs2_b, ts_b)
    vsxc_par_a = ec_pol_a * gvt4_ss_a * D_plain_a
    vsxc_par_b = ec_pol_b * gvt4_ss_b * D_plain_b

    # VSXC part: opposite-spin  (reduces to z_ab=4*(ts-K) for zeta=0)
    z_ab = 2.0 * (ts_ab - 2.0 * _K_FACTOR_C)
    gvt4_ab = _gvt4_7(alpha_ab, dab, xs2_ab, z_ab)
    vsxc_perp = perp * gvt4_ab

    return rho_a * (m05_par_a + vsxc_par_a) + rho_b * (m05_par_b + vsxc_par_b) + rho * (m05_perp + vsxc_perp)


def _mn15_correlation_spin7(
    rho_a: _T7,
    sigma_aa: _T7,
    tau_a: _T7,
    rho_b: _T7,
    sigma_bb: _T7,
    tau_b: _T7,
    sigma_ab: _T7,
    *,
    m08a,
    m08b,
) -> _T7:
    """MN15 correlation (M08-type) generalized to arbitrary spin polarization."""
    rho = rho_a + rho_b
    zeta = (rho_a - rho_b) / rho
    rs = _RS_FACTOR / rho ** (1.0 / 3.0)

    # total reduced gradient squared (PBE-style)
    sigma = sigma_aa + sigma_bb + 2.0 * sigma_ab
    xt2 = sigma / rho ** (8.0 / 3.0)

    # MN15 correlation uses a total-density reduced tau (libxc convention).
    tau = tau_a + tau_b
    ts = tau * _CBRT2_SQ / rho ** (5.0 / 3.0)

    S_a = _series_w7(m08a, ts)
    S_b = _series_w7(m08b, ts)

    ec_pw = _pw_ec_pw92_spin(rs, zeta)
    ec_pbe = _pbe_ec_spin7(rs, zeta, xt2)

    exc_c = S_a * ec_pw + S_b * (ec_pbe - ec_pw)
    return rho * exc_c


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


def eval_xc_sp(
    spec: FunctionalSpec,
    rho_a: Any,
    rho_b: Any,
    sigma_aa: Any,
    sigma_ab: Any,
    sigma_bb: Any,
    tau_a: Any,
    tau_b: Any,
    *,
    deriv: int = 1,
) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
    """Evaluate spin-polarized XC (UKS ingredients) via forward-mode AD.

    Parameters
    ----------
    spec : FunctionalSpec
    rho_a, rho_b : (npt,) alpha/beta spin densities
    sigma_aa, sigma_ab, sigma_bb : (npt,) gradient invariants:
        sigma_aa = ∇rho_a·∇rho_a, sigma_ab = ∇rho_a·∇rho_b, sigma_bb = ∇rho_b·∇rho_b
    tau_a, tau_b : (npt,) alpha/beta kinetic energy densities
    deriv : int — only 1 is supported

    Returns
    -------
    exc : (npt,) XC energy density per electron (so E_xc = ∫ rho * exc dr)
    vrho_a, vrho_b : (npt,) d(rho*exc)/d(rho_a/b)
    vsigma_aa, vsigma_ab, vsigma_bb : (npt,) d(rho*exc)/d(sigma_ij)
    vtau_a, vtau_b : (npt,) d(rho*exc)/d(tau_a/b)
    """
    if deriv != 1:
        raise NotImplementedError("Only deriv=1 is supported")

    xp = _xp(rho_a)
    rho_a_safe = xp.maximum(rho_a, _RHO_CUTOFF)
    rho_b_safe = xp.maximum(rho_b, _RHO_CUTOFF)
    rho = rho_a + rho_b
    rho_safe = xp.maximum(rho, _RHO_CUTOFF)

    sigma_aa_safe = xp.maximum(sigma_aa, 0.0)
    sigma_bb_safe = xp.maximum(sigma_bb, 0.0)
    sigma_ab_use = sigma_ab

    tau_a_safe = xp.maximum(tau_a, _TAU_CUTOFF)
    tau_b_safe = xp.maximum(tau_b, _TAU_CUTOFF)

    # Derivative basis vectors as broadcasted views (avoid allocating 7x7xN explicitly).
    dtype = rho_safe.dtype
    npt = int(rho_safe.shape[0])
    eye = xp.eye(7, dtype=dtype)
    d0 = xp.broadcast_to(eye[:, 0][:, None], (7, npt))
    d1 = xp.broadcast_to(eye[:, 1][:, None], (7, npt))
    d2 = xp.broadcast_to(eye[:, 2][:, None], (7, npt))
    d3 = xp.broadcast_to(eye[:, 3][:, None], (7, npt))
    d4 = xp.broadcast_to(eye[:, 4][:, None], (7, npt))
    d5 = xp.broadcast_to(eye[:, 5][:, None], (7, npt))
    d6 = xp.broadcast_to(eye[:, 6][:, None], (7, npt))

    rho_a_t = _T7(rho_a_safe, d0)
    rho_b_t = _T7(rho_b_safe, d1)
    sigma_aa_t = _T7(sigma_aa_safe, d2)
    sigma_ab_t = _T7(sigma_ab_use, d3)
    sigma_bb_t = _T7(sigma_bb_safe, d4)
    tau_a_t = _T7(tau_a_safe, d5)
    tau_b_t = _T7(tau_b_safe, d6)

    name = str(spec.name).strip().lower()
    if name == "m06-l":
        Ex_a = _m06_exchange7(2.0 * rho_a_t, 4.0 * sigma_aa_t, 2.0 * tau_a_t, P.M06L_X_A, P.M06L_X_D, P.M06_X_ALPHA)
        Ex_b = _m06_exchange7(2.0 * rho_b_t, 4.0 * sigma_bb_t, 2.0 * tau_b_t, P.M06L_X_A, P.M06L_X_D, P.M06_X_ALPHA)
        E_x = 0.5 * (Ex_a + Ex_b)
        E_c = _m06_correlation_spin7(
            rho_a_t, sigma_aa_t, tau_a_t, rho_b_t, sigma_bb_t, tau_b_t, sigma_ab_t,
            css=P.M06L_C_CSS, cab=P.M06L_C_CAB, dss=P.M06L_C_DSS, dab=P.M06L_C_DAB,
            gamma_ss=P.M06_C_GAMMA_SS, gamma_ab=P.M06_C_GAMMA_AB,
            alpha_ss=P.M06_C_ALPHA_SS, alpha_ab=P.M06_C_ALPHA_AB, fermi_cnst=P.M06_C_FERMI_D_CNST,
        )
    elif name == "m06":
        Ex_a = _m06_exchange7(2.0 * rho_a_t, 4.0 * sigma_aa_t, 2.0 * tau_a_t, P.M06_X_A, P.M06_X_D, P.M06_X_ALPHA)
        Ex_b = _m06_exchange7(2.0 * rho_b_t, 4.0 * sigma_bb_t, 2.0 * tau_b_t, P.M06_X_A, P.M06_X_D, P.M06_X_ALPHA)
        E_x = 0.5 * (Ex_a + Ex_b)
        E_c = _m06_correlation_spin7(
            rho_a_t, sigma_aa_t, tau_a_t, rho_b_t, sigma_bb_t, tau_b_t, sigma_ab_t,
            css=P.M06_C_CSS, cab=P.M06_C_CAB, dss=P.M06_C_DSS, dab=P.M06_C_DAB,
            gamma_ss=P.M06_C_GAMMA_SS, gamma_ab=P.M06_C_GAMMA_AB,
            alpha_ss=P.M06_C_ALPHA_SS, alpha_ab=P.M06_C_ALPHA_AB, fermi_cnst=P.M06_C_FERMI_D_CNST,
        )
    elif name == "m06-2x":
        Ex_a = _m06_exchange7(2.0 * rho_a_t, 4.0 * sigma_aa_t, 2.0 * tau_a_t, P.M06_2X_X_A, P.M06_2X_X_D, 0.0)
        Ex_b = _m06_exchange7(2.0 * rho_b_t, 4.0 * sigma_bb_t, 2.0 * tau_b_t, P.M06_2X_X_A, P.M06_2X_X_D, 0.0)
        E_x = 0.5 * (Ex_a + Ex_b)
        E_c = _m06_correlation_spin7(
            rho_a_t, sigma_aa_t, tau_a_t, rho_b_t, sigma_bb_t, tau_b_t, sigma_ab_t,
            css=P.M06_2X_C_CSS, cab=P.M06_2X_C_CAB, dss=P.M06_2X_C_DSS, dab=P.M06_2X_C_DAB,
            gamma_ss=P.M06_C_GAMMA_SS, gamma_ab=P.M06_C_GAMMA_AB,
            alpha_ss=P.M06_C_ALPHA_SS, alpha_ab=P.M06_C_ALPHA_AB, fermi_cnst=P.M06_C_FERMI_D_CNST,
        )
    elif name == "mn15":
        Ex_a = _mn15_exchange7(2.0 * rho_a_t, 4.0 * sigma_aa_t, 2.0 * tau_a_t, P.MN15_X_CC)
        Ex_b = _mn15_exchange7(2.0 * rho_b_t, 4.0 * sigma_bb_t, 2.0 * tau_b_t, P.MN15_X_CC)
        E_x = 0.5 * (Ex_a + Ex_b)
        E_c = _mn15_correlation_spin7(
            rho_a_t, sigma_aa_t, tau_a_t, rho_b_t, sigma_bb_t, tau_b_t, sigma_ab_t,
            m08a=P.MN15_C_A, m08b=P.MN15_C_B,
        )
    else:
        raise ValueError(f"Unknown functional: {spec.name!r}")

    E = E_x + E_c

    mask = rho > _RHO_CUTOFF
    exc = xp.where(mask, E.v / rho_safe, 0.0)
    vrho_a = xp.where(mask, E.d[0], 0.0)
    vrho_b = xp.where(mask, E.d[1], 0.0)
    vsigma_aa = xp.where(mask, E.d[2], 0.0)
    vsigma_ab = xp.where(mask, E.d[3], 0.0)
    vsigma_bb = xp.where(mask, E.d[4], 0.0)
    vtau_a = xp.where(mask, E.d[5], 0.0)
    vtau_b = xp.where(mask, E.d[6], 0.0)

    return exc, vrho_a, vrho_b, vsigma_aa, vsigma_ab, vsigma_bb, vtau_a, vtau_b


def eval_xc_u(
    spec: FunctionalSpec,
    rho_a: Any,
    sigma_aa: Any,
    tau_a: Any,
    rho_b: Any,
    sigma_bb: Any,
    tau_b: Any,
    *,
    deriv: int = 1,
) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
    """Spin-polarized XC evaluation via spin-scaling.

    Uses E_xc[ρ_α, ρ_β] = E_xc^RKS[2ρ_α]/2 + E_xc^RKS[2ρ_β]/2, which is
    exact for exchange and a standard approximation for correlation.

    Parameters
    ----------
    rho_a, sigma_aa, tau_a : (npt,) — alpha-spin density, |∇ρ_α|², KE density
    rho_b, sigma_bb, tau_b : (npt,) — beta-spin density, |∇ρ_β|², KE density

    Returns
    -------
    exc : (npt,) — total XC energy density per electron (for E_xc accumulation)
    vrho_a, vsigma_aa, vtau_a : (npt,) — alpha-spin potentials
    vrho_b, vsigma_bb, vtau_b : (npt,) — beta-spin potentials

    Notes
    -----
    The VXC matrix elements use the same coefficients as RKS but with
    different scaling: vsigma coefficient is 4 (not 2) and vtau is 1 (not 0.5),
    because the scaled inputs 4σ_αα and 2τ_α carry extra chain-rule factors.
    The returned vsigma_aa and vtau_a already include these factors, so
    build_vxc_u can use the standard _build_vxc_batch with the right prefactors.
    """
    # Evaluate at scaled alpha density: (2ρ_α, 4σ_αα, 2τ_α)
    exc_a, vrho_a_out, vsigma_aa_out, vtau_a_out = eval_xc(
        spec, 2.0 * rho_a, 4.0 * sigma_aa, 2.0 * tau_a, deriv=deriv,
    )
    # Evaluate at scaled beta density: (2ρ_β, 4σ_ββ, 2τ_β)
    exc_b, vrho_b_out, vsigma_bb_out, vtau_b_out = eval_xc(
        spec, 2.0 * rho_b, 4.0 * sigma_bb, 2.0 * tau_b, deriv=deriv,
    )
    # exc is used as: E_xc += w * (exc_a * rho_a + exc_b * rho_b)
    # No need for a combined exc per electron since build_vxc_u accumulates separately.
    return exc_a, vrho_a_out, vsigma_aa_out, vtau_a_out, exc_b, vrho_b_out, vsigma_bb_out, vtau_b_out


__all__ = ["eval_xc", "eval_xc_u", "eval_xc_sp"]
