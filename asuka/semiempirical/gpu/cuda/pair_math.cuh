#pragma once

struct Dual3 {
    double v;
    double dx;
    double dy;
    double dz;

    __device__ __forceinline__ Dual3() : v(0.0), dx(0.0), dy(0.0), dz(0.0) {}
    __device__ __forceinline__ explicit Dual3(double val) : v(val), dx(0.0), dy(0.0), dz(0.0) {}
    __device__ __forceinline__ Dual3(double val, double gx, double gy, double gz)
        : v(val), dx(gx), dy(gy), dz(gz) {}
};

__device__ __forceinline__ Dual3 d_make_const(double x) {
    return Dual3(x, 0.0, 0.0, 0.0);
}

__device__ __forceinline__ Dual3 d_make_var(double x, int axis) {
    return Dual3(
        x,
        axis == 0 ? 1.0 : 0.0,
        axis == 1 ? 1.0 : 0.0,
        axis == 2 ? 1.0 : 0.0
    );
}

__device__ __forceinline__ Dual3 operator+(const Dual3& a, const Dual3& b) {
    return Dual3(a.v + b.v, a.dx + b.dx, a.dy + b.dy, a.dz + b.dz);
}

__device__ __forceinline__ Dual3 operator-(const Dual3& a, const Dual3& b) {
    return Dual3(a.v - b.v, a.dx - b.dx, a.dy - b.dy, a.dz - b.dz);
}

__device__ __forceinline__ Dual3 operator-(const Dual3& a) {
    return Dual3(-a.v, -a.dx, -a.dy, -a.dz);
}

__device__ __forceinline__ Dual3 operator*(const Dual3& a, const Dual3& b) {
    return Dual3(
        a.v * b.v,
        a.dx * b.v + a.v * b.dx,
        a.dy * b.v + a.v * b.dy,
        a.dz * b.v + a.v * b.dz
    );
}

__device__ __forceinline__ Dual3 operator/(const Dual3& a, const Dual3& b) {
    const double inv = 1.0 / b.v;
    const double inv2 = inv * inv;
    return Dual3(
        a.v * inv,
        (a.dx * b.v - a.v * b.dx) * inv2,
        (a.dy * b.v - a.v * b.dy) * inv2,
        (a.dz * b.v - a.v * b.dz) * inv2
    );
}

__device__ __forceinline__ Dual3 operator+(const Dual3& a, double b) { return a + Dual3(b); }
__device__ __forceinline__ Dual3 operator+(double a, const Dual3& b) { return Dual3(a) + b; }
__device__ __forceinline__ Dual3 operator-(const Dual3& a, double b) { return a - Dual3(b); }
__device__ __forceinline__ Dual3 operator-(double a, const Dual3& b) { return Dual3(a) - b; }
__device__ __forceinline__ Dual3 operator*(const Dual3& a, double b) { return a * Dual3(b); }
__device__ __forceinline__ Dual3 operator*(double a, const Dual3& b) { return Dual3(a) * b; }
__device__ __forceinline__ Dual3 operator/(const Dual3& a, double b) { return a / Dual3(b); }
__device__ __forceinline__ Dual3 operator/(double a, const Dual3& b) { return Dual3(a) / b; }

__device__ __forceinline__ double d_val(double x) { return x; }
__device__ __forceinline__ double d_val(const Dual3& x) { return x.v; }

__device__ __forceinline__ double d_exp(double x) { return exp(x); }
__device__ __forceinline__ Dual3 d_exp(const Dual3& x) {
    const double ev = exp(x.v);
    return Dual3(ev, ev * x.dx, ev * x.dy, ev * x.dz);
}

__device__ __forceinline__ double d_sqrt(double x) { return sqrt(x); }
__device__ __forceinline__ Dual3 d_sqrt(const Dual3& x) {
    const double sv = sqrt(x.v);
    const double inv2 = 0.5 / sv;
    return Dual3(sv, inv2 * x.dx, inv2 * x.dy, inv2 * x.dz);
}

__device__ __forceinline__ int pm_idx2(int r, int c, int lda) {
    return r * lda + c;
}

__device__ __forceinline__ int pm_idx4(int a, int b, int c, int d) {
    return ((a * 4 + b) * 4 + c) * 4 + d;
}

static __device__ __constant__ double PM_NRI[22] = {
    1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0,
    -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
};

struct AtomParams {
    int Z;
    int zval;
    int ngauss;
    double zeta_s;
    double zeta_p;
    double beta_s;
    double beta_p;
    double alpha;
    double dd;
    double qq;
    double am;
    double ad;
    double aq;
    double gk[4];
    double gl[4];
    double gm[4];
};

__device__ __forceinline__ AtomParams load_atom_params(const double* atom_params, int atom_id) {
    const double* p = atom_params + ((size_t)atom_id) * 32;
    AtomParams out;
    out.Z = (int)(p[0] >= 0.0 ? p[0] + 0.5 : p[0] - 0.5);
    out.zval = (int)(p[1] >= 0.0 ? p[1] + 0.5 : p[1] - 0.5);
    out.zeta_s = p[2];
    out.zeta_p = p[3];
    out.beta_s = p[4];
    out.beta_p = p[5];
    out.alpha = p[6];
    out.ngauss = (int)(p[7] >= 0.0 ? p[7] + 0.5 : p[7] - 0.5);
    for (int i = 0; i < 4; ++i) {
        out.gk[i] = p[8 + i];
        out.gl[i] = p[12 + i];
        out.gm[i] = p[16 + i];
    }
    out.dd = p[20];
    out.qq = p[21];
    out.am = p[22];
    out.ad = p[23];
    out.aq = p[24];
    return out;
}

__device__ __forceinline__ double beta_for_orb(const AtomParams& ap, int orb) {
    return orb == 0 ? ap.beta_s : ap.beta_p;
}

template <typename T>
__device__ __forceinline__ T pm_A(int k, const T& x) {
    const double xv = d_val(x);
    if (fabs(xv) < 1e-15) {
        return T(1.0 / (double)(k + 1));
    }
    const T ex = d_exp(-x);
    T a = ex / x;
    for (int j = 1; j <= k; ++j) {
        a = (double(j) * a + ex) / x;
    }
    return a;
}

template <typename T>
__device__ __forceinline__ T pm_B(int k, const T& x) {
    const double xv = d_val(x);
    if (fabs(xv) < 1e-10) {
        return (k % 2 == 0) ? T(2.0 / (double)(k + 1)) : T(0.0);
    }
    const T ep = d_exp(x);
    const T em = d_exp(-x);
    T b = (ep - em) / x;
    for (int j = 1; j <= k; ++j) {
        const double sign = (j % 2 == 0) ? 1.0 : -1.0;
        b = (sign * ep - em + double(j) * b) / x;
    }
    return b;
}

template <typename T>
__device__ __forceinline__ T pm_overlap_1s_1s(double za, double zb, const T& R) {
    const T p = (za + zb) * R / 2.0;
    const T q = (za - zb) * R / 2.0;
    const double Na = 2.0 * pow(za, 1.5);
    const double Nb = 2.0 * pow(zb, 1.5);
    const T pref = Na * Nb / 2.0 * (R / 2.0) * (R / 2.0) * (R / 2.0);
    return pref * (pm_A(2, p) * pm_B(0, q) - pm_A(0, p) * pm_B(2, q));
}

template <typename T>
__device__ __forceinline__ T pm_overlap_1s_2s(double za, double zb, const T& R) {
    const T p = (za + zb) * R / 2.0;
    const T q = (za - zb) * R / 2.0;
    const double Na = 2.0 * pow(za, 1.5);
    const double Nb = 2.0 * pow(zb, 2.5) / sqrt(3.0);
    const T pref = Na * Nb / 2.0 * (R / 2.0) * (R / 2.0) * (R / 2.0) * (R / 2.0);
    return pref * (
        pm_A(3, p) * pm_B(0, q) - pm_A(2, p) * pm_B(1, q)
        - pm_A(1, p) * pm_B(2, q) + pm_A(0, p) * pm_B(3, q)
    );
}

template <typename T>
__device__ __forceinline__ T pm_overlap_2s_1s(double za, double zb, const T& R) {
    const T p = (za + zb) * R / 2.0;
    const T q = (za - zb) * R / 2.0;
    const double Na = 2.0 * pow(za, 2.5) / sqrt(3.0);
    const double Nb = 2.0 * pow(zb, 1.5);
    const T pref = Na * Nb / 2.0 * (R / 2.0) * (R / 2.0) * (R / 2.0) * (R / 2.0);
    return pref * (
        pm_A(3, p) * pm_B(0, q) + pm_A(2, p) * pm_B(1, q)
        - pm_A(1, p) * pm_B(2, q) - pm_A(0, p) * pm_B(3, q)
    );
}

template <typename T>
__device__ __forceinline__ T pm_overlap_2s_2s(double za, double zb, const T& R) {
    const T p = (za + zb) * R / 2.0;
    const T q = (za - zb) * R / 2.0;
    const double Na = 2.0 * pow(za, 2.5) / sqrt(3.0);
    const double Nb = 2.0 * pow(zb, 2.5) / sqrt(3.0);
    const T pref = Na * Nb / 2.0 * (R / 2.0) * (R / 2.0) * (R / 2.0) * (R / 2.0) * (R / 2.0);
    return pref * (
        pm_A(4, p) * pm_B(0, q) - 2.0 * pm_A(2, p) * pm_B(2, q) + pm_A(0, p) * pm_B(4, q)
    );
}

template <typename T>
__device__ __forceinline__ T pm_overlap_1s_2ps(double za, double zb, const T& R) {
    const T p = (za + zb) * R / 2.0;
    const T q = (za - zb) * R / 2.0;
    const double Na = 2.0 * pow(za, 1.5);
    const double Nb = 2.0 * pow(zb, 2.5) / sqrt(3.0);
    const T pref = Na * Nb * sqrt(3.0) / 2.0 * (R / 2.0) * (R / 2.0) * (R / 2.0) * (R / 2.0);
    return pref * (
        pm_A(3, p) * pm_B(1, q) - pm_A(2, p) * pm_B(0, q)
        - pm_A(1, p) * pm_B(3, q) + pm_A(0, p) * pm_B(2, q)
    );
}

template <typename T>
__device__ __forceinline__ T pm_overlap_2ps_1s(double za, double zb, const T& R) {
    const T p = (za + zb) * R / 2.0;
    const T q = (za - zb) * R / 2.0;
    const double Na = 2.0 * pow(za, 2.5) / sqrt(3.0);
    const double Nb = 2.0 * pow(zb, 1.5);
    const T pref = Na * Nb * sqrt(3.0) / 2.0 * (R / 2.0) * (R / 2.0) * (R / 2.0) * (R / 2.0);
    return pref * (
        pm_A(2, p) * pm_B(0, q) + pm_A(3, p) * pm_B(1, q)
        - pm_A(0, p) * pm_B(2, q) - pm_A(1, p) * pm_B(3, q)
    );
}

template <typename T>
__device__ __forceinline__ T pm_overlap_2s_2ps(double za, double zb, const T& R) {
    const T p = (za + zb) * R / 2.0;
    const T q = (za - zb) * R / 2.0;
    const double Na = 2.0 * pow(za, 2.5) / sqrt(3.0);
    const double Nb = 2.0 * pow(zb, 2.5) / sqrt(3.0);
    const T pref = Na * Nb * sqrt(3.0) / 2.0 * (R / 2.0) * (R / 2.0) * (R / 2.0) * (R / 2.0) * (R / 2.0);
    return pref * (
        pm_A(4, p) * pm_B(1, q) + pm_A(3, p) * pm_B(2, q)
        - pm_A(2, p) * pm_B(3, q) - pm_A(1, p) * pm_B(4, q)
        - pm_A(3, p) * pm_B(0, q) - pm_A(2, p) * pm_B(1, q)
        + pm_A(1, p) * pm_B(2, q) + pm_A(0, p) * pm_B(3, q)
    );
}

template <typename T>
__device__ __forceinline__ T pm_overlap_2ps_2s(double za, double zb, const T& R) {
    const T p = (za + zb) * R / 2.0;
    const T q = (za - zb) * R / 2.0;
    const double Na = 2.0 * pow(za, 2.5) / sqrt(3.0);
    const double Nb = 2.0 * pow(zb, 2.5) / sqrt(3.0);
    const T pref = Na * Nb * sqrt(3.0) / 2.0 * (R / 2.0) * (R / 2.0) * (R / 2.0) * (R / 2.0) * (R / 2.0);
    return pref * (
        pm_A(3, p) * pm_B(0, q) + pm_A(4, p) * pm_B(1, q)
        - pm_A(2, p) * pm_B(1, q) - pm_A(3, p) * pm_B(2, q)
        - pm_A(1, p) * pm_B(2, q) - pm_A(2, p) * pm_B(3, q)
        + pm_A(0, p) * pm_B(3, q) + pm_A(1, p) * pm_B(4, q)
    );
}

template <typename T>
__device__ __forceinline__ T pm_overlap_2ps_2ps(double za, double zb, const T& R) {
    const T p = (za + zb) * R / 2.0;
    const T q = (za - zb) * R / 2.0;
    const double Na = 2.0 * pow(za, 2.5) / sqrt(3.0);
    const double Nb = 2.0 * pow(zb, 2.5) / sqrt(3.0);
    const T pref = Na * Nb * 3.0 / 2.0 * (R / 2.0) * (R / 2.0) * (R / 2.0) * (R / 2.0) * (R / 2.0);
    return pref * (
        pm_A(4, p) * pm_B(2, q) - pm_A(2, p) * pm_B(0, q)
        - pm_A(2, p) * pm_B(4, q) + pm_A(0, p) * pm_B(2, q)
    );
}

template <typename T>
__device__ __forceinline__ T pm_overlap_2pp_2pp(double za, double zb, const T& R) {
    const T p = (za + zb) * R / 2.0;
    const T q = (za - zb) * R / 2.0;
    const double Na = 2.0 * pow(za, 2.5) / sqrt(3.0);
    const double Nb = 2.0 * pow(zb, 2.5) / sqrt(3.0);
    const T pref = Na * Nb * 3.0 / 4.0 * (R / 2.0) * (R / 2.0) * (R / 2.0) * (R / 2.0) * (R / 2.0);
    return pref * (
        pm_A(4, p) * pm_B(0, q) - pm_A(4, p) * pm_B(2, q)
        - pm_A(2, p) * pm_B(0, q) + pm_A(2, p) * pm_B(4, q)
        + pm_A(0, p) * pm_B(2, q) - pm_A(0, p) * pm_B(4, q)
    );
}

template <typename T>
__device__ __forceinline__ void pm_compute_diatomic_overlaps(
    int naoA,
    int naoB,
    double zsA,
    double zpA,
    double zsB,
    double zpB,
    const T& R,
    T* Sss,
    T* Ssp,
    T* Sps,
    T* Spp_sig,
    T* Spp_pi
) {
    *Sss = T(0.0);
    *Ssp = T(0.0);
    *Sps = T(0.0);
    *Spp_sig = T(0.0);
    *Spp_pi = T(0.0);

    if (naoA == 1 && naoB == 1) {
        *Sss = pm_overlap_1s_1s(zsA, zsB, R);
        return;
    }
    if (naoA == 1 && naoB == 4) {
        *Sss = pm_overlap_1s_2s(zsA, zsB, R);
        *Ssp = pm_overlap_1s_2ps(zsA, zpB, R);
        return;
    }
    if (naoA == 4 && naoB == 1) {
        *Sss = pm_overlap_2s_1s(zsA, zsB, R);
        *Sps = pm_overlap_2ps_1s(zpA, zsB, R);
        return;
    }

    *Sss = pm_overlap_2s_2s(zsA, zsB, R);
    *Ssp = pm_overlap_2s_2ps(zsA, zpB, R);
    *Sps = pm_overlap_2ps_2s(zpA, zsB, R);
    *Spp_sig = pm_overlap_2ps_2ps(zpA, zpB, R);
    *Spp_pi = pm_overlap_2pp_2pp(zpA, zpB, R);
}

template <typename T>
__device__ __forceinline__ void pm_rotate_overlaps_to_global(
    int naoA,
    int naoB,
    const T& S_ss,
    const T& S_sp,
    const T& S_ps,
    const T& S_pp_sig,
    const T& S_pp_pi,
    const T& cx,
    const T& cy,
    const T& cz,
    T* S
) {
    for (int i = 0; i < 16; ++i) {
        S[i] = T(0.0);
    }
    S[pm_idx2(0, 0, 4)] = S_ss;

    if (naoB == 4) {
        S[pm_idx2(0, 1, 4)] = S_sp * cx;
        S[pm_idx2(0, 2, 4)] = S_sp * cy;
        S[pm_idx2(0, 3, 4)] = S_sp * cz;
    }

    if (naoA == 4) {
        S[pm_idx2(1, 0, 4)] = S_ps * cx;
        S[pm_idx2(2, 0, 4)] = S_ps * cy;
        S[pm_idx2(3, 0, 4)] = S_ps * cz;
    }

    if (naoA == 4 && naoB == 4) {
        T c[3] = {cx, cy, cz};
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                const double delta = (i == j) ? 1.0 : 0.0;
                S[pm_idx2(1 + i, 1 + j, 4)] = S_pp_sig * c[i] * c[j] + S_pp_pi * (delta - c[i] * c[j]);
            }
        }
    }
}

template <typename T>
__device__ __forceinline__ void pm_build_transform4(
    const T& cx,
    const T& cy,
    const T& cz,
    int nao,
    T* Tmat
) {
    for (int i = 0; i < 16; ++i) {
        Tmat[i] = T(0.0);
    }
    Tmat[pm_idx2(0, 0, 4)] = T(1.0);
    if (nao == 1) {
        return;
    }

    const T rxy = d_sqrt(cx * cx + cy * cy);
    T e1x, e1y, e1z;
    T e2x, e2y, e2z;
    if (d_val(rxy) > 1e-10) {
        const T ca = cx / rxy;
        const T sa = cy / rxy;
        e1x = ca * cz;
        e1y = sa * cz;
        e1z = -rxy;
        e2x = -sa;
        e2y = ca;
        e2z = T(0.0);
    } else {
        const double sign = d_val(cz) > 0.0 ? 1.0 : -1.0;
        e1x = T(sign);
        e1y = T(0.0);
        e1z = T(0.0);
        e2x = T(0.0);
        e2y = T(1.0);
        e2z = T(0.0);
    }

    Tmat[pm_idx2(1, 1, 4)] = cx;
    Tmat[pm_idx2(2, 1, 4)] = cy;
    Tmat[pm_idx2(3, 1, 4)] = cz;

    Tmat[pm_idx2(1, 2, 4)] = e1x;
    Tmat[pm_idx2(2, 2, 4)] = e1y;
    Tmat[pm_idx2(3, 2, 4)] = e1z;

    Tmat[pm_idx2(1, 3, 4)] = e2x;
    Tmat[pm_idx2(2, 3, 4)] = e2y;
    Tmat[pm_idx2(3, 3, 4)] = e2z;
}

template <typename T>
__device__ __forceinline__ void pm_fill_local_tensor4(const T* ri, int naoA, int naoB, T* L) {
    for (int i = 0; i < 256; ++i) {
        L[i] = T(0.0);
    }

    L[pm_idx4(0, 0, 0, 0)] = ri[0];

    if (naoA >= 4) {
        L[pm_idx4(0, 1, 0, 0)] = ri[1];
        L[pm_idx4(1, 0, 0, 0)] = ri[1];
        L[pm_idx4(1, 1, 0, 0)] = ri[2];
        L[pm_idx4(2, 2, 0, 0)] = ri[3];
        L[pm_idx4(3, 3, 0, 0)] = ri[3];
    }

    if (naoB >= 4) {
        L[pm_idx4(0, 0, 0, 1)] = ri[4];
        L[pm_idx4(0, 0, 1, 0)] = ri[4];
        L[pm_idx4(0, 0, 1, 1)] = ri[10];
        L[pm_idx4(0, 0, 2, 2)] = ri[11];
        L[pm_idx4(0, 0, 3, 3)] = ri[11];
    }

    if (naoA >= 4 && naoB >= 4) {
        L[pm_idx4(0, 1, 0, 1)] = ri[5];
        L[pm_idx4(1, 0, 0, 1)] = ri[5];
        L[pm_idx4(0, 1, 1, 0)] = ri[5];
        L[pm_idx4(1, 0, 1, 0)] = ri[5];

        for (int p = 2; p <= 3; ++p) {
            L[pm_idx4(0, p, 0, p)] = ri[6];
            L[pm_idx4(p, 0, 0, p)] = ri[6];
            L[pm_idx4(0, p, p, 0)] = ri[6];
            L[pm_idx4(p, 0, p, 0)] = ri[6];
        }

        L[pm_idx4(1, 1, 0, 1)] = ri[7];
        L[pm_idx4(1, 1, 1, 0)] = ri[7];

        for (int p = 2; p <= 3; ++p) {
            L[pm_idx4(p, p, 0, 1)] = ri[8];
            L[pm_idx4(p, p, 1, 0)] = ri[8];
        }

        for (int p = 2; p <= 3; ++p) {
            L[pm_idx4(1, p, 0, p)] = ri[9];
            L[pm_idx4(p, 1, 0, p)] = ri[9];
            L[pm_idx4(1, p, p, 0)] = ri[9];
            L[pm_idx4(p, 1, p, 0)] = ri[9];
        }

        L[pm_idx4(0, 1, 1, 1)] = ri[12];
        L[pm_idx4(1, 0, 1, 1)] = ri[12];

        for (int p = 2; p <= 3; ++p) {
            L[pm_idx4(0, 1, p, p)] = ri[13];
            L[pm_idx4(1, 0, p, p)] = ri[13];
        }

        for (int p = 2; p <= 3; ++p) {
            L[pm_idx4(0, p, 1, p)] = ri[14];
            L[pm_idx4(p, 0, 1, p)] = ri[14];
            L[pm_idx4(0, p, p, 1)] = ri[14];
            L[pm_idx4(p, 0, p, 1)] = ri[14];
        }

        L[pm_idx4(1, 1, 1, 1)] = ri[15];
        for (int p = 2; p <= 3; ++p) {
            L[pm_idx4(p, p, 1, 1)] = ri[16];
            L[pm_idx4(1, 1, p, p)] = ri[17];
        }

        L[pm_idx4(2, 2, 2, 2)] = ri[18];
        L[pm_idx4(3, 3, 3, 3)] = ri[18];

        for (int p = 2; p <= 3; ++p) {
            L[pm_idx4(1, p, 1, p)] = ri[19];
            L[pm_idx4(p, 1, 1, p)] = ri[19];
            L[pm_idx4(1, p, p, 1)] = ri[19];
            L[pm_idx4(p, 1, p, 1)] = ri[19];
        }

        L[pm_idx4(2, 2, 3, 3)] = ri[20];
        L[pm_idx4(3, 3, 2, 2)] = ri[20];
        L[pm_idx4(2, 3, 2, 3)] = ri[21];
        L[pm_idx4(3, 2, 3, 2)] = ri[21];
        L[pm_idx4(2, 3, 3, 2)] = ri[21];
        L[pm_idx4(3, 2, 2, 3)] = ri[21];
    }
}

template <typename T>
__device__ __forceinline__ void pm_build_w_from_ri(
    const T* ri,
    const T* TA,
    const T* TB,
    int naoA,
    int naoB,
    T* W
) {
    T L[256];
    pm_fill_local_tensor4(ri, naoA, naoB, L);

    for (int i = 0; i < 256; ++i) {
        W[i] = T(0.0);
    }

    for (int m = 0; m < naoA; ++m) {
        for (int n = 0; n < naoA; ++n) {
            for (int l = 0; l < naoB; ++l) {
                for (int s = 0; s < naoB; ++s) {
                    T acc = T(0.0);
                    for (int a = 0; a < naoA; ++a) {
                        const T tma = TA[pm_idx2(m, a, 4)];
                        for (int b = 0; b < naoA; ++b) {
                            const T tnb = TA[pm_idx2(n, b, 4)];
                            for (int c = 0; c < naoB; ++c) {
                                const T tlc = TB[pm_idx2(l, c, 4)];
                                for (int d = 0; d < naoB; ++d) {
                                    const T tsd = TB[pm_idx2(s, d, 4)];
                                    acc = acc + tma * tnb * L[pm_idx4(a, b, c, d)] * tlc * tsd;
                                }
                            }
                        }
                    }
                    W[pm_idx4(m, n, l, s)] = acc;
                }
            }
        }
    }
}

template <typename T>
__device__ __forceinline__ void pm_reppd_heavy_heavy(const T& R, const AtomParams& A, const AtomParams& B, T* ri) {
    const double am_A = A.am;
    const double ad_A = A.ad;
    const double aq_A = A.aq;
    const double am_B = B.am;
    const double ad_B = B.ad;
    const double aq_B = B.aq;
    const double da = A.dd;
    const double db = B.dd;
    const double qa2 = A.qq * 2.0;
    const double qb2 = B.qq * 2.0;

    const double aee = (am_A + am_B) * (am_A + am_B);
    const double ade = (ad_A + am_B) * (ad_A + am_B);
    const double aqe = (aq_A + am_B) * (aq_A + am_B);
    const double aed = (am_A + ad_B) * (am_A + ad_B);
    const double aeq = (am_A + aq_B) * (am_A + aq_B);
    const double axx = (ad_A + ad_B) * (ad_A + ad_B);
    const double adq = (ad_A + aq_B) * (ad_A + aq_B);
    const double aqd = (aq_A + ad_B) * (aq_A + ad_B);
    const double aqq = (aq_A + aq_B) * (aq_A + aq_B);

    const T r = R;
    const T rsq = r * r;

    T a[72];
    a[0] = rsq + aee;
    a[1] = (r + da) * (r + da) + ade;
    a[2] = (r - da) * (r - da) + ade;
    a[3] = (r - qa2) * (r - qa2) + aqe;
    a[4] = (r + qa2) * (r + qa2) + aqe;
    a[5] = rsq + aqe;
    a[6] = a[5] + qa2 * qa2;
    a[7] = (r - db) * (r - db) + aed;
    a[8] = (r + db) * (r + db) + aed;
    a[9] = (r - qb2) * (r - qb2) + aeq;
    a[10] = (r + qb2) * (r + qb2) + aeq;
    a[11] = rsq + aeq;
    a[12] = a[11] + qb2 * qb2;
    a[13] = rsq + axx + (da - db) * (da - db);
    a[14] = rsq + axx + (da + db) * (da + db);
    a[15] = (r + da - db) * (r + da - db) + axx;
    a[16] = (r - da + db) * (r - da + db) + axx;
    a[17] = (r - da - db) * (r - da - db) + axx;
    a[18] = (r + da + db) * (r + da + db) + axx;
    a[19] = (r + da) * (r + da) + adq;
    a[20] = a[19] + qb2 * qb2;
    a[21] = (r - da) * (r - da) + adq;
    a[22] = a[21] + qb2 * qb2;
    a[23] = (r - db) * (r - db) + aqd;
    a[24] = a[23] + qa2 * qa2;
    a[25] = (r + db) * (r + db) + aqd;
    a[26] = a[25] + qa2 * qa2;
    a[27] = (r + da - qb2) * (r + da - qb2) + adq;
    a[28] = (r - da - qb2) * (r - da - qb2) + adq;
    a[29] = (r + da + qb2) * (r + da + qb2) + adq;
    a[30] = (r - da + qb2) * (r - da + qb2) + adq;
    a[31] = (r + qa2 - db) * (r + qa2 - db) + aqd;
    a[32] = (r + qa2 + db) * (r + qa2 + db) + aqd;
    a[33] = (r - qa2 - db) * (r - qa2 - db) + aqd;
    a[34] = (r - qa2 + db) * (r - qa2 + db) + aqd;
    a[35] = rsq + aqq;
    a[36] = a[35] + (qa2 - qb2) * (qa2 - qb2);
    a[37] = a[35] + (qa2 + qb2) * (qa2 + qb2);
    a[38] = a[35] + qa2 * qa2;
    a[39] = a[35] + qb2 * qb2;
    a[40] = a[38] + qb2 * qb2;
    a[41] = (r - qb2) * (r - qb2) + aqq;
    a[42] = a[41] + qa2 * qa2;
    a[43] = (r + qb2) * (r + qb2) + aqq;
    a[44] = a[43] + qa2 * qa2;
    a[45] = (r + qa2) * (r + qa2) + aqq;
    a[46] = a[45] + qb2 * qb2;
    a[47] = (r - qa2) * (r - qa2) + aqq;
    a[48] = a[47] + qb2 * qb2;
    a[49] = (r + qa2 - qb2) * (r + qa2 - qb2) + aqq;
    a[50] = (r + qa2 + qb2) * (r + qa2 + qb2) + aqq;
    a[51] = (r - qa2 - qb2) * (r - qa2 - qb2) + aqq;
    a[52] = (r - qa2 + qb2) * (r - qa2 + qb2) + aqq;

    const double qa = A.qq;
    const double qb = B.qq;
    const double da_m_qb2 = (da - qb) * (da - qb);
    const double da_p_qb2 = (da + qb) * (da + qb);
    const T r_m_qb2 = (r - qb) * (r - qb);
    const T r_p_qb2 = (r + qb) * (r + qb);
    a[53] = da_m_qb2 + r_m_qb2 + adq;
    a[54] = da_m_qb2 + r_p_qb2 + adq;
    a[55] = da_p_qb2 + r_m_qb2 + adq;
    a[56] = da_p_qb2 + r_p_qb2 + adq;

    const double qa_m_db2 = (qa - db) * (qa - db);
    const double qa_p_db2 = (qa + db) * (qa + db);
    const T r_p_qa2 = (r + qa) * (r + qa);
    const T r_m_qa2 = (r - qa) * (r - qa);
    a[57] = r_p_qa2 + qa_m_db2 + aqd;
    a[58] = r_m_qa2 + qa_m_db2 + aqd;
    a[59] = r_p_qa2 + qa_p_db2 + aqd;
    a[60] = r_m_qa2 + qa_p_db2 + aqd;

    const double qa_m_qb2 = (qa - qb) * (qa - qb);
    const double qa_p_qb2 = (qa + qb) * (qa + qb);
    a[61] = a[35] + 2.0 * qa_m_qb2;
    a[62] = a[35] + 2.0 * qa_p_qb2;
    a[63] = a[35] + 2.0 * (qa * qa + qb * qb);

    const T rpqamqb2 = (r + qa - qb) * (r + qa - qb);
    a[64] = rpqamqb2 + qa_m_qb2 + aqq;
    a[65] = rpqamqb2 + qa_p_qb2 + aqq;
    const T rpqapqb2 = (r + qa + qb) * (r + qa + qb);
    a[66] = rpqapqb2 + qa_m_qb2 + aqq;
    a[67] = rpqapqb2 + qa_p_qb2 + aqq;
    const T rmqamqb2 = (r - qa - qb) * (r - qa - qb);
    a[68] = rmqamqb2 + qa_m_qb2 + aqq;
    a[69] = rmqamqb2 + qa_p_qb2 + aqq;
    const T rmqapqb2 = (r - qa + qb) * (r - qa + qb);
    a[70] = rmqapqb2 + qa_m_qb2 + aqq;
    a[71] = rmqapqb2 + qa_p_qb2 + aqq;

    T s[72];
    for (int i = 0; i < 72; ++i) {
        s[i] = d_sqrt(a[i]);
    }

    const T ee = 1.0 / s[0];
    const T dze = -0.5 / s[1] + 0.5 / s[2];
    const T qzze = 0.25 / s[3] + 0.25 / s[4] - 0.5 / s[5];
    const T qxxe = 0.5 / s[6] - 0.5 / s[5];
    const T edz = -0.5 / s[7] + 0.5 / s[8];
    const T eqzz = 0.25 / s[9] + 0.25 / s[10] - 0.5 / s[11];
    const T eqxx = 0.5 / s[12] - 0.5 / s[11];
    const T dxdx = 0.5 / s[13] - 0.5 / s[14];
    const T dzdz = 0.25 * (1.0 / s[15] + 1.0 / s[16] - 1.0 / s[17] - 1.0 / s[18]);
    const T dzqxx = 0.25 * (1.0 / s[19] - 1.0 / s[20] - 1.0 / s[21] + 1.0 / s[22]);
    const T qxxdz = 0.25 * (1.0 / s[23] - 1.0 / s[24] - 1.0 / s[25] + 1.0 / s[26]);
    const T dzqzz =
        0.125 * (-1.0 / s[27] + 1.0 / s[28] - 1.0 / s[29] + 1.0 / s[30])
        + 0.25 * (-1.0 / s[21] + 1.0 / s[19]);
    const T qzzdz =
        0.125 * (-1.0 / s[31] + 1.0 / s[32] - 1.0 / s[33] + 1.0 / s[34])
        + 0.25 * (1.0 / s[23] - 1.0 / s[25]);
    const T qxxqxx =
        0.125 * (1.0 / s[36] + 1.0 / s[37])
        - 0.25 * (1.0 / s[38] + 1.0 / s[39])
        + 0.25 / s[35];
    const T qxxqyy = 0.25 * (1.0 / s[40] - 1.0 / s[38] - 1.0 / s[39] + 1.0 / s[35]);
    const T qxxqzz =
        0.125 * (1.0 / s[42] + 1.0 / s[44] - 1.0 / s[41] - 1.0 / s[43])
        + 0.25 * (-1.0 / s[38] + 1.0 / s[35]);
    const T qzzqxx =
        0.125 * (1.0 / s[46] + 1.0 / s[48] - 1.0 / s[45] - 1.0 / s[47])
        + 0.25 * (-1.0 / s[39] + 1.0 / s[35]);
    const T qzzqzz =
        0.0625 * (1.0 / s[49] + 1.0 / s[50] + 1.0 / s[51] + 1.0 / s[52])
        - 0.125 * (1.0 / s[47] + 1.0 / s[45] + 1.0 / s[41] + 1.0 / s[43])
        + 0.25 / s[35];
    const T dxqxz = 0.25 * (-1.0 / s[53] + 1.0 / s[54] + 1.0 / s[55] - 1.0 / s[56]);
    const T qxzdx = 0.25 * (-1.0 / s[57] + 1.0 / s[58] + 1.0 / s[59] - 1.0 / s[60]);
    const T qxzqxz =
        0.125 * (1.0 / s[64] - 1.0 / s[66] - 1.0 / s[68] + 1.0 / s[70]
            - 1.0 / s[65] + 1.0 / s[67] + 1.0 / s[69] - 1.0 / s[71]);

    ri[0] = ee;
    ri[1] = -dze;
    ri[2] = ee + qzze;
    ri[3] = ee + qxxe;
    ri[4] = -edz;
    ri[5] = dzdz;
    ri[6] = dxdx;
    ri[7] = -edz - qzzdz;
    ri[8] = -edz - qxxdz;
    ri[9] = -qxzdx;
    ri[10] = ee + eqzz;
    ri[11] = ee + eqxx;
    ri[12] = -dze - dzqzz;
    ri[13] = -dze - dzqxx;
    ri[14] = -dxqxz;
    ri[15] = ee + eqzz + qzze + qzzqzz;
    ri[16] = ee + eqzz + qxxe + qxxqzz;
    ri[17] = ee + eqxx + qzze + qzzqxx;
    ri[18] = ee + eqxx + qxxe + qxxqxx;
    ri[19] = qxzqxz;
    ri[20] = ee + eqxx + qxxe + qxxqyy;
    ri[21] = 0.5 * (qxxqxx - qxxqyy);
}

template <typename T>
__device__ __forceinline__ void pm_reppd(
    const T& R,
    const AtomParams& A,
    const AtomParams& B,
    bool si,
    bool sj,
    T* ri,
    T* gab
) {
    for (int i = 0; i < 22; ++i) {
        ri[i] = T(0.0);
    }

    const double aee = (A.am + B.am) * (A.am + B.am);
    const T rsq = R * R;
    *gab = 1.0 / d_sqrt(rsq + aee);

    if (!si && !sj) {
        ri[0] = 1.0 / d_sqrt(rsq + aee);
        return;
    }

    if (si && !sj) {
        const double da = A.dd;
        const double qa = A.qq * 2.0;
        const double ade = (A.ad + B.am) * (A.ad + B.am);
        const double aqe = (A.aq + B.am) * (A.aq + B.am);

        const T s1 = d_sqrt(rsq + aee);
        const T s2 = d_sqrt((R + da) * (R + da) + ade);
        const T s3 = d_sqrt((R - da) * (R - da) + ade);
        const T s4 = d_sqrt((R + qa) * (R + qa) + aqe);
        const T s5 = d_sqrt((R - qa) * (R - qa) + aqe);
        const T s6 = d_sqrt(rsq + aqe);
        const T s7 = d_sqrt(rsq + aqe + qa * qa);

        const T ee = 1.0 / s1;
        ri[0] = ee;
        ri[1] = 0.5 / s2 - 0.5 / s3;
        ri[2] = ee + 0.25 / s4 + 0.25 / s5 - 0.5 / s6;
        ri[3] = ee + 0.5 / s7 - 0.5 / s6;
        for (int i = 0; i < 4; ++i) {
            ri[i] = ri[i] * PM_NRI[i];
        }
        return;
    }

    if (!si && sj) {
        const double db = B.dd;
        const double qb = B.qq * 2.0;
        const double aed = (A.am + B.ad) * (A.am + B.ad);
        const double aeq = (A.am + B.aq) * (A.am + B.aq);

        const T s1 = d_sqrt(rsq + aee);
        const T s2 = d_sqrt((R - db) * (R - db) + aed);
        const T s3 = d_sqrt((R + db) * (R + db) + aed);
        const T s4 = d_sqrt((R - qb) * (R - qb) + aeq);
        const T s5 = d_sqrt((R + qb) * (R + qb) + aeq);
        const T s6 = d_sqrt(rsq + aeq);
        const T s7 = d_sqrt(rsq + aeq + qb * qb);

        const T ee = 1.0 / s1;
        ri[0] = ee;
        ri[4] = 0.5 / s2 - 0.5 / s3;
        ri[10] = ee + 0.25 / s4 + 0.25 / s5 - 0.5 / s6;
        ri[11] = ee + 0.5 / s7 - 0.5 / s6;
        ri[0] = ri[0] * PM_NRI[0];
        ri[4] = ri[4] * PM_NRI[4];
        ri[10] = ri[10] * PM_NRI[10];
        ri[11] = ri[11] * PM_NRI[11];
        return;
    }

    pm_reppd_heavy_heavy(R, A, B, ri);
    for (int i = 0; i < 22; ++i) {
        ri[i] = ri[i] * PM_NRI[i];
    }
}

__device__ __forceinline__ bool pm_is_nh_or_oh(int ZA, int ZB) {
    return (ZA == 1 && (ZB == 7 || ZB == 8)) || (ZB == 1 && (ZA == 7 || ZA == 8));
}

template <typename T>
__device__ __forceinline__ T pm_pair_core_repulsion(
    const AtomParams& A,
    const AtomParams& B,
    const T& R,
    const T& gamma_ss
) {
    const double EV_TO_HARTREE = 1.0 / 27.211386245988;
    const double BOHR_TO_ANG = 0.529177210903;

    const T R_ang = R * BOHR_TO_ANG;

    T fA;
    T fB;
    if (pm_is_nh_or_oh(A.Z, B.Z)) {
        if (A.Z == 1) {
            fA = d_exp(-A.alpha * R);
            fB = R_ang * d_exp(-B.alpha * R);
        } else {
            fA = R_ang * d_exp(-A.alpha * R);
            fB = d_exp(-B.alpha * R);
        }
    } else {
        fA = d_exp(-A.alpha * R);
        fB = d_exp(-B.alpha * R);
    }

    T E_mndo = double(A.zval) * double(B.zval) * gamma_ss * (1.0 + fA + fB);

    T gauss_sum = T(0.0);
    for (int i = 0; i < A.ngauss; ++i) {
        gauss_sum = gauss_sum + A.gk[i] * d_exp(-A.gl[i] * (R_ang - A.gm[i]) * (R_ang - A.gm[i]));
    }
    for (int i = 0; i < B.ngauss; ++i) {
        gauss_sum = gauss_sum + B.gk[i] * d_exp(-B.gl[i] * (R_ang - B.gm[i]) * (R_ang - B.gm[i]));
    }

    T E_gauss = T(0.0);
    if (d_val(R_ang) > 1e-10) {
        E_gauss = double(A.zval) * double(B.zval) * gauss_sum * EV_TO_HARTREE / R_ang;
    }

    return E_mndo + E_gauss;
}

template <int NA, int NB>
__device__ __forceinline__ Dual3 pm_pair_energy_dual(
    int atomA,
    int atomB,
    const double* coords,
    const double* atom_params,
    const double* PAA,
    const double* PBB,
    const double* PAB
) {
    const AtomParams apA = load_atom_params(atom_params, atomA);
    const AtomParams apB = load_atom_params(atom_params, atomB);

    const Dual3 rx = d_make_var(coords[atomB * 3 + 0] - coords[atomA * 3 + 0], 0);
    const Dual3 ry = d_make_var(coords[atomB * 3 + 1] - coords[atomA * 3 + 1], 1);
    const Dual3 rz = d_make_var(coords[atomB * 3 + 2] - coords[atomA * 3 + 2], 2);

    const Dual3 R = d_sqrt(rx * rx + ry * ry + rz * rz);
    const Dual3 cx = rx / R;
    const Dual3 cy = ry / R;
    const Dual3 cz = rz / R;

    Dual3 ri[22];
    Dual3 gamma_ss;
    pm_reppd(R, apA, apB, NA >= 4, NB >= 4, ri, &gamma_ss);

    Dual3 TA[16];
    Dual3 TB[16];
    pm_build_transform4(cx, cy, cz, NA, TA);
    pm_build_transform4(cx, cy, cz, NB, TB);

    Dual3 W[256];
    pm_build_w_from_ri(ri, TA, TB, NA, NB, W);

    Dual3 Sss, Ssp, Sps, Spp_sig, Spp_pi;
    pm_compute_diatomic_overlaps(
        NA,
        NB,
        apA.zeta_s,
        apA.zeta_p,
        apB.zeta_s,
        apB.zeta_p,
        R,
        &Sss,
        &Ssp,
        &Sps,
        &Spp_sig,
        &Spp_pi
    );

    Dual3 SAB[16];
    pm_rotate_overlaps_to_global(NA, NB, Sss, Ssp, Sps, Spp_sig, Spp_pi, cx, cy, cz, SAB);

    Dual3 e_one = d_make_const(0.0);
    for (int m = 0; m < NA; ++m) {
        for (int n = 0; n < NA; ++n) {
            const Dual3 HAA = -double(apB.zval) * W[pm_idx4(m, n, 0, 0)];
            e_one = e_one + PAA[pm_idx2(m, n, 4)] * HAA;
        }
    }
    for (int l = 0; l < NB; ++l) {
        for (int s = 0; s < NB; ++s) {
            const Dual3 HBB = -double(apA.zval) * W[pm_idx4(0, 0, l, s)];
            e_one = e_one + PBB[pm_idx2(l, s, 4)] * HBB;
        }
    }
    for (int m = 0; m < NA; ++m) {
        const double beta_m = beta_for_orb(apA, m);
        for (int l = 0; l < NB; ++l) {
            const double beta_l = beta_for_orb(apB, l);
            const Dual3 HAB = 0.5 * SAB[pm_idx2(m, l, 4)] * (beta_m + beta_l);
            e_one = e_one + 2.0 * PAB[pm_idx2(m, l, 4)] * HAB;
        }
    }

    Dual3 e_two = d_make_const(0.0);
    for (int m = 0; m < NA; ++m) {
        for (int n = 0; n < NA; ++n) {
            Dual3 J = d_make_const(0.0);
            for (int l = 0; l < NB; ++l) {
                for (int s = 0; s < NB; ++s) {
                    J = J + PBB[pm_idx2(l, s, 4)] * W[pm_idx4(m, n, l, s)];
                }
            }
            e_two = e_two + 0.5 * PAA[pm_idx2(m, n, 4)] * J;
        }
    }
    for (int l = 0; l < NB; ++l) {
        for (int s = 0; s < NB; ++s) {
            Dual3 J = d_make_const(0.0);
            for (int m = 0; m < NA; ++m) {
                for (int n = 0; n < NA; ++n) {
                    J = J + PAA[pm_idx2(m, n, 4)] * W[pm_idx4(m, n, l, s)];
                }
            }
            e_two = e_two + 0.5 * PBB[pm_idx2(l, s, 4)] * J;
        }
    }
    for (int m = 0; m < NA; ++m) {
        for (int l = 0; l < NB; ++l) {
            Dual3 K = d_make_const(0.0);
            for (int n = 0; n < NA; ++n) {
                for (int s = 0; s < NB; ++s) {
                    K = K + PAB[pm_idx2(n, s, 4)] * W[pm_idx4(m, n, l, s)];
                }
            }
            e_two = e_two - 0.5 * PAB[pm_idx2(m, l, 4)] * K;
        }
    }

    const Dual3 e_core = pm_pair_core_repulsion(apA, apB, R, gamma_ss);
    return e_one + e_two + e_core;
}
