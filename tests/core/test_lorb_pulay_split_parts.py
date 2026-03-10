from __future__ import annotations

import numpy as np

from asuka.hf import df_scf as _df_scf
from asuka.mcscf.nuc_grad_df import _build_dme0_lorb_response


def _rand_sym(rng: np.random.Generator, n: int) -> np.ndarray:
    a = rng.standard_normal((n, n))
    return np.asarray(0.5 * (a + a.T), dtype=np.float64)


def test_lorb_pulay_parts_jk_split_consistency_cpu() -> None:
    rng = np.random.default_rng(3)
    nao = 4
    nmo = 4
    ncore = 1
    ncas = 2
    nocc = ncore + ncas
    naux = 5

    C_raw = rng.standard_normal((nao, nmo))
    C, _ = np.linalg.qr(C_raw)
    C = np.asarray(C, dtype=np.float64)

    B_ao = rng.standard_normal((nao, nao, naux))
    B_ao = np.asarray(0.5 * (B_ao + B_ao.transpose(1, 0, 2)), dtype=np.float64)
    h_ao = _rand_sym(rng, nao)

    L = rng.standard_normal((nmo, nmo))
    dm1 = _rand_sym(rng, ncas)
    dm2 = rng.standard_normal((ncas, ncas, ncas, ncas))
    ppaa = rng.standard_normal((nmo, nmo, ncas, ncas))
    papa = rng.standard_normal((nmo, ncas, nmo, ncas))

    dme0, parts = _build_dme0_lorb_response(
        B_ao,
        h_ao,
        C,
        L,
        dm1,
        dm2,
        ppaa,
        papa,
        ncore=int(ncore),
        ncas=int(ncas),
        return_parts=True,
    )
    dme0 = np.asarray(dme0, dtype=np.float64)
    p = {str(k): np.asarray(v, dtype=np.float64) for k, v in dict(parts).items()}

    np.testing.assert_allclose(dme0, p["dme0_total"], atol=1.0e-10, rtol=1.0e-10)
    np.testing.assert_allclose(
        p["dme0_mean_vmix"],
        p["dme0_mean_vmix_j"] + p["dme0_mean_vmix_k"],
        atol=1.0e-10,
        rtol=1.0e-10,
    )
    np.testing.assert_allclose(
        p["dme0_mean_vmix_j"],
        p["dme0_mean_vmix_j_core"] + p["dme0_mean_vmix_j_act"],
        atol=1.0e-10,
        rtol=1.0e-10,
    )
    np.testing.assert_allclose(
        p["dme0_mean_vmix_k"],
        p["dme0_mean_vmix_k_core"] + p["dme0_mean_vmix_k_act"],
        atol=1.0e-10,
        rtol=1.0e-10,
    )
    np.testing.assert_allclose(
        p["dme0_mean_vLmix"],
        p["dme0_mean_vLmix_j"] + p["dme0_mean_vLmix_k"],
        atol=1.0e-10,
        rtol=1.0e-10,
    )
    np.testing.assert_allclose(
        p["dme0_mean_vL_c"],
        p["dme0_mean_vL_c_j"] + p["dme0_mean_vL_c_k"],
        atol=1.0e-10,
        rtol=1.0e-10,
    )
    np.testing.assert_allclose(
        p["dme0_mean_vc_L"],
        p["dme0_mean_vc_L_j"] + p["dme0_mean_vc_L_k"],
        atol=1.0e-10,
        rtol=1.0e-10,
    )
    np.testing.assert_allclose(
        p["dme0_mean"],
        p["dme0_mean_j"] + p["dme0_mean_k"],
        atol=1.0e-10,
        rtol=1.0e-10,
    )
    np.testing.assert_allclose(
        p["dme0_aa1"],
        p["dme0_aa1_j"] + p["dme0_aa1_k"],
        atol=1.0e-10,
        rtol=1.0e-10,
    )

    # Sanity: ppaa/papa active slices are the expected dimensions.
    assert ppaa[:, :, :ncas, :ncas].shape == (nmo, nmo, ncas, ncas)
    assert papa[:, :ncas, :, :ncas].shape == (nmo, ncas, nmo, ncas)
    assert int(nocc) <= int(nmo)


def test_lorb_pulay_parts_with_vhf_cache_keeps_base_channels() -> None:
    rng = np.random.default_rng(11)
    nao = 4
    nmo = 4
    ncore = 1
    ncas = 2
    nocc = ncore + ncas
    naux = 4

    C_raw = rng.standard_normal((nao, nmo))
    C, _ = np.linalg.qr(C_raw)
    C = np.asarray(C, dtype=np.float64)

    B_ao = rng.standard_normal((nao, nao, naux))
    B_ao = np.asarray(0.5 * (B_ao + B_ao.transpose(1, 0, 2)), dtype=np.float64)
    h_ao = _rand_sym(rng, nao)

    L = rng.standard_normal((nmo, nmo))
    dm1 = _rand_sym(rng, ncas)
    dm2 = rng.standard_normal((ncas, ncas, ncas, ncas))
    ppaa = rng.standard_normal((nmo, nmo, ncas, ncas))
    papa = rng.standard_normal((nmo, ncas, nmo, ncas))

    C_core = C[:, :ncore]
    C_act = C[:, ncore:nocc]
    D_core = 2.0 * (C_core @ C_core.T)
    D_act = C_act @ dm1 @ C_act.T
    Jc, Kc = _df_scf._df_JK(B_ao, D_core, want_J=True, want_K=True)  # noqa: SLF001
    Ja, Ka = _df_scf._df_JK(B_ao, D_act, want_J=True, want_K=True)  # noqa: SLF001
    vhf_cache = {
        "vhf_c": np.asarray(Jc - 0.5 * Kc, dtype=np.float64),
        "vhf_a": np.asarray(Ja - 0.5 * Ka, dtype=np.float64),
    }

    dme0, parts = _build_dme0_lorb_response(
        B_ao,
        h_ao,
        C,
        L,
        dm1,
        dm2,
        ppaa,
        papa,
        ncore=int(ncore),
        ncas=int(ncas),
        vhf_cache=vhf_cache,
        return_parts=True,
    )
    dme0 = np.asarray(dme0, dtype=np.float64)
    p = {str(k): np.asarray(v, dtype=np.float64) for k, v in dict(parts).items()}

    np.testing.assert_allclose(dme0, p["dme0_total"], atol=1.0e-10, rtol=1.0e-10)
    assert "dme0_mean_vL_c" in p
    assert "dme0_aa1_j" in p
    # J/K mean-field split channels are unavailable when vhf_cache is passed.
    assert "dme0_mean_vL_c_j" not in p


def test_lorb_pulay_parts_accept_asym_debug_modes(monkeypatch) -> None:
    rng = np.random.default_rng(19)
    nao = 4
    nmo = 4
    ncore = 1
    ncas = 2
    naux = 5

    C_raw = rng.standard_normal((nao, nmo))
    C, _ = np.linalg.qr(C_raw)
    C = np.asarray(C, dtype=np.float64)

    B_ao = rng.standard_normal((nao, nao, naux))
    B_ao = np.asarray(0.5 * (B_ao + B_ao.transpose(1, 0, 2)), dtype=np.float64)
    h_ao = _rand_sym(rng, nao)
    L = rng.standard_normal((nmo, nmo))
    dm1 = _rand_sym(rng, ncas)
    dm2 = rng.standard_normal((ncas, ncas, ncas, ncas))
    ppaa = rng.standard_normal((nmo, nmo, ncas, ncas))
    papa = rng.standard_normal((nmo, ncas, nmo, ncas))

    monkeypatch.setenv("ASUKA_CASPT2_LORB_DML_SYM_MODE", "core_asym")
    monkeypatch.setenv("ASUKA_CASPT2_LORB_VMIX_DML_CORE_MODE", "asym")
    monkeypatch.setenv("ASUKA_CASPT2_LORB_VMIX_DML_SPLIT_MODE", "j_sym_k_asym")
    try:
        dme0, parts = _build_dme0_lorb_response(
            B_ao,
            h_ao,
            C,
            L,
            dm1,
            dm2,
            ppaa,
            papa,
            ncore=int(ncore),
            ncas=int(ncas),
            return_parts=True,
        )
    finally:
        monkeypatch.delenv("ASUKA_CASPT2_LORB_DML_SYM_MODE", raising=False)
        monkeypatch.delenv("ASUKA_CASPT2_LORB_VMIX_DML_CORE_MODE", raising=False)
        monkeypatch.delenv("ASUKA_CASPT2_LORB_VMIX_DML_SPLIT_MODE", raising=False)

    d = np.asarray(dme0, dtype=np.float64)
    p = {str(k): np.asarray(v, dtype=np.float64) for k, v in dict(parts).items()}
    np.testing.assert_allclose(d, p["dme0_total"], atol=1.0e-10, rtol=1.0e-10)
    np.testing.assert_allclose(
        p["dme0_mean_vmix"],
        p["dme0_mean_vmix_j"] + p["dme0_mean_vmix_k"],
        atol=1.0e-10,
        rtol=1.0e-10,
    )
    assert np.isfinite(d).all()


def test_lorb_pulay_parts_vmix_mm_mode_changes_k_channel(monkeypatch) -> None:
    rng = np.random.default_rng(31)
    nao = 4
    nmo = 4
    ncore = 1
    ncas = 2
    naux = 4

    C_raw = rng.standard_normal((nao, nmo))
    C, _ = np.linalg.qr(C_raw)
    C = np.asarray(C, dtype=np.float64)

    B_ao = rng.standard_normal((nao, nao, naux))
    B_ao = np.asarray(0.5 * (B_ao + B_ao.transpose(1, 0, 2)), dtype=np.float64)
    h_ao = _rand_sym(rng, nao)
    L = rng.standard_normal((nmo, nmo))
    dm1 = _rand_sym(rng, ncas)
    dm2 = rng.standard_normal((ncas, ncas, ncas, ncas))
    ppaa = rng.standard_normal((nmo, nmo, ncas, ncas))
    papa = rng.standard_normal((nmo, ncas, nmo, ncas))

    monkeypatch.setenv("ASUKA_CASPT2_LORB_PARTS_INCLUDE_RAW", "1")
    try:
        _, parts_left = _build_dme0_lorb_response(
            B_ao,
            h_ao,
            C,
            L,
            dm1,
            dm2,
            ppaa,
            papa,
            ncore=int(ncore),
            ncas=int(ncas),
            return_parts=True,
        )
        p_left = {str(k): np.asarray(v, dtype=np.float64) for k, v in dict(parts_left).items()}

        monkeypatch.setenv("ASUKA_CASPT2_LORB_VMIX_K_MM_MODE", "right")
        monkeypatch.setenv("ASUKA_CASPT2_LORB_VMIX_J_MM_MODE", "dl")
        dme0_right, parts_right = _build_dme0_lorb_response(
            B_ao,
            h_ao,
            C,
            L,
            dm1,
            dm2,
            ppaa,
            papa,
            ncore=int(ncore),
            ncas=int(ncas),
            return_parts=True,
        )
    finally:
        monkeypatch.delenv("ASUKA_CASPT2_LORB_PARTS_INCLUDE_RAW", raising=False)
        monkeypatch.delenv("ASUKA_CASPT2_LORB_VMIX_K_MM_MODE", raising=False)
        monkeypatch.delenv("ASUKA_CASPT2_LORB_VMIX_J_MM_MODE", raising=False)

    d_right = np.asarray(dme0_right, dtype=np.float64)
    p_right = {str(k): np.asarray(v, dtype=np.float64) for k, v in dict(parts_right).items()}
    np.testing.assert_allclose(d_right, p_right["dme0_total"], atol=1.0e-10, rtol=1.0e-10)
    np.testing.assert_allclose(
        p_right["dme0_mean_vmix"],
        p_right["dme0_mean_vmix_j"] + p_right["dme0_mean_vmix_k"],
        atol=1.0e-10,
        rtol=1.0e-10,
    )
    assert np.isfinite(d_right).all()
    # The symmetric Pulay component may stay unchanged, but raw vmix-k
    # should change when switching left-multiply to right-multiply.
    assert "raw_g_vmix_k" in p_left and "raw_g_vmix_k" in p_right
    assert not np.allclose(p_left["raw_g_vmix_k"], p_right["raw_g_vmix_k"], atol=1.0e-12, rtol=1.0e-12)


def test_lorb_pulay_parts_raw_dump_keys(monkeypatch) -> None:
    rng = np.random.default_rng(23)
    nao = 4
    nmo = 4
    ncore = 1
    ncas = 2
    naux = 4

    C_raw = rng.standard_normal((nao, nmo))
    C, _ = np.linalg.qr(C_raw)
    C = np.asarray(C, dtype=np.float64)

    B_ao = rng.standard_normal((nao, nao, naux))
    B_ao = np.asarray(0.5 * (B_ao + B_ao.transpose(1, 0, 2)), dtype=np.float64)
    h_ao = _rand_sym(rng, nao)
    L = rng.standard_normal((nmo, nmo))
    dm1 = _rand_sym(rng, ncas)
    dm2 = rng.standard_normal((ncas, ncas, ncas, ncas))
    ppaa = rng.standard_normal((nmo, nmo, ncas, ncas))
    papa = rng.standard_normal((nmo, ncas, nmo, ncas))

    monkeypatch.setenv("ASUKA_CASPT2_LORB_PARTS_INCLUDE_RAW", "1")
    try:
        dme0, parts = _build_dme0_lorb_response(
            B_ao,
            h_ao,
            C,
            L,
            dm1,
            dm2,
            ppaa,
            papa,
            ncore=int(ncore),
            ncas=int(ncas),
            return_parts=True,
        )
    finally:
        monkeypatch.delenv("ASUKA_CASPT2_LORB_PARTS_INCLUDE_RAW", raising=False)

    d = np.asarray(dme0, dtype=np.float64)
    p = {str(k): np.asarray(v, dtype=np.float64) for k, v in dict(parts).items()}
    np.testing.assert_allclose(d, p["dme0_total"], atol=1.0e-10, rtol=1.0e-10)
    for key in (
        "raw_D_core",
        "raw_D_act",
        "raw_D_L_core_raw",
        "raw_D_L_core_sym",
        "raw_D_L_core_vmix",
        "raw_D_L_core_vmix_j",
        "raw_D_L_core_vmix_k",
        "raw_D_L_core_vmix_j_act",
        "raw_D_L_core_vmix_k_act",
        "raw_D_L_total",
        "raw_g_vmix",
        "raw_g_vLmix",
        "raw_g_vL_c",
        "raw_g_vc_L",
    ):
        assert key in p
        assert p[key].shape == (nao, nao)


def test_lorb_pulay_parts_k_act_oitd_builds_oitd_with_full_dml(monkeypatch) -> None:
    rng = np.random.default_rng(41)
    nao = 5
    nmo = 5
    ncore = 2
    ncas = 2
    naux = 6

    C_raw = rng.standard_normal((nao, nmo))
    C, _ = np.linalg.qr(C_raw)
    C = np.asarray(C, dtype=np.float64)

    B_ao = rng.standard_normal((nao, nao, naux))
    B_ao = np.asarray(0.5 * (B_ao + B_ao.transpose(1, 0, 2)), dtype=np.float64)
    h_ao = _rand_sym(rng, nao)
    L = rng.standard_normal((nmo, nmo))
    dm1 = _rand_sym(rng, ncas)
    dm2 = rng.standard_normal((ncas, ncas, ncas, ncas))
    ppaa = rng.standard_normal((nmo, nmo, ncas, ncas))
    papa = rng.standard_normal((nmo, ncas, nmo, ncas))

    monkeypatch.setenv("ASUKA_CASPT2_LORB_PARTS_INCLUDE_RAW", "1")
    monkeypatch.setenv("ASUKA_CASPT2_LORB_DML_SYM_MODE", "full")
    monkeypatch.setenv("ASUKA_CASPT2_LORB_VMIX_DML_K_ACT_MODE", "oitd")
    try:
        _, parts = _build_dme0_lorb_response(
            B_ao,
            h_ao,
            C,
            L,
            dm1,
            dm2,
            ppaa,
            papa,
            ncore=int(ncore),
            ncas=int(ncas),
            return_parts=True,
        )
    finally:
        monkeypatch.delenv("ASUKA_CASPT2_LORB_PARTS_INCLUDE_RAW", raising=False)
        monkeypatch.delenv("ASUKA_CASPT2_LORB_DML_SYM_MODE", raising=False)
        monkeypatch.delenv("ASUKA_CASPT2_LORB_VMIX_DML_K_ACT_MODE", raising=False)

    p = {str(k): np.asarray(v, dtype=np.float64) for k, v in dict(parts).items()}
    assert "raw_D_L_core_vmix_k_act" in p

    d_core_mo = np.zeros((nmo, nmo), dtype=np.float64)
    d_core_mo[:ncore, :ncore] = 2.0 * np.eye(ncore, dtype=np.float64)
    expected = np.asarray(C @ (d_core_mo @ L.T - L.T @ d_core_mo) @ C.T, dtype=np.float64)
    np.testing.assert_allclose(
        p["raw_D_L_core_vmix_k_act"],
        expected,
        atol=1.0e-10,
        rtol=1.0e-10,
    )
    assert float(np.linalg.norm(p["raw_D_L_core_vmix_k_act"])) > 1.0e-10


def test_lorb_pulay_parts_k_act_none_zeros_channel(monkeypatch) -> None:
    rng = np.random.default_rng(43)
    nao = 4
    nmo = 4
    ncore = 1
    ncas = 2
    naux = 5

    C_raw = rng.standard_normal((nao, nmo))
    C, _ = np.linalg.qr(C_raw)
    C = np.asarray(C, dtype=np.float64)

    B_ao = rng.standard_normal((nao, nao, naux))
    B_ao = np.asarray(0.5 * (B_ao + B_ao.transpose(1, 0, 2)), dtype=np.float64)
    h_ao = _rand_sym(rng, nao)
    L = rng.standard_normal((nmo, nmo))
    dm1 = _rand_sym(rng, ncas)
    dm2 = rng.standard_normal((ncas, ncas, ncas, ncas))
    ppaa = rng.standard_normal((nmo, nmo, ncas, ncas))
    papa = rng.standard_normal((nmo, ncas, nmo, ncas))

    monkeypatch.setenv("ASUKA_CASPT2_LORB_PARTS_INCLUDE_RAW", "1")
    monkeypatch.setenv("ASUKA_CASPT2_LORB_VMIX_DML_K_ACT_MODE", "none")
    try:
        _, parts = _build_dme0_lorb_response(
            B_ao,
            h_ao,
            C,
            L,
            dm1,
            dm2,
            ppaa,
            papa,
            ncore=int(ncore),
            ncas=int(ncas),
            return_parts=True,
        )
    finally:
        monkeypatch.delenv("ASUKA_CASPT2_LORB_PARTS_INCLUDE_RAW", raising=False)
        monkeypatch.delenv("ASUKA_CASPT2_LORB_VMIX_DML_K_ACT_MODE", raising=False)

    p = {str(k): np.asarray(v, dtype=np.float64) for k, v in dict(parts).items()}
    np.testing.assert_allclose(
        p["dme0_mean_vmix_k_act"],
        np.zeros_like(p["dme0_mean_vmix_k_act"]),
        atol=1.0e-12,
        rtol=1.0e-12,
    )
    np.testing.assert_allclose(
        p["raw_D_L_core_vmix_k_act"],
        np.zeros_like(p["raw_D_L_core_vmix_k_act"]),
        atol=1.0e-12,
        rtol=1.0e-12,
    )
