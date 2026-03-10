from __future__ import annotations

from asuka.hf import df_jk


def test_hf_df_jk_qblock_tune_default_is_enabled(monkeypatch):
    monkeypatch.delenv("ASUKA_HF_K_EXT_TUNE_QBLOCK", raising=False)
    assert df_jk._hf_k_ext_tune_enabled() is True


def test_hf_df_jk_qblock_autotune_default_is_disabled(monkeypatch):
    monkeypatch.delenv("ASUKA_HF_K_EXT_AUTOTUNE_QBLOCK", raising=False)
    assert df_jk._hf_k_ext_autotune_enabled() is False


def test_release_cuda_ext_workspace_cache_preserves_qblock_tuning_cache():
    class _DummyWS:
        def __init__(self) -> None:
            self.released = False

        def release(self) -> None:
            self.released = True

    old_ws = dict(df_jk._HF_DF_JK_WS_BY_DEVICE)
    old_bq = dict(df_jk._HF_DF_JK_BQ_CACHE_BY_DEVICE)
    old_tune = {k: dict(v) for k, v in df_jk._HF_DF_JK_QBLOCK_TUNE_BY_DEVICE.items()}
    try:
        ws = _DummyWS()
        df_jk._HF_DF_JK_WS_BY_DEVICE.clear()
        df_jk._HF_DF_JK_BQ_CACHE_BY_DEVICE.clear()
        df_jk._HF_DF_JK_QBLOCK_TUNE_BY_DEVICE.clear()
        df_jk._HF_DF_JK_WS_BY_DEVICE[0] = ws
        df_jk._HF_DF_JK_BQ_CACHE_BY_DEVICE[0] = ((1, 2, 3, 4), object())
        df_jk._HF_DF_JK_QBLOCK_TUNE_BY_DEVICE[0] = {("mnq", 1): 192}

        df_jk.release_cuda_ext_workspace_cache()

        assert ws.released is True
        assert df_jk._HF_DF_JK_WS_BY_DEVICE == {}
        assert df_jk._HF_DF_JK_BQ_CACHE_BY_DEVICE == {}
        assert df_jk._HF_DF_JK_QBLOCK_TUNE_BY_DEVICE == {0: {("mnq", 1): 192}}
    finally:
        df_jk._HF_DF_JK_WS_BY_DEVICE.clear()
        df_jk._HF_DF_JK_WS_BY_DEVICE.update(old_ws)
        df_jk._HF_DF_JK_BQ_CACHE_BY_DEVICE.clear()
        df_jk._HF_DF_JK_BQ_CACHE_BY_DEVICE.update(old_bq)
        df_jk._HF_DF_JK_QBLOCK_TUNE_BY_DEVICE.clear()
        df_jk._HF_DF_JK_QBLOCK_TUNE_BY_DEVICE.update(old_tune)
