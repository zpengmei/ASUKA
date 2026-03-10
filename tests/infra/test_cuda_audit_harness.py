from __future__ import annotations

from asuka.audit import cuda as cuda_audit
from asuka.kernels import kernel_report


def test_kernel_report_includes_hf_thc_and_caspt2_extensions():
    rep = kernel_report()
    exts = rep.get("extensions", {})
    assert isinstance(exts, dict)
    assert "hf_thc" in exts
    assert "caspt2" in exts


def test_resolve_workloads_uses_named_profiles_and_explicit_includes():
    assert cuda_audit._resolve_workloads("quick", []) == list(cuda_audit.QUICK_WORKLOADS)
    assert cuda_audit._resolve_workloads("full", []) == list(cuda_audit.FULL_WORKLOADS)
    assert cuda_audit._resolve_workloads("quick", ["caspt2", "hf_thc"]) == ["caspt2", "hf_thc"]


def test_summarize_results_ranks_time_and_memory_and_ignores_non_ok():
    summary = cuda_audit.summarize_results(
        [
            {
                "name": "slow_kernel",
                "family": "demo",
                "status": "ok",
                "timing": {"event_ms": {"median": 12.0}},
                "memory": {"delta": {"pool_total_bytes_delta": 2048}},
            },
            {
                "name": "large_kernel",
                "family": "demo",
                "status": "ok",
                "timing": {"wall_ms": {"median": 8.0}},
                "memory": {"peak_pool_total_bytes": 4096},
            },
            {
                "name": "workflow_case",
                "family": "workflow",
                "status": "ok",
                "timing": {"wall_s": {"t_total": 0.020}},
                "memory": {"peak_driver_used_bytes": 1024},
            },
            {
                "name": "failed_kernel",
                "family": "demo",
                "status": "error",
                "timing": {"event_ms": {"median": 100.0}},
                "memory": {"delta": {"pool_total_bytes_delta": 999999}},
            },
        ]
    )

    assert [row["name"] for row in summary["time_ranked"]] == [
        "workflow_case",
        "slow_kernel",
        "large_kernel",
    ]
    assert [row["name"] for row in summary["memory_ranked"]] == [
        "large_kernel",
        "slow_kernel",
        "workflow_case",
    ]


def test_csv_rows_from_text_skips_prologue_and_parses_header():
    text = """
Generating SQLite file /tmp/demo.sqlite
Processing [/tmp/demo.sqlite] ...
Time (%),Total Time (ns),Instances,Name
41.2,1504,1,kernel_a
30.7,1120,2,kernel_b
"""
    rows = cuda_audit._csv_rows_from_text(text)
    assert len(rows) == 2
    assert rows[0]["Name"] == "kernel_a"
    assert rows[1]["Instances"] == "2"


def test_summarize_nsys_trace_rows_extracts_streams_and_kernel_counts():
    rows = [
        {"Duration (ns)": "1120", "Strm": "7", "Reg/Trd": "20", "Name": "kernel_a"},
        {"Duration (ns)": "1440", "Strm": "7", "Reg/Trd": "", "Name": "[CUDA memcpy Device-to-Host]"},
        {"Duration (ns)": "1504", "Strm": "9", "Reg/Trd": "34", "Name": "kernel_b"},
    ]
    summary = cuda_audit._summarize_nsys_trace_rows(rows)
    assert summary["kernel_launches"] == 2
    assert summary["memcpy_events"] == 1
    assert summary["streams"] == [7, 9]
    assert summary["max_registers_per_thread"] == 34


def test_nsight_targets_union_time_and_memory_rankings_without_duplicates():
    summary = {
        "time_ranked": [
            {"name": "hf_df_jk_bq_cuda_ext"},
            {"name": "orbitals_eval_value"},
        ],
        "memory_ranked": [
            {"name": "orbitals_eval_value"},
            {"name": "caspt2_mltmv"},
        ],
    }
    targets = cuda_audit._nsight_targets_from_summary(summary, top_n=2)
    assert targets == ["hf_df_jk_bq_cuda_ext", "orbitals_eval_value", "caspt2_mltmv"]


def test_build_promotion_gate_snapshot_keeps_thresholds_and_workload_metrics():
    gate = cuda_audit._build_promotion_gate_snapshot(
        [
            {
                "name": "workflow_stilbene",
                "family": "workflow",
                "status": "ok",
                "timing": {"wall_s": {"t_total": 12.5, "t_grad": 8.2}},
                "memory": {"peak_driver_used_bytes": 1024, "peak_pool_total_bytes": 2048},
            },
            {
                "name": "hf_df_jk_bq_cuda_ext",
                "family": "hf_df_jk",
                "status": "ok",
                "timing": {"event_ms": {"median": 10.0, "jitter_pct": 4.0}},
                "memory": {"delta": {"pool_total_bytes_delta": 4096}},
            },
        ]
    )
    assert gate["policy"]["min_total_time_improvement_pct"] == 5.0
    assert gate["policy"]["min_gradient_time_improvement_pct"] == 7.0
    snap = {row["name"]: row for row in gate["snapshot"]}
    assert snap["workflow_stilbene"]["t_total_s"] == 12.5
    assert snap["workflow_stilbene"]["peak_driver_used_bytes"] == 1024
    assert snap["hf_df_jk_bq_cuda_ext"]["event_ms_median"] == 10.0
