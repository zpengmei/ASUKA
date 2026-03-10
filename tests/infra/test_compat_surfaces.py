from __future__ import annotations


def test_audit_surface_exports_cuda_main():
    from asuka.audit import run_cuda_audit
    from asuka.audit.cuda import main

    assert callable(run_cuda_audit)
    assert run_cuda_audit is main


def test_mcscf_sort_mo_compat_surface_is_callable():
    from asuka.mcscf import sort_mo, sort_mo_by_irrep

    assert callable(sort_mo)
    assert callable(sort_mo_by_irrep)
