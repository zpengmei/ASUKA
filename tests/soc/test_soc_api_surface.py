from __future__ import annotations


def test_soc_api_and_package_surface_match():
    import asuka.soc as soc
    from asuka.soc import api as soc_api

    assert set(soc.__all__) == set(soc_api.__all__)
    for name in soc_api.__all__:
        assert hasattr(soc, name)
        assert hasattr(soc_api, name)
        assert getattr(soc, name) is getattr(soc_api, name)


def test_soc_public_entrypoints_present():
    import asuka.soc as soc

    expected = {
        "soc_state_interaction",
        "soc_state_interaction_rassi",
        "build_soc_ci_rhs_for_zvector",
        "solve_soc_ci_zvector_response",
        "has_soc_cuda",
    }
    for name in expected:
        assert name in soc.__all__
        assert hasattr(soc, name)

