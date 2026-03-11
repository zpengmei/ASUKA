from __future__ import annotations


def test_soc_si_facade_matches_state_interaction_module():
    from asuka.soc import si
    from asuka.soc import state_interaction as si_mod

    assert set(si.__all__) == set(si_mod.__all__)
    for name in si_mod.__all__:
        assert hasattr(si, name)
        assert hasattr(si_mod, name)
        assert getattr(si, name) is getattr(si_mod, name)


def test_soc_state_interaction_key_entrypoints_present():
    from asuka.soc import si

    expected = {
        "SpinFreeState",
        "SOCIntegrals",
        "soc_state_interaction",
        "soc_state_interaction_from_Gm",
        "soc_state_interaction_rassi",
        "solve_spinfree_state_interaction",
    }
    for name in expected:
        assert name in si.__all__
        assert hasattr(si, name)
