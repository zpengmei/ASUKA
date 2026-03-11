from __future__ import annotations


def test_mrci_api_and_package_surface_match():
    import asuka.mrci as mrci
    from asuka.mrci import api as mrci_api

    assert set(mrci.__all__) == set(mrci_api.__all__)
    for name in mrci_api.__all__:
        assert hasattr(mrci, name)
        assert hasattr(mrci_api, name)
        assert getattr(mrci, name) is getattr(mrci_api, name)


def test_mrci_public_entrypoints_present():
    import asuka.mrci as mrci

    expected = {
        "mrci_from_ref",
        "mrci_states_from_ref",
        "mrci_grad_states_from_ref",
        "run_mrci_grad",
        "ic_mrcisd_kernel",
    }
    for name in expected:
        assert name in mrci.__all__
        assert hasattr(mrci, name)

