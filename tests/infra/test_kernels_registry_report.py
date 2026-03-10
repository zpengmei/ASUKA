import json


def test_kernels_registry_report_is_importable():
    import asuka.kernels as k

    rep = k.kernel_report()
    assert isinstance(rep, dict)
    assert "extensions" in rep
    assert isinstance(rep["extensions"], dict)

    # Must be JSON-serializable (used by CLI).
    _ = json.dumps(rep)

