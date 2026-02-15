from __future__ import annotations


class CSFFCIRowOracle:
    """Former PySCF csf_fci-based reference oracle (removed).

    This class used to wrap `pyscf-forge`'s `csf_fci` implementation to provide an
    independent reference for tiny-system validation. ASUKA is now a standalone
    runtime, so this reference path is no longer shipped.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        _ = (args, kwargs)
        raise RuntimeError(
            "CSFFCIRowOracle is not available in the standalone ASUKA runtime. "
            "Use ASUKA's native GUGA oracles/contract backends for production, or "
            "keep this reference code in a downstream validation harness."
        )
