from __future__ import annotations

from typing import Any

import numpy as np

WARM_STATE_FORMAT_VERSION = 1
WARM_CUDA_MATVEC_BACKENDS = frozenset(
    ("cuda_eri_mat", "cuda", "cuda_direct", "cuda_fixed_ell", "cuda_ell", "cuda_fixed_sell", "cuda_sell")
)


def normalize_warm_cas_metadata(
    metadata: dict[str, Any] | None,
    *,
    default_ncas: int,
    default_nelecas: tuple[int, int],
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "ncas": int(default_ncas),
        "nelecas": (int(default_nelecas[0]), int(default_nelecas[1])),
    }
    if metadata is None:
        return out
    if not isinstance(metadata, dict):
        raise TypeError("warm_state_context must be a dict or None")

    if metadata.get("ncore", None) is not None:
        out["ncore"] = int(metadata["ncore"])
    if metadata.get("ncas", None) is not None:
        out["ncas"] = int(metadata["ncas"])
    if metadata.get("nelecas", None) is not None:
        nelecas_arr = np.asarray(metadata["nelecas"], dtype=np.int64).ravel()
        if nelecas_arr.size != 2:
            raise ValueError("warm_state_context['nelecas'] must contain exactly two integers")
        out["nelecas"] = (int(nelecas_arr[0]), int(nelecas_arr[1]))
    if metadata.get("cas_orbsym", None) is not None:
        out["cas_orbsym"] = tuple(int(x) for x in np.asarray(metadata["cas_orbsym"], dtype=np.int64).ravel().tolist())
    if metadata.get("active_orbital_indices", None) is not None:
        out["active_orbital_indices"] = tuple(
            int(x) for x in np.asarray(metadata["active_orbital_indices"], dtype=np.int64).ravel().tolist()
        )
    return out


def warm_cas_metadata_to_jsonable(metadata: dict[str, Any] | None) -> dict[str, Any] | None:
    if metadata is None:
        return None
    out: dict[str, Any] = {}
    for key, val in metadata.items():
        if isinstance(val, tuple):
            out[str(key)] = list(val)
        else:
            out[str(key)] = val
    return out


def warm_cas_metadata_from_jsonable(metadata: Any) -> dict[str, Any] | None:
    if metadata is None:
        return None
    if not isinstance(metadata, dict):
        raise ValueError("warm state cas_metadata must be a dictionary")

    out: dict[str, Any] = {}
    if metadata.get("ncore", None) is not None:
        out["ncore"] = int(metadata["ncore"])
    if metadata.get("ncas", None) is not None:
        out["ncas"] = int(metadata["ncas"])
    if metadata.get("nelecas", None) is not None:
        nelecas_arr = np.asarray(metadata["nelecas"], dtype=np.int64).ravel()
        if nelecas_arr.size != 2:
            raise ValueError("warm state cas_metadata['nelecas'] must contain exactly two integers")
        out["nelecas"] = (int(nelecas_arr[0]), int(nelecas_arr[1]))
    if metadata.get("cas_orbsym", None) is not None:
        out["cas_orbsym"] = tuple(int(x) for x in np.asarray(metadata["cas_orbsym"], dtype=np.int64).ravel().tolist())
    if metadata.get("active_orbital_indices", None) is not None:
        out["active_orbital_indices"] = tuple(
            int(x) for x in np.asarray(metadata["active_orbital_indices"], dtype=np.int64).ravel().tolist()
        )
    return out if out else None
