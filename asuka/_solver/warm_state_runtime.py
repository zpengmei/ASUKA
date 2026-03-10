from __future__ import annotations

import json
import os
from collections.abc import Callable
from typing import Any

import numpy as np

from .warm_state import (
    WARM_CUDA_MATVEC_BACKENDS,
    WARM_STATE_FORMAT_VERSION,
    warm_cas_metadata_from_jsonable,
    warm_cas_metadata_to_jsonable,
)


def warm_state_summary(state: dict[str, Any] | None) -> dict[str, Any] | None:
    if state is None:
        return None
    return {
        "format_version": int(state.get("format_version", WARM_STATE_FORMAT_VERSION)),
        "norb": int(state["norb"]),
        "nelec_total": int(state["nelec_total"]),
        "twos": int(state["twos"]),
        "nroots": int(state["nroots"]),
        "ncsf": int(state["ncsf"]),
        "wfnsym": None if state.get("wfnsym", None) is None else int(state["wfnsym"]),
        "orbsym": state.get("orbsym", None),
        "ne_constraints_key": state.get("ne_constraints_key", None),
        "ci_dtype": str(state.get("ci_dtype", "float64")),
        "ci_device": str(state.get("ci_device", "cpu")),
        "has_ci": bool(state.get("ci", None) is not None),
        "has_mo_coeff": bool(state.get("mo_coeff", None) is not None),
        "has_mo_occ": bool(state.get("mo_occ", None) is not None),
        "cas_metadata": state.get("cas_metadata", None),
    }


def allowed_ci_devices_for_backend(matvec_backend: str) -> tuple[str, ...]:
    if str(matvec_backend).strip().lower() in WARM_CUDA_MATVEC_BACKENDS:
        return ("cpu", "cuda")
    return ("cpu",)


def warm_state_ci0_if_compatible(
    *,
    state: dict[str, Any] | None,
    norb: int,
    nelec_total: int,
    twos: int,
    nroots: int,
    ncsf: int,
    orbsym_key: tuple[int, ...] | None,
    wfnsym: int | None,
    ne_constraints_key: tuple[tuple[int, int, int], ...] | None,
    matvec_backend: str,
    cas_metadata: dict[str, Any],
) -> tuple[list[np.ndarray] | None, str]:
    if state is None:
        return None, "no_warm_state"
    ci_stored = state.get("ci", None)
    if ci_stored is None:
        return None, "warm_state_has_no_ci"

    if int(state.get("norb", -1)) != int(norb):
        return None, "norb_mismatch"
    if int(state.get("nelec_total", -1)) != int(nelec_total):
        return None, "nelec_mismatch"
    if int(state.get("twos", -1)) != int(twos):
        return None, "twos_mismatch"
    if int(state.get("nroots", -1)) != int(nroots):
        return None, "nroots_mismatch"
    if int(state.get("ncsf", -1)) != int(ncsf):
        return None, "ci_size_mismatch"
    if state.get("wfnsym", None) != (None if wfnsym is None else int(wfnsym)):
        return None, "wfnsym_mismatch"
    if state.get("orbsym", None) != orbsym_key:
        return None, "orbsym_mismatch"
    if state.get("ne_constraints_key", None) != ne_constraints_key:
        return None, "constraints_mismatch"
    if state.get("cas_metadata", None) != cas_metadata:
        return None, "cas_metadata_mismatch"

    ci_dtype = str(state.get("ci_dtype", "float64")).strip().lower()
    if ci_dtype not in ("float64", "f8"):
        return None, "ci_dtype_mismatch"
    ci_device = str(state.get("ci_device", "cpu")).strip().lower()
    if ci_device not in allowed_ci_devices_for_backend(matvec_backend):
        return None, "ci_device_mismatch"

    ci_arr = np.asarray(ci_stored)
    if ci_arr.ndim != 2:
        return None, "ci_rank_mismatch"
    if ci_arr.shape != (int(nroots), int(ncsf)):
        return None, "ci_shape_mismatch"
    if np.asarray(ci_arr).dtype != np.float64:
        return None, "ci_array_dtype_mismatch"

    ci_out = [np.ascontiguousarray(ci_arr[i], dtype=np.float64) for i in range(int(nroots))]
    return ci_out, "applied"


def update_warm_state(
    *,
    prev_state: dict[str, Any] | None,
    normalize_ci0_fn: Callable[..., list[np.ndarray]],
    ci: Any,
    norb: int,
    nelec_total: int,
    twos: int,
    nroots: int,
    ncsf: int,
    orbsym_key: tuple[int, ...] | None,
    wfnsym: int | None,
    ne_constraints_key: tuple[tuple[int, int, int], ...] | None,
    cas_metadata: dict[str, Any],
    mo_coeff: Any | None,
    mo_occ: Any | None,
) -> dict[str, Any]:
    ci_rows = normalize_ci0_fn(ci, nroots=int(nroots), ncsf=int(ncsf))
    ci_mat = np.ascontiguousarray(np.vstack(ci_rows), dtype=np.float64)

    prev = prev_state if prev_state is not None else {}
    if mo_coeff is None:
        mo_coeff_arr = prev.get("mo_coeff", None)
    else:
        mo_coeff_arr = np.ascontiguousarray(np.asarray(mo_coeff, dtype=np.float64))
    if mo_occ is None:
        mo_occ_arr = prev.get("mo_occ", None)
    else:
        mo_occ_arr = np.ascontiguousarray(np.asarray(mo_occ))

    return {
        "format_version": int(WARM_STATE_FORMAT_VERSION),
        "norb": int(norb),
        "nelec_total": int(nelec_total),
        "twos": int(twos),
        "nroots": int(nroots),
        "ncsf": int(ncsf),
        "wfnsym": None if wfnsym is None else int(wfnsym),
        "orbsym": orbsym_key,
        "ne_constraints_key": ne_constraints_key,
        "ci_dtype": "float64",
        "ci_device": "cpu",
        "ci": ci_mat,
        "mo_coeff": mo_coeff_arr,
        "mo_occ": mo_occ_arr,
        "cas_metadata": dict(cas_metadata),
    }


def save_warm_state(
    path: str | os.PathLike[str],
    *,
    state: dict[str, Any] | None,
    include_ci: bool = True,
    include_mo: bool = True,
) -> str:
    if state is None:
        raise ValueError("no warm state is attached to this solver")

    payload: dict[str, Any] = {}
    meta = {
        "format_version": int(state.get("format_version", WARM_STATE_FORMAT_VERSION)),
        "norb": int(state["norb"]),
        "nelec_total": int(state["nelec_total"]),
        "twos": int(state["twos"]),
        "nroots": int(state["nroots"]),
        "ncsf": int(state["ncsf"]),
        "wfnsym": None if state.get("wfnsym", None) is None else int(state["wfnsym"]),
        "orbsym": None if state.get("orbsym", None) is None else list(state["orbsym"]),
        "ne_constraints_key": (
            None
            if state.get("ne_constraints_key", None) is None
            else [list(x) for x in state["ne_constraints_key"]]
        ),
        "ci_dtype": str(state.get("ci_dtype", "float64")),
        "ci_device": str(state.get("ci_device", "cpu")),
        "cas_metadata": warm_cas_metadata_to_jsonable(state.get("cas_metadata", None)),
    }
    payload["meta_json"] = np.asarray(
        json.dumps(meta, sort_keys=True, separators=(",", ":")),
        dtype=np.str_,
    )

    if bool(include_ci) and state.get("ci", None) is not None:
        payload["ci"] = np.ascontiguousarray(np.asarray(state["ci"], dtype=np.float64))
    if bool(include_mo) and state.get("mo_coeff", None) is not None:
        payload["mo_coeff"] = np.ascontiguousarray(np.asarray(state["mo_coeff"], dtype=np.float64))
    if bool(include_mo) and state.get("mo_occ", None) is not None:
        payload["mo_occ"] = np.ascontiguousarray(np.asarray(state["mo_occ"]))

    path_str = os.fspath(path)
    outdir = os.path.dirname(path_str)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    with open(path_str, "wb") as fobj:
        np.savez_compressed(fobj, **payload)
    return path_str


def load_warm_state(
    path: str | os.PathLike[str],
    *,
    require_ci: bool = False,
) -> dict[str, Any]:
    path_str = os.fspath(path)
    with np.load(path_str, allow_pickle=False) as data:
        if "meta_json" not in data:
            raise ValueError("warm state file is missing 'meta_json'")

        meta_raw = json.loads(str(np.asarray(data["meta_json"]).item()))
        if not isinstance(meta_raw, dict):
            raise ValueError("warm state metadata must decode to a dictionary")

        fmt_ver = int(meta_raw.get("format_version", -1))
        if fmt_ver != int(WARM_STATE_FORMAT_VERSION):
            raise ValueError(
                f"unsupported warm-state format version {fmt_ver}; expected {WARM_STATE_FORMAT_VERSION}"
            )

        ne_constraints_key = meta_raw.get("ne_constraints_key", None)
        ne_constraints_norm: tuple[tuple[int, int, int], ...] | None = None
        if ne_constraints_key is not None:
            items: list[tuple[int, int, int]] = []
            for x in ne_constraints_key:
                if len(x) != 3:
                    raise ValueError("invalid warm state ne_constraints_key entry")
                items.append((int(x[0]), int(x[1]), int(x[2])))
            ne_constraints_norm = tuple(items)

        ci_data = None
        if "ci" in data:
            ci_arr = np.asarray(data["ci"])
            if ci_arr.ndim != 2:
                raise ValueError("warm state 'ci' must be a 2D array")
            if ci_arr.dtype != np.float64:
                raise ValueError("warm state currently supports only float64 CI vectors")
            ci_data = np.ascontiguousarray(ci_arr, dtype=np.float64)
        if require_ci and ci_data is None:
            raise ValueError("warm state file does not contain CI data")

        state = {
            "format_version": int(fmt_ver),
            "norb": int(meta_raw["norb"]),
            "nelec_total": int(meta_raw["nelec_total"]),
            "twos": int(meta_raw["twos"]),
            "nroots": int(meta_raw["nroots"]),
            "ncsf": int(meta_raw["ncsf"]),
            "wfnsym": None if meta_raw.get("wfnsym", None) is None else int(meta_raw["wfnsym"]),
            "orbsym": (
                None
                if meta_raw.get("orbsym", None) is None
                else tuple(int(x) for x in np.asarray(meta_raw["orbsym"], dtype=np.int64).ravel().tolist())
            ),
            "ne_constraints_key": ne_constraints_norm,
            "ci_dtype": str(meta_raw.get("ci_dtype", "float64")).strip().lower(),
            "ci_device": str(meta_raw.get("ci_device", "cpu")).strip().lower(),
            "ci": ci_data,
            "mo_coeff": (
                None
                if "mo_coeff" not in data
                else np.ascontiguousarray(np.asarray(data["mo_coeff"], dtype=np.float64))
            ),
            "mo_occ": None if "mo_occ" not in data else np.ascontiguousarray(np.asarray(data["mo_occ"])),
            "cas_metadata": warm_cas_metadata_from_jsonable(meta_raw.get("cas_metadata", None)),
        }
        if state["ci_dtype"] not in ("float64", "f8"):
            raise ValueError(f"unsupported warm state ci_dtype={state['ci_dtype']!r}; expected float64")
        if state["ci_device"] not in ("cpu", "cuda"):
            raise ValueError(f"unsupported warm state ci_device={state['ci_device']!r}; expected cpu/cuda")

    if state["ci"] is not None:
        ci_shape = tuple(np.asarray(state["ci"]).shape)
        if ci_shape != (int(state["nroots"]), int(state["ncsf"])):
            raise ValueError(
                f"warm state CI shape mismatch: {ci_shape} vs ({state['nroots']}, {state['ncsf']})"
            )
    return state
