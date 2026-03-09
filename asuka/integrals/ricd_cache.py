from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from asuka.integrals.ricd_types import RICDOptions, RICDShell

_RICD_DISK_CACHE_VERSION = 1


def _asuka_version_str() -> str:
    # Avoid importing `asuka` here to prevent circular imports.
    try:
        from importlib.metadata import PackageNotFoundError, version  # noqa: PLC0415

        return str(version("asuka"))
    except PackageNotFoundError:
        return "0.0.0"
    except Exception:
        return "unknown"


def _ricd_cache_root() -> Path:
    override = os.getenv("ASUKA_RICD_CACHE_DIR")
    if override:
        return Path(override).expanduser()
    base = os.getenv("XDG_CACHE_HOME")
    if base:
        root = Path(base).expanduser()
    else:
        root = Path.home() / ".cache"
    return root / "asuka" / "ricd"


def _symbol_from_type_key(type_key: str) -> str:
    sym = str(type_key).split("|", 1)[0].strip()
    return sym or "X"


def _type_cache_meta(
    type_key: str,
    *,
    tau_t: float,
    opts: RICDOptions,
) -> dict[str, Any]:
    # Use exact float encodings to make fingerprints stable and unambiguous.
    return {
        "cache_version": int(_RICD_DISK_CACHE_VERSION),
        "type_key": str(type_key),
        "mode": str(opts.mode),
        "tau_t_hex": float(tau_t).hex(),
        "skip_high_ac": bool(opts.skip_high_ac),
        "primitive_threshold_ratio_hex": float(opts.primitive_threshold_ratio).hex(),
        "primitive_retry_halves": int(opts.primitive_retry_halves),
        "renorm_rel_factor_hex": float(opts.renorm_rel_factor).hex(),
        "renorm_abs_floor_hex": float(opts.renorm_abs_floor).hex(),
        "dccd": bool(opts.dccd),
    }


def _type_cache_fingerprint(meta: dict[str, Any]) -> str:
    txt = json.dumps(meta, sort_keys=True, separators=(",", ":"), default=str)
    h = hashlib.blake2b(digest_size=20)
    h.update(txt.encode("utf-8", errors="strict"))
    return h.hexdigest()


def ricd_type_cache_path(
    type_key: str,
    *,
    tau_t: float,
    opts: RICDOptions,
) -> Path:
    meta = _type_cache_meta(type_key, tau_t=float(tau_t), opts=opts)
    digest = _type_cache_fingerprint(meta)
    sym = _symbol_from_type_key(type_key)
    sub = digest[:2]
    return _ricd_cache_root() / "types" / sym / sub / f"{sym}_{digest}.npz"


def _shells_to_arrays(shells: list[RICDShell]) -> dict[str, np.ndarray]:
    if not shells:
        return {
            "shell_l": np.empty(0, dtype=np.int32),
            "shell_prim_start": np.empty(0, dtype=np.int32),
            "shell_nprim": np.empty(0, dtype=np.int32),
            "prim_exp": np.empty(0, dtype=np.float64),
            "prim_coef": np.empty(0, dtype=np.float64),
        }

    shell_l = np.asarray([int(sh.l) for sh in shells], dtype=np.int32)
    shell_nprim = np.asarray([int(np.asarray(sh.prim_exp).size) for sh in shells], dtype=np.int32)
    shell_prim_start = np.zeros(int(shell_nprim.size), dtype=np.int32)
    if int(shell_nprim.size) > 1:
        shell_prim_start[1:] = np.cumsum(shell_nprim[:-1], dtype=np.int64).astype(np.int32, copy=False)

    prim_exp = np.concatenate([np.asarray(sh.prim_exp, dtype=np.float64).ravel() for sh in shells], axis=0)
    prim_coef = np.concatenate([np.asarray(sh.prim_coef, dtype=np.float64).ravel() for sh in shells], axis=0)

    return {
        "shell_l": np.asarray(shell_l, dtype=np.int32),
        "shell_prim_start": np.asarray(shell_prim_start, dtype=np.int32),
        "shell_nprim": np.asarray(shell_nprim, dtype=np.int32),
        "prim_exp": np.asarray(prim_exp, dtype=np.float64),
        "prim_coef": np.asarray(prim_coef, dtype=np.float64),
    }


def _arrays_to_shells(type_key: str, arrays: Mapping[str, Any]) -> list[RICDShell]:
    shell_l = np.asarray(arrays["shell_l"], dtype=np.int32).ravel()
    shell_prim_start = np.asarray(arrays["shell_prim_start"], dtype=np.int32).ravel()
    shell_nprim = np.asarray(arrays["shell_nprim"], dtype=np.int32).ravel()
    prim_exp = np.asarray(arrays["prim_exp"], dtype=np.float64).ravel()
    prim_coef = np.asarray(arrays["prim_coef"], dtype=np.float64).ravel()

    n_shell = int(shell_l.size)
    if int(shell_prim_start.size) != n_shell or int(shell_nprim.size) != n_shell:
        raise ValueError("RICD cache arrays have inconsistent shell dimensions")

    shells: list[RICDShell] = []
    for ish in range(n_shell):
        ps = int(shell_prim_start[ish])
        np_ = int(shell_nprim[ish])
        if ps < 0 or np_ < 0 or ps + np_ > int(prim_exp.size):
            raise ValueError("RICD cache arrays have invalid primitive offsets")
        shells.append(RICDShell(
            atom_type_key=str(type_key),
            l=int(shell_l[ish]),
            prim_exp=np.asarray(prim_exp[ps: ps + np_], dtype=np.float64).copy(),
            prim_coef=np.asarray(prim_coef[ps: ps + np_], dtype=np.float64).copy(),
        ))
    return shells


def load_ricd_type_cache(
    type_key: str,
    *,
    tau_t: float,
    opts: RICDOptions,
) -> tuple[list[RICDShell], dict[str, Any]] | None:
    """Load cached per-atom-type RICD shells, or return None on miss/error."""
    path = ricd_type_cache_path(type_key, tau_t=float(tau_t), opts=opts)
    expected_meta = _type_cache_meta(type_key, tau_t=float(tau_t), opts=opts)
    expected_fp = _type_cache_fingerprint(expected_meta)

    try:
        with np.load(os.fspath(path), allow_pickle=False) as data:
            if "meta_json" not in data:
                return None
            meta = json.loads(str(np.asarray(data["meta_json"]).item()))
            if not isinstance(meta, dict):
                return None
            if int(meta.get("cache_version", -1)) != int(_RICD_DISK_CACHE_VERSION):
                return None
            if str(meta.get("fingerprint", "")) != str(expected_fp):
                return None
            if str(meta.get("type_key", "")) != str(type_key):
                return None
            arrays = {
                "shell_l": np.asarray(data["shell_l"], dtype=np.int32),
                "shell_prim_start": np.asarray(data["shell_prim_start"], dtype=np.int32),
                "shell_nprim": np.asarray(data["shell_nprim"], dtype=np.int32),
                "prim_exp": np.asarray(data["prim_exp"], dtype=np.float64),
                "prim_coef": np.asarray(data["prim_coef"], dtype=np.float64),
            }
    except FileNotFoundError:
        return None
    except Exception:
        return None

    try:
        shells = _arrays_to_shells(type_key, arrays)
    except Exception:
        return None
    return shells, meta


def save_ricd_type_cache(
    type_key: str,
    shells: list[RICDShell],
    *,
    tau_t: float,
    opts: RICDOptions,
    stats: dict[str, Any] | None = None,
) -> Path | None:
    """Save per-atom-type RICD shells to the on-disk cache (best-effort)."""
    path = ricd_type_cache_path(type_key, tau_t=float(tau_t), opts=opts)
    meta = _type_cache_meta(type_key, tau_t=float(tau_t), opts=opts)
    meta["fingerprint"] = _type_cache_fingerprint(meta)
    meta["asuka_version"] = _asuka_version_str()
    if stats is not None:
        meta["stats"] = stats

    arrays = _shells_to_arrays(shells)
    payload: dict[str, Any] = dict(arrays)
    payload["meta_json"] = np.asarray(json.dumps(meta, sort_keys=True, separators=(",", ":"), default=str), dtype=np.str_)

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(os.fspath(tmp), "wb") as fobj:
            np.savez_compressed(fobj, **payload)
        os.replace(os.fspath(tmp), os.fspath(path))
        return path
    except Exception:
        return None


__all__ = [
    "load_ricd_type_cache",
    "ricd_type_cache_path",
    "save_ricd_type_cache",
]
