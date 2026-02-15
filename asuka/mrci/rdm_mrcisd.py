from __future__ import annotations

"""RDM helpers for truncated (max_virt_e) MRCISD spaces.

Why this exists
---------------
cuGUGA's CSF RDM12 builders compute the 2-RDM from Gram matrices of
``T[pq] = E_pq |Ψ>`` vectors. This is exact only when the working CSF/DRT space
is closed under *one-body* generator actions.

For the common MRCISD truncation "virtual electrons <= max_virt_e", the
restricted DRT is *not* closed under ``E_pq``: applying ``E_pq`` to a state
with ``max_virt_e`` electrons in the external space can yield a state with
``max_virt_e+1`` external electrons.

To obtain the correct dm2 for a truncated MRCISD wavefunction, embed the CI
vector into an *augmented* DRT with ``max_virt_e+1`` and evaluate RDMs there.
The wavefunction coefficients of the augmented-only CSFs are zero, but they
appear in ``E_pq|Ψ>`` and contribute to the Gram matrix.
"""

from dataclasses import dataclass
from typing import Literal, Sequence, Tuple

import numpy as np

from asuka.cuguga.drt import DRT
from asuka.mrci.mrcisd import build_drt_mrcisd


RDMBackend = Literal["cuda", "cpu"]


@dataclass(frozen=True)
class MRCISDRDMWorkspace:
    """Precomputed embedding from a restricted MRCISD DRT into an augmented DRT.

    Attributes
    ----------
    drt_sub : DRT
        Restricted subspace DRT.
    drt_full : DRT
        Full (augmented) DRT to evaluate RDMs.
    sub_to_full : np.ndarray
        Mapping from subspace CSF index to full-space CSF index. Shape: (nsub,).
    restricted : bool
        Whether the DRT is restricted (requires embedding).
    c_full_buf : np.ndarray | None
        Reusable buffer for the full-space vector (restricted only).
    """

    drt_sub: DRT
    drt_full: DRT
    sub_to_full: np.ndarray  # (nsub,), int64
    restricted: bool
    c_full_buf: np.ndarray | None = None  # reusable embedding buffer (restricted only)


def _is_restricted(*, nelec: int, n_virt: int, max_virt_e: int) -> bool:
    nelec = int(nelec)
    n_virt = int(n_virt)
    max_virt_e = int(max_virt_e)
    if nelec < 0 or n_virt < 0 or max_virt_e < 0:
        raise ValueError("nelec/n_virt/max_virt_e must be non-negative")
    max_virt_e_eff = min(max_virt_e, nelec)
    return bool(max_virt_e_eff < min(nelec, 2 * n_virt))


def prepare_mrcisd_rdm_workspace(
    drt_sub: DRT,
    *,
    n_act: int,
    n_virt: int,
    nelec: int,
    twos: int,
    max_virt_e: int,
    orbsym: Sequence[int] | None = None,
    wfnsym: int | None = None,
) -> MRCISDRDMWorkspace:
    """Prepare an augmented-DRT embedding for correct RDM12 evaluation.

    Parameters mirror :func:`~asuka.mrci.mrcisd.build_drt_mrcisd`.

    Parameters
    ----------
    drt_sub : DRT
        Restricted subspace DRT.
    n_act : int
        Number of active orbitals.
    n_virt : int
        Number of virtual orbitals.
    nelec : int
        Number of electrons.
    twos : int
        Twice the total spin (2S).
    max_virt_e : int
        Maximum number of virtual electrons.
    orbsym : Sequence[int] | None, optional
        Orbital symmetries.
    wfnsym : int | None, optional
        Wavefunction symmetry.

    Returns
    -------
    MRCISDRDMWorkspace
        Workspace containing the embedding map.

    Raises
    ------
    ValueError
        If DRT parameters do not match provided arguments.
    """

    n_act = int(n_act)
    n_virt = int(n_virt)
    nelec = int(nelec)
    twos = int(twos)
    max_virt_e = int(max_virt_e)

    if int(drt_sub.norb) != n_act + n_virt:
        raise ValueError("drt_sub.norb does not match n_act+n_virt")
    if int(drt_sub.nelec) != nelec:
        raise ValueError("drt_sub.nelec does not match nelec")
    if int(drt_sub.twos_target) != twos:
        raise ValueError("drt_sub.twos_target does not match twos")

    restricted = _is_restricted(nelec=nelec, n_virt=n_virt, max_virt_e=max_virt_e)
    if not restricted:
        sub_to_full = np.arange(int(drt_sub.ncsf), dtype=np.int64)
        return MRCISDRDMWorkspace(
            drt_sub=drt_sub,
            drt_full=drt_sub,
            sub_to_full=sub_to_full,
            restricted=False,
            c_full_buf=None,
        )

    max_virt_e_eff = min(max_virt_e, nelec)
    drt_full = build_drt_mrcisd(
        n_act=n_act,
        n_virt=n_virt,
        nelec=nelec,
        twos=twos,
        orbsym=orbsym,
        wfnsym=wfnsym,
        max_virt_e=int(max_virt_e_eff) + 1,
    )

    from asuka.mrci.projected_hop import build_subspace_map  # noqa: PLC0415

    mapping = build_subspace_map(drt_full=drt_full, drt_sub=drt_sub)
    sub_to_full = np.asarray(mapping.sub_to_full, dtype=np.int64).ravel()
    if sub_to_full.shape != (int(drt_sub.ncsf),):
        raise RuntimeError("unexpected sub_to_full shape from build_subspace_map")
    c_full_buf = np.zeros(int(drt_full.ncsf), dtype=np.float64)
    return MRCISDRDMWorkspace(
        drt_sub=drt_sub,
        drt_full=drt_full,
        sub_to_full=sub_to_full,
        restricted=True,
        c_full_buf=c_full_buf,
    )


def make_rdm12_mrcisd(
    ws: MRCISDRDMWorkspace,
    ci_sub: np.ndarray,
    *,
    rdm_backend: RDMBackend = "cuda",
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute (dm1, dm2) for a truncated MRCISD CI vector (with augmentation if needed).

    Parameters
    ----------
    ws : MRCISDRDMWorkspace
        RDM workspace containing embedding info.
    ci_sub : np.ndarray
        CI vector in the subspace DRT.
    rdm_backend : {"cuda", "cpu"}, optional
        Backend for RDM evaluation. Default is "cuda".

    Returns
    -------
    dm1 : np.ndarray
        1-RDM. Shape: (norb, norb).
    dm2 : np.ndarray
        2-RDM. Shape: (norb, norb, norb, norb).
    """

    c_sub = np.asarray(ci_sub, dtype=np.float64).ravel()
    if c_sub.size != int(ws.drt_sub.ncsf):
        raise ValueError("ci_sub has wrong length for ws.drt_sub")

    if ws.restricted:
        c_full = ws.c_full_buf
        if c_full is None or int(getattr(c_full, "size", -1)) != int(ws.drt_full.ncsf):
            c_full = np.zeros(int(ws.drt_full.ncsf), dtype=np.float64)
        else:
            c_full.fill(0)
        c_full[ws.sub_to_full] = c_sub
    else:
        c_full = c_sub

    if str(rdm_backend).strip().lower() == "cuda":
        try:
            from asuka.cuda.rdm_gpu import make_rdm12_cuda  # noqa: PLC0415

            dm1, dm2 = make_rdm12_cuda(ws.drt_full, c_full)
            return np.asarray(dm1), np.asarray(dm2)
        except Exception:
            pass

    from asuka.rdm.stream import make_rdm12_streaming  # noqa: PLC0415

    dm1, dm2 = make_rdm12_streaming(ws.drt_full, c_full)
    return np.asarray(dm1), np.asarray(dm2)
