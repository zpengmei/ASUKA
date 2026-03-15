#!/usr/bin/env python
"""Benchmark SCI-CASCI energies: HF vs CASSCF MOs, CAS(18,18) vs CAS(22,22)."""
import time
import numpy as np

from asuka.frontend import Molecule
from asuka.frontend.scf import run_hf_df
from asuka.mcscf import run_casscf
from asuka.mcscf.casci import run_casci_df
from asuka.sci.solver import GUGASCISolver

mol = Molecule.from_atoms(
    [("O", np.array([0.0, 0.0, -1.16])),
     ("C", np.array([0.0, 0.0,  0.0])),
     ("O", np.array([0.0, 0.0,  1.16]))],
    unit="Angstrom", basis="6-31g", cart=True, spin=0,
)

scf_out = run_hf_df(mol, method="rhf", backend="cuda", df=True, auxbasis="autoaux")
print(f"RHF        {scf_out.e_tot:.6f}", flush=True)

t0 = time.perf_counter()
casscf_ref = run_casscf(scf_out, ncore=7, ncas=10, nelecas=8, nroots=1,
                        backend="cuda", df=True)
print(f"CASSCF108  {casscf_ref.e_tot:.6f}  {time.perf_counter()-t0:.1f}s", flush=True)

for mo_label, mo_coeff in [("HF", None), ("CASSCF", casscf_ref.mo_coeff)]:
    for ncore, ncas, nelecas in [(2, 18, 18), (0, 22, 22)]:
        sci = GUGASCISolver(
            max_ncsf=5000, grow_by=2000, selection_mode="heat_bath", backend="auto",
        )
        t0 = time.perf_counter()
        r = run_casci_df(scf_out, ncore=ncore, ncas=ncas, nelecas=nelecas,
                         mo_coeff=mo_coeff, fcisolver=sci)
        wall = time.perf_counter() - t0
        nsel = sci._sci_sel_idx.size
        print(f"CAS{ncas}{nelecas}_{mo_label}  {r.e_tot:.6f}  nsel={nsel}  {wall:.1f}s",
              flush=True)
