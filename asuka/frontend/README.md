# asuka.frontend

Workflow building blocks for ASUKA scripts/notebooks: molecule containers,
basis/DF setup, SCF wrappers, and geometry/vibrational analysis adapters.

## Public Package Exports

`asuka.frontend` re-exports:

- Core objects: `Molecule`, `MethodWorkflow`
- Workflow/analysis helpers:
  `geomopt_molecule`, `fd_hessian_molecule`, `frequency_analysis_molecule`,
  `make_df_casscf_energy_grad`, `make_df_casci_energy_grad`
- Basis/DF/1e helpers:
  `build_ao_basis_cart`, `build_df_B_cueri`, `build_df_bases_cart`,
  `build_int1e_cart_from_mol`, `build_int1e_cart_deriv_from_mol`
- SCF result types: `RHFDFRunResult`, `UHFDFRunResult`, `ROHFDFRunResult`
- SCF entry points:
  `run_hf`, `run_hf_df`, `run_rhf`, `run_rhf_dense`, `run_rhf_df`, `run_rhf_df_cpu`,
  `run_uhf`, `run_uhf_dense`, `run_uhf_df`, `run_uhf_df_cpu`,
  `run_rohf`, `run_rohf_dense`, `run_rohf_df`, `run_rohf_df_cpu`

## Quick Usage

### Run DF-RHF from a lightweight molecule object

```python
from asuka.frontend import Molecule, run_rhf_df

mol = Molecule.from_atoms(
    "N 0.0 0.0 0.0; N 0.0 0.0 1.10",
    unit="Angstrom",
    basis="cc-pvdz",
    spin=0,
)
out = run_rhf_df(mol, auxbasis="autoaux")
print(out.scf.e_tot)
```

### Build a reusable DF-CASSCF workflow callback

```python
from asuka.frontend import MethodWorkflow

wf = MethodWorkflow.df_casscf_method(
    mol,
    ncore=0,
    ncas=8,
    nelecas=(4, 4),
    backend="cuda",
)
e, g = wf(mol.coords_bohr)
print(e, g.shape)
```

## Module Map

| Module | Purpose |
| --- | --- |
| `__init__.py` | Public API re-export hub |
| `molecule.py` | Lightweight `Molecule` container and geometry/electron helpers |
| `one_electron.py` | AO basis packing and `(S,T,V)` / derivative integrals from `Molecule` |
| `df.py` | DF basis builders and cuERI `B[mu,nu,Q]` construction wrappers |
| `scf.py` | Unified RHF/UHF/ROHF SCF entry points (`df=True` DF and `df=False` dense AO-ERI) |
| `analysis.py` | Method callback builders, geomopt/FD Hessian/frequency workflow helpers |
| `basis_bse.py` / `basis_packer.py` | Basis-set loading and cartesian packed basis conversion |
| `periodic_table.py` | Element metadata (Z, symbols, masses) used by frontend helpers |

## Notes

- DF and 1e integral paths require `mol.cart=True` (cartesian basis layout).
- `run_hf(..., df=True)` / `run_*_df(...)` builds DF factors with cuERI
  (GPU or CPU DF builder paths).
- `run_hf(..., df=False)` / `run_*_dense(...)` builds full AO dense ERIs and
  stores them in `scf_out.ao_eri` (`scf_out.df_B` is `None`).
- The module is structured to avoid importing PySCF directly at package import
  time; method-level integrations are loaded lazily.
