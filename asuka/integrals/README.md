# asuka.integrals

Integral helpers and DF plumbing used by ASUKA (1e AO integrals, cuERI-backed
DF builders, and DF/MO utility containers).

## Public Package Exports

`asuka.integrals` re-exports:

- `CuERIDFConfig`
- `build_df_B_from_cueri_packed_bases`
- `Int1eResult`
- `Int1eDerivResult`
- `build_S_cart`, `build_T_cart`, `build_V_cart`
- `build_dS_cart`, `build_dT_cart`, `build_dV_cart`
- `build_int1e_cart`, `build_int1e_cart_deriv`
- `nao_cart_from_basis`, `shell_to_atom_map`

More advanced APIs (DF contexts, DF gradients, adjoints) live in submodules.

## Quick Usage

### Build 1e AO integrals

```python
from asuka.integrals import build_int1e_cart

int1e = build_int1e_cart(
    ao_basis,  # packed BasisCartSoA
    atom_coords_bohr=atom_coords_bohr,  # shape (natm, 3)
    atom_charges=atom_charges,          # shape (natm,)
)
hcore = int1e.hcore
```

### Build AO DF factors via cuERI

```python
from asuka.integrals import CuERIDFConfig, build_df_B_from_cueri_packed_bases

cfg = CuERIDFConfig(backend="gpu_rys", threads=256)
B = build_df_B_from_cueri_packed_bases(ao_basis, aux_basis, config=cfg)
# B has shape (nao, nao, naux)
```

### Build MO DF integrals (submodule API)

```python
from asuka.integrals.df_integrals import build_df_mo_integrals

df_mo = build_df_mo_integrals(mol, mo_coeff, auxbasis="autoaux")
# df_mo.l_full: (norb*norb, naux)
```

## Module Map

| Module | Purpose |
| --- | --- |
| `__init__.py` | Stable package-level integral APIs |
| `int1e_cart.py` | AO 1e integrals (S/T/V), derivatives, and contraction helpers |
| `cueri_df.py` | cuERI-backed AO DF builder (`B[mu,nu,Q]`) |
| `cueri_df_cpu.py` | CPU fallback/path for cuERI DF AO factors |
| `df_integrals.py` | `DFMOIntegrals`/`DeviceDFMOIntegrals` containers and builders |
| `df_context.py` | Cached AO DF/Cholesky context + repeated AO->MO transforms |
| `df_grad_context.py` / `grad.py` | DF gradient contraction contexts and analytic/FD helpers |
| `df_adjoint.py` | Adjoint utilities for DF whitening/Cholesky pieces |
| `eri4c_deriv_contracted.py` | Contracted 4c derivative task evaluators (CPU/CUDA contexts) |
| `oracle_df.py` | DF row-oracle helper for cuguga-style workflows |

## Notes

- `build_int1e_cart` and `build_int1e_cart_deriv` take packed basis plus
  explicit `atom_coords_bohr` / `atom_charges` arrays.
- 1e backend selection is controlled by `ASUKA_INT1E_BACKEND`:
  `auto|python|numba|cython`.
- cuERI/GPU-backed DF paths require CuPy and cuERI CUDA extension support.

