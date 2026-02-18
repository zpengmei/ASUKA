# ASUKA

<p align="center">
  <img src="resources/asuka.jpg" alt="ASUKA" width="100%">
</p>

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

CUDA-accelerated multireference methods built on a native GUGA/DRT engine.

Status: experimental — APIs and performance characteristics are evolving.

## Install

Requires Python >= 3.10, a C compiler, and the CUDA toolkit (`nvcc` on PATH).

### CUDA toolkit

If you don't already have `nvcc` available, the easiest route is via conda:

```bash
conda install -c nvidia cuda-toolkit
```

Alternatively, install from the [NVIDIA CUDA downloads page](https://developer.nvidia.com/cuda-downloads) and make sure `nvcc` is on your `PATH`:

```bash
export PATH=/usr/local/cuda/bin:$PATH
nvcc --version   # should print the CUDA compiler version
```

### ASUKA

```bash
conda create -n asuka python=3.10 -y && conda activate asuka
conda install -c nvidia cuda-toolkit   # if nvcc is not already available

# CUDA 13.x (default)
pip install -e ".[cuda]"

# CUDA 12.x
# pip install -e ".[cuda12]"
```

This single command installs all dependencies, compiles Cython extensions, and builds the CUDA kernels. If `nvcc` is not found, CUDA extensions are skipped and only CPU paths are available.

CPU-only install (skip CUDA builds explicitly):

```bash
ASUKA_SKIP_CUDA_EXT=1 pip install -e .
```

Verify:

```bash
python -c "import asuka; print(asuka.__version__)"
```

## Quick start

```python
from asuka.frontend import Molecule, run_hf
from asuka.mcscf import run_casscf

mol = Molecule.from_atoms(
    "N 0 0 0; N 0 0 1.0977",
    unit="Angstrom",
    basis="cc-pvtz",
    cart=True,
    spin=0,
)

scf_out = run_hf(mol, method="rhf", backend="cuda", df=True)
out = run_casscf(
    scf_out,
    ncore=2, ncas=10, nelecas=10, nroots=1,
    backend="cuda", df=True,
)
print(f"CASSCF energy: {out.e_tot:.10f}  ({out.niter} iterations)")
```

No tuning needed — defaults use the augmented Hessian orbital optimizer with TF32-accelerated Krylov solver and cuBLAS-based sigma vectors.

## State-averaged CASSCF

```python
out = run_casscf(
    scf_out,
    ncore=0, ncas=2, nelecas=2, nroots=2,
    root_weights=(0.5, 0.5),
    backend="cuda", df=True,
)
print("SA-CASSCF e_roots =", out.e_roots)
```

## Warm-starting across geometries

Pass the previous result as `guess=` to reuse MO coefficients and CI vectors — useful for PES scans, geometry optimization, and finite-difference Hessians.

```python
hf_kw = dict(method="rhf", backend="cuda", df=True)
cas_kw = dict(ncore=0, ncas=2, nelecas=2, nroots=2,
              root_weights=(0.5, 0.5), backend="cuda", df=True)

for r_bohr in (1.40, 1.45, 1.50):
    coords = mol.coords_bohr.copy()
    coords[1, 2] = float(r_bohr)
    mol.set_coords_bohr_inplace(coords)

    scf_out = run_hf(mol, guess=scf_out, **hf_kw)
    out = run_casscf(scf_out, guess=out, **cas_kw)
    print(f"r = {r_bohr:.2f}  e_avg = {out.e_tot:.10f}")
```

## Geometry optimization + harmonic frequencies

```python
from asuka.frontend import MethodWorkflow
from asuka.geomopt import GeomOptSettings

wf = MethodWorkflow.from_casscf_singlepoint(scf_out, out, warm_start=True)
opt = wf.geomopt(settings=GeomOptSettings(verbose=1))
nm = wf.frequencies_fd(step_bohr=1e-3, linear=True)
print("freq_cm1:", nm.freq_cm1)
```

## Build notes

Environment variables for the CUDA build:

| Variable | Purpose |
|---|---|
| `GUGA_CUDA_ARCH` / `CUERI_CUDA_ARCH` | Override compute capability (e.g. `89`) |
| `GUGA_CUDA_NVCC` / `CUERI_CUDA_NVCC` | Override nvcc path |
| `GUGA_CUDA_ROOT` / `CUDA_HOME` | Non-standard CUDA toolkit root |
| `ASUKA_SKIP_CUDA_EXT=1` | Skip all CUDA extension builds |
| `ASUKA_REQUIRE_CUDA_EXT=1` | Fail if nvcc is not found |

If CUDA libraries were installed via pip wheels (`nvidia-*`), you may need to expose them to the dynamic linker:

```bash
PREFIX="$(python -c 'import sys; print(sys.prefix)')"
PYVER="$(python -c 'import sys; print(f"python{sys.version_info[0]}.{sys.version_info[1]}")')"
for d in "$PREFIX/lib/$PYVER/site-packages/nvidia"/cu{13,12}/lib; do
  [ -d "$d" ] && for f in "$d"/lib*.so.*; do ln -sf "$f" "$PREFIX/lib/$(basename "$f")"; done
done
```

## Repo layout

- `asuka/cuguga/` — core GUGA/DRT engine (CSF indexing, row-oracle, EPQ)
- `asuka/cuda/` — CUDA extension sources and backends
- `asuka/cueri/` — GPU ERI/DF integral kernels
- `asuka/mcscf/` — CASCI, CASSCF, Newton-CASSCF, nuclear gradients
- `asuka/mrci/` — internally-contracted MRCI
- `asuka/mrpt2/` — NEVPT2 and CASPT2 (WIP)
- `asuka/frontend/` — molecule, SCF, and workflow drivers
- `asuka/qmc/` — GUGA-FCIQMC/FCI-FRI (WIP)

## Citation

If you use ASUKA in your research, please cite the following paper:

```bibtex
@misc{pengmei2026cugugaoperatordirectgraphicalunitary,
      title={cuGUGA: Operator-Direct Graphical Unitary Group Approach Accelerated with CUDA},
      author={Zihan Pengmei},
      year={2026},
      eprint={2601.17729},
      archivePrefix={arXiv},
      primaryClass={physics.chem-ph},
      url={https://arxiv.org/abs/2601.17729},
}
```

If you wish to cite the software itself directly, please use the Zenodo record:

```bibtex
@software{pengmei2026asuka,
      author={Zihan Pengmei},
      title={ASUKA},
      year={2026},
      publisher={Zenodo},
      doi={10.5281/zenodo.XXXXXXX},
      url={https://doi.org/10.5281/zenodo.XXXXXXX},
}
```

See also `CITATION.cff` for additional citation formats.

## Acknowledgments

ASUKA builds on ideas, algorithms, and validation references from several open-source quantum chemistry packages. We gratefully acknowledge:

- [PySCF](https://github.com/pyscf/pyscf) — SCF and CASSCF reference implementations, integral conventions, and the augmented Hessian Newton-CASSCF formulation
- [OpenMolcas](https://gitlab.com/Molcas/OpenMolcas) — GUGA operator formalism, CASPT2 reference data, and spin-orbit coupling methodology
- [COLUMBUS](https://www.univie.ac.at/columbus/) — GUGA-CI theory and the graphical unitary group approach that underpins this work

## License

ASUKA is licensed under the [ASUKA Noncommercial License 1.0](LICENSE). Free for academic and non-commercial research use. Commercial use or redistribution as a standalone package requires explicit written permission from the author. See the LICENSE file for full details.
