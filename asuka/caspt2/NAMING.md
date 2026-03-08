# CASPT2 Variable Naming Reference

This document maps the Fortran-derived variable names used in ASUKA's CASPT2 code
to their mathematical/physical meaning. These names originate from OpenMolcas and
are used throughout the codebase for consistency with the reference implementation.

**Why not rename?** These names are embedded in shared dataclass field names
(`CASPT2Fock.fimo`, `CASOrbitals.nish`, `SuperindexMap.nasup`) that are used
across both ASUKA's native code and the Molcas translation package
(`tools/molcas_caspt2_grad/translation/`). Renaming in ASUKA alone would create
inconsistency; renaming in both would violate the constraint that the translation
package stays faithful to OpenMolcas source. This reference table provides the
mapping instead.

## Fock Matrices (`CASPT2Fock` dataclass in `fock.py`)

| Code name | Full name | Definition |
|---|---|---|
| `fimo` | Fock inactive | h + J_core - 0.5*K_core (inactive Fock in MO basis) |
| `famo` | Fock active | J_act - 0.5*K_act (active Fock in MO basis) |
| `fifa` | Fock full | fimo + famo (total Fock in MO basis) |
| `epsa` | Orbital energies | Diagonal of fifa in active block (pseudo-canonical) |
| `e_core` | Core energy | Tr[h*D_core] + 0.5*Tr[V_core*D_core] + E_nuc |

## Orbital Space Dimensions (`CASOrbitals` dataclass in `superindex.py`)

| Code name | Full name | Definition |
|---|---|---|
| `nfro` | N frozen | Number of frozen core orbitals |
| `nish` | N inactive | Number of inactive (doubly occupied, non-frozen) orbitals |
| `nash` | N active | Number of active orbitals |
| `nssh` | N secondary | Number of secondary (virtual) orbitals |
| `nmo` | N MO | Total MOs = nfro + nish + nash + nssh |
| `nocc` | N occupied | nfro + nish + nash |
| `ncore` | N core | nfro + nish |

## Superindex Infrastructure (`SuperindexMap` in `superindex.py`)

| Code name | Full name | Definition |
|---|---|---|
| `smap` | Superindex map | Precomputed index mappings for all 13 IC cases |
| `nasup` | N active super | Active superindex dimension per case |
| `nisup` | N external super | External superindex dimension per case |
| `nindep` | N independent | Independent functions after linear-dep removal |
| `ktuv` | Key(t,u,v) | Triple-index lookup: (nash,nash,nash) -> superindex |
| `mtuv` | Map(t,u,v) | Reverse: superindex -> (t,u,v) triple |
| `ktu` | Key(t,u) | All-pair lookup: (nash,nash) -> superindex |
| `ktgeu` | Key(t>=u) | Symmetric pair lookup |
| `ktgtu` | Key(t>u) | Antisymmetric pair lookup |

## Density Matrices

| Code name | Full name | Definition |
|---|---|---|
| `dm1` | 1-RDM | One-electron reduced density matrix (active space) |
| `dm2` | 2-RDM | Two-electron reduced density matrix (active space) |
| `dm3` | 3-RDM | Three-electron reduced density matrix (active space) |
| `DG1`/`dg1` | DPT2 1-RDM | PT2 correction to 1-RDM |
| `DG2`/`dg2` | DPT2 2-RDM | PT2 correction to 2-RDM |
| `DF1`/`df1` | DPT2C 1-RDM | PT2 correlated density correction (1-body) |
| `DF2`/`df2` | DPT2C 2-RDM | PT2 correlated density correction (2-body) |

## MO Coefficients

| Code name | Full name | Definition |
|---|---|---|
| `CMO`/`cmo` | MO coefficients | AO-to-MO coefficient matrix (CASSCF natural orbitals) |
| `CMOPT2` | MO coefficients PT2 | Quasi-canonical MO coefficients for CASPT2 |

## IC Cases (`ICCase` enum in `superindex.py`)

| Case | Enum | Active indices | External indices |
|---|---|---|---|
| 1 | A | (t,u,v) triple | i inactive |
| 2 | Bp | t>=u sym pair | i>=j inactive sym pair |
| 3 | Bm | t>u asym pair | i>j inactive asym pair |
| 4 | C | (t,u,v) triple | a virtual |
| 5 | D | (t,u) all pairs | (a,i) mixed |
| 6 | Ep | t single | (i>=j,a) sym inactive + virtual |
| 7 | Em | t single | (i>j,a) asym inactive + virtual |
| 8 | Fp | t>=u sym pair | a>=b virtual sym pair |
| 9 | Fm | t>u asym pair | a>b virtual asym pair |
| 10 | Gp | t single | (a>=b,i) sym virtual + inactive |
| 11 | Gm | t single | (a>b,i) asym virtual + inactive |
| 12 | Hp | (none) | (a>=b,i>=j) sym virtual + sym inactive |
| 13 | Hm | (none) | (a>b,i>j) asym virtual + asym inactive |

## Gradient Lagrangians

| Code name | Full name | Definition |
|---|---|---|
| `clag` | CI Lagrangian | Derivative of PT2 energy w.r.t. CI coefficients |
| `olag` | Orbital Lagrangian | Derivative of PT2 energy w.r.t. orbital rotations |
| `wlag` | Energy-weighted density | W matrix for overlap-derivative (Pulay) terms |
| `slag` | State Lagrangian | MS/XMS state-averaging weights for gradient |
| `dpt2_1rdm` | DPT2 density | Total PT2-corrected 1-RDM for gradient |
