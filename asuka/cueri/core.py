from __future__ import annotations

"""cuERI public API surface.

This module centralizes imports/exports for the cuERI Python package under the
ASUKA namespace. External consumers should import either:
- `asuka.cueri` (re-exported from this module), or
- `asuka.cueri.core` (explicit public surface).

For a minimal/stable cuERI ↔ cuGUGA integration boundary, prefer `asuka.cueri.api`.
"""

from .api import ActiveDFResult, build_active_df
from .basis import BasisSoA
from .basis_cart import BasisCartSoA
from .basis_cart_contracted import BasisCartContractedSoA
from .cart import cart_comp_str, cart_index, cartesian_components, ncart
from .dense import build_active_eri_mat_dense_rys, build_active_eri_mat_dense_sp_only, build_active_eri_packed_dense_sp_only
from .dense_cpu import build_active_eri_mat_dense_cpu, build_active_eri_packed_dense_cpu, schwarz_shellpairs_cpu
from .active_space_dense_gpu import CuERIActiveSpaceDenseGPUBuilder
from .eri_dispatch import KernelBatch, plan_kernel_batches_spd, run_kernel_batch_spd
from .eri_utils import (
    build_pair_coeff_ordered,
    build_pair_coeff_packed,
    expand_eri_packed_to_ordered,
    j_ps_from_eri_mat,
    j_ps_from_eri_packed,
    npair,
    ordered_to_packed_index,
    pair_id,
    pair_id_matrix,
    unpack_pair_id,
)
from .gpu import warmup_cuda
from .intor_sph_gpu import CuERISphERIEngine, int2e_sph_device
from .mol_basis import SphMapForCartBasis, get_cached_or_pack_cart_ao_basis, pack_cart_shells_from_mol, pack_cart_shells_from_mol_with_sph_map
from .pair_tables_cpu import PairTablesCPU, build_pair_tables_cpu
from .reference_ssss import eri_ssss
from .screening import schwarz_shellpairs_device, schwarz_sp_device
from .shell_pairs import ShellPairs, build_shell_pairs, build_shell_pairs_l_order
from .sph import cart2sph_matrix, nsph
from .ssss import SSSSDigestResult, digest_ssss_shellpairs
from .tasks import (
    build_tasks_screened,
    build_tasks_screened_sorted_q,
    compute_task_class_id,
    decode_eri_class_id,
    eri_class_id,
    group_tasks_by_class,
    group_tasks_by_spab,
    with_task_class_id,
)
from .tile_consumer import TileConsumer
from .tile_eval import TileEvalBatch, iter_tile_batches_spd
from .tile_deriv import DerivKernelBatch, iter_deriv_kernel_batches_spd

__all__ = [
    "ActiveDFResult",
    "BasisCartSoA",
    "BasisCartContractedSoA",
    "BasisSoA",
    "KernelBatch",
    "CuERIActiveSpaceDenseGPUBuilder",
    "CuERISphERIEngine",
    "PairTablesCPU",
    "SSSSDigestResult",
    "ShellPairs",
    "SphMapForCartBasis",
    "TileConsumer",
    "TileEvalBatch",
    "DerivKernelBatch",
    "build_active_df",
    "build_active_eri_mat_dense_rys",
    "build_active_eri_mat_dense_sp_only",
    "build_active_eri_mat_dense_cpu",
    "build_active_eri_packed_dense_sp_only",
    "build_active_eri_packed_dense_cpu",
    "build_pair_coeff_ordered",
    "build_pair_coeff_packed",
    "build_pair_tables_cpu",
    "build_shell_pairs",
    "build_shell_pairs_l_order",
    "build_tasks_screened",
    "build_tasks_screened_sorted_q",
    "cart2sph_matrix",
    "cart_comp_str",
    "cart_index",
    "cartesian_components",
    "compute_task_class_id",
    "decode_eri_class_id",
    "digest_ssss_shellpairs",
    "eri_class_id",
    "eri_ssss",
    "expand_eri_packed_to_ordered",
    "group_tasks_by_class",
    "group_tasks_by_spab",
    "iter_deriv_kernel_batches_spd",
    "iter_tile_batches_spd",
    "int2e_sph_device",
    "j_ps_from_eri_mat",
    "j_ps_from_eri_packed",
    "ncart",
    "npair",
    "nsph",
    "ordered_to_packed_index",
    "pair_id",
    "pair_id_matrix",
    "pack_cart_shells_from_mol",
    "pack_cart_shells_from_mol_with_sph_map",
    "plan_kernel_batches_spd",
    "run_kernel_batch_spd",
    "schwarz_shellpairs_device",
    "schwarz_shellpairs_cpu",
    "schwarz_sp_device",
    "get_cached_or_pack_cart_ao_basis",
    "unpack_pair_id",
    "with_task_class_id",
    "warmup_cuda",
]
