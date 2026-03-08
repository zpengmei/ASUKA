"""SS-CASPT2 analytic nuclear gradient (CPU, DF).

Re-export shim: all implementations have been split into focused modules.
This file preserves backward compatibility for existing imports.
"""

from asuka.caspt2.gradient.driver_ss import (  # noqa: F401
    caspt2_ss_gradient_native,
    _compute_clag_fd_with_context,
)
from asuka.caspt2.gradient.zvector import _solve_zvector  # noqa: F401
from asuka.caspt2.gradient.assembly import _assemble_gradient  # noqa: F401
from asuka.caspt2.pt2lag import (  # noqa: F401, E402
    _build_case_amps_from_asuka,
    _build_lagrangians,
    _superindex_c_to_f_perm,
)
from asuka.caspt2.gradient.debug_utils import (  # noqa: F401
    _asnumpy_f64,
    _apply_ci_basis_map,
    _apply_debug_zorb_block_signs,
    _align_df_like_factors_to_reference,
    _best_signed_perm_vec,
    _build_dlao_candidate_ao,
    _build_exact_df_like_factors_from_ao_eri,
    _infer_ci_basis_map_from_dump,
    _infer_ci_basis_map_from_resp,
    _parse_ci_basis_map_from_env,
    _parse_csv_floats,
    _parse_csv_ints,
    _read_molcas_dump_matrix,
    _resolve_response_dpt2_mode,
)
