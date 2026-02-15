# Aggregator for EPQ+QMC compiled kernels (include-only).
# Split to keep diffs/debugging manageable; this is still one Cython module.

include "_epq_cy_core_common.pxi"
include "_epq_cy_core_df_expand_cderi.pxi"
include "_epq_cy_core_tables.pxi"
include "_epq_cy_core_contribs.pxi"

include "_epq_cy_qmc_sampling_api.pxi"
include "_epq_cy_qmc_spawn_api.pxi"

include "_epq_cy_core_csc.pxi"
include "_epq_cy_core_apply_weighted_many.pxi"
include "_epq_cy_core_apply_g.pxi"
include "_epq_cy_core_apply_g_accum.pxi"
