# cuERI ERI Quartet Implementation Map

This note maps the exact ERI quartets requested for investigation:

`psss`, `dsss`, `ppss`, `psps`, `ppps`, `ddss`, `ssdp`, `psds`, `psdp`, `ppds`, `dsds`, `dsdp`, `psdd`.

Line references below are 1-based.

## Implementation Families

All 13 quartets are part of the native Step-2 class set in `asuka/cueri/native_class_sets.py:10-29`, and the dispatcher treats them as native classes in `asuka/cueri/eri_dispatch.py:194-211`.

Handwritten `step2.cu` family:

- `psss`, `ppss`, `psps`, `dsss`, `ppps`
- Block kernels live in `asuka/cueri/cuda/ext/src/cueri_cuda_kernels_step2.cu:133`, `:222`, `:360`, `:508`, `:630`.
- Dedicated warp kernels live in `asuka/cueri/cuda/ext/src/cueri_cuda_kernels_step2.cu:984`, `:1492`, `:1342`, `:1792`, `:1179`.
- Dedicated multiblock launchers live in `asuka/cueri/cuda/ext/src/cueri_cuda_kernels_step2.cu:3969-4013`, `:4071-4115`, `:4172-4216`, `:4274-4318`, `:4379-4422`.
- Dedicated fused-Fock kernels are in `asuka/cueri/cuda/ext/src/cueri_cuda_kernels_step2.cu:3079`, `:3198`, `:3352`, `:3533`, `:3714`, with fused launchers starting at `:4527`.

Generated wave family:

- `psds`, `ppds`, `dsds`, `dsdp` are wave-1 quartets in `asuka/cueri/cuda/tools/gen_cuda_kernels.py:63-68`.
- `ssdp`, `psdp`, `psdd`, `ddss` are wave-2 quartets in `asuka/cueri/cuda/tools/gen_cuda_kernels.py:100-106`.
- The split generated translation units are compiled into the extension by `asuka/cueri/cuda/ext/CMakeLists.txt:35-46`.
- The generated family uses fixed-class kernels in:
  - `asuka/cueri/cuda/ext/src/generated/cueri_cuda_kernels_wave1_generated_part1.cu`
  - `asuka/cueri/cuda/ext/src/generated/cueri_cuda_kernels_wave2_generated_part1.cu`
  - `asuka/cueri/cuda/ext/src/generated/cueri_cuda_kernels_wave2_generated_part2.cu`

## Dispatch Path

- `resolve_kernel_class_id()` returns either the native class directly or a bra/ket-swapped native class plus `transpose=True`; see `asuka/cueri/eri_dispatch.py:102-122`.
- `run_kernel_batch_spd()` is the central runtime dispatcher. It recognizes the 13 quartets above as native classes and routes them to `eri_*_device` wrappers instead of the generic Rys kernels; see `asuka/cueri/eri_dispatch.py:304-697`.
- Handwritten wrappers are individual implementations in `asuka/cueri/gpu.py`:
  - `eri_psss_device` at `:2981`
  - `eri_ppss_device` at `:3187`
  - `eri_psps_device` at `:3394`
  - `eri_ppps_device` at `:3601`
  - `eri_dsss_device` at `:4015`
- Generated-family wrappers all funnel through `_eri_fixed_class_specialized_device()` in `asuka/cueri/gpu.py:4258-4564`, then expose per-quartet adapters:
  - `eri_ddss_device` at `:4221`
  - `eri_ssdp_device` at `:4567`
  - `eri_psds_device` at `:4604`
  - `eri_psdp_device` at `:4638`
  - `eri_psdd_device` at `:4675`
  - `eri_ppds_device` at `:4712`
  - `eri_dsds_device` at `:4820`
  - `eri_dsdp_device` at `:4854`
- Extension bindings for generated fixed-class ERI families are installed in `asuka/cueri/cuda/ext/src/cueri_cuda_ext_part1.cpp:1214-1233`.
- Extension bindings for fused-Fock entry points for all 13 quartets are installed in `asuka/cueri/cuda/ext/src/cueri_cuda_ext_part9.cpp:1354-1366`.

## Runtime Behavior

- The handwritten family has genuinely distinct block, warp, and multiblock implementations:
  - `psss` and `dsss` use subwarp-8 kernels for the warp path (`cueri_cuda_kernels_step2.cu:1074`, `:1915`) and distinct multiblock partial/reduce kernels.
  - `ppss`, `psps`, and `ppps` also have dedicated warp kernels (`:1492`, `:1342`, `:1179`) plus distinct multiblock launch paths.
- The generated family is exposed through three launcher names per quartet (`launch`, `warp_launch`, `multiblock_launch`) via `asuka/cueri/cuda/ext/src/cueri_cuda_kernels_api.h:493-597`, but the generated `warp` and `multiblock` entry points are only aliases for the plain block launcher.
  - Example: `psds` in `asuka/cueri/cuda/ext/src/generated/cueri_cuda_kernels_wave1_generated_part1.cu:6918-7001`
  - Example: `ssdp` and `psdp` in `asuka/cueri/cuda/ext/src/generated/cueri_cuda_kernels_wave2_generated_part1.cu:3345-3513`
  - Example: `ddss` in `asuka/cueri/cuda/ext/src/generated/cueri_cuda_kernels_wave2_generated_part2.cu:2866-2949`
- `_eri_fixed_class_specialized_device()` still offers `mode="block"|"warp"|"multiblock"|"auto"` for the generated family, but for these quartets the extension symbols collapse back to the same fixed block kernel, so the mode changes the symbol name, not the algorithm; see `asuka/cueri/gpu.py:4383-4564`.
- The generated-family wrappers apply tuned thread counts from Python:
  - `ddss` 96 in `asuka/cueri/gpu.py:4221-4255`
  - `ssdp` 256 in `:4567-4601`
  - `psdp` 128 in `:4638-4672`
  - `psdd` 160 in `:4675-4709`
  - `psds`, `ppds`, `dsds`, `dsdp` use the helper default based on component count in `:4295-4302`
- Generated fused-Fock kernels exist and are callable:
  - `psds`, `ppds`, `dsds`, `dsdp` in `asuka/cueri/cuda/ext/src/generated/cueri_cuda_kernels_wave1_generated_part1.cu:7857-7994`
  - `ssdp`, `psdp`, `psdd` in `asuka/cueri/cuda/ext/src/generated/cueri_cuda_kernels_wave2_generated_part1.cu:3689-3805`
  - `ddss` in `asuka/cueri/cuda/ext/src/generated/cueri_cuda_kernels_wave2_generated_part2.cu:2951-2992`
- Direct-JK does not enable those generated fused kernels by default. `asuka/hf/direct_jk.py:982-989` explicitly restricts the default fused set to `psss`, `dsss`, `ppss`, `psps`, `ppps`, and documents that the generated quartets regress because they currently behave as block-per-task kernels.

## Coverage Gaps

- The CPU cartesian reference only has explicit formulas for the handwritten family. `asuka/cueri/reference_eri_cart.py:115-124` lists support for `psss`, `ppss`, `psps`, `ppps`, and `dsss`; none of `ddss`, `ssdp`, `psds`, `psdp`, `psdd`, `ppds`, `dsds`, `dsdp` appear there.
- The explicit fused-path regression test only forces `psss` via `ASUKA_DIRECT_FOCK_FUSED_ONLY=psss`; see `tests/test_direct_scf.py:158-172`.
- `tests/test_cueri_jk_warp.py:34-39` validates contraction kernels on synthetic tiles, not the ERI quartet kernels themselves.
- The repo likely exercises some generated quartets indirectly through broader direct-SCF runs, but no quartet-targeted reference-backed tests were found for the generated family.

## Quartet Table

| quartet | family | source origin | Python wrapper | launcher behavior | fused-Fock bound | fused-Fock used by direct-JK | explicit reference/test coverage |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `psss` | handwritten | `step2.cu` | `eri_psss_device` | distinct block + subwarp8 warp + multiblock | yes | yes, default | CPU ref; explicit fused-path test |
| `dsss` | handwritten | `step2.cu` | `eri_dsss_device` | distinct block + subwarp8 warp + multiblock | yes | yes, default | CPU ref only |
| `ppss` | handwritten | `step2.cu` | `eri_ppss_device` | distinct block + subwarp8 warp + multiblock | yes | yes, default | CPU ref only |
| `psps` | handwritten | `step2.cu` | `eri_psps_device` | distinct block + warp + multiblock | yes | yes, default | CPU ref only |
| `ppps` | handwritten | `step2.cu` | `eri_ppps_device` | distinct block + warp + multiblock | yes | yes, default | CPU ref only |
| `ddss` | generated | wave2 part 2 | `eri_ddss_device` | block kernel; `warp` and `multiblock` alias block launcher | yes | no, disabled by default | no explicit quartet ref/test found |
| `ssdp` | generated | wave2 part 1 | `eri_ssdp_device` | block kernel; `warp` and `multiblock` alias block launcher | yes | no, disabled by default | no explicit quartet ref/test found |
| `psds` | generated | wave1 part 1 | `eri_psds_device` | block kernel; `warp` and `multiblock` alias block launcher | yes | no, disabled by default | no explicit quartet ref/test found |
| `psdp` | generated | wave2 part 1 | `eri_psdp_device` | block kernel; `warp` and `multiblock` alias block launcher | yes | no, disabled by default | no explicit quartet ref/test found |
| `ppds` | generated | wave1 part 1 | `eri_ppds_device` | block kernel; `warp` and `multiblock` alias block launcher | yes | no, disabled by default | no explicit quartet ref/test found |
| `dsds` | generated | wave1 part 1 | `eri_dsds_device` | block kernel; `warp` and `multiblock` alias block launcher | yes | no, disabled by default | no explicit quartet ref/test found |
| `dsdp` | generated | wave1 part 1 | `eri_dsdp_device` | block kernel; `warp` and `multiblock` alias block launcher | yes | no, disabled by default | no explicit quartet ref/test found |
| `psdd` | generated | wave2 part 1 | `eri_psdd_device` | block kernel; `warp` and `multiblock` alias block launcher | yes | no, disabled by default | no explicit quartet ref/test found |
