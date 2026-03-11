# asuka.audit

CUDA diagnostics and build/runtime audit entrypoints.

## Purpose

Provides runtime checks and diagnostics used by CLI tooling to inspect CUDA
availability, extension loading, and kernel-report health.

## Public API

- `run_cuda_audit` (exported from package root)

## Workflows

Primary use is via CLI (`asuka-cuda-audit`) and internal diagnostic scripts.

## Optional Dependencies

May probe optional CUDA stacks (CuPy and native extensions) when available.

## Test Status

Covered by CLI smoke and kernel/audit regression checks.
