# asuka.utils

Small shared utility helpers.

## Purpose

Holds compact utility code reused by multiple runtime paths while keeping
package-level dependencies minimal.

## Public API

No broad stable package-root API is currently defined; import specific helpers
from utility submodules.

## Workflows

Used by runtime modules for local caching/utility behavior.

## Optional Dependencies

May use optional stacks (for example CuPy) in specific utility modules.

## Test Status

Covered indirectly through package-level regression suites.
