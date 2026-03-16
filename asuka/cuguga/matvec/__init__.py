"""GUGA matvec: controller + executor split.

The controller (GugaMatvecController) owns the backend-neutral tiling loop.
The executor (CudaGugaExecutor) owns all CUDA-specific kernel calls and buffers.

To add a new backend (e.g. ROCm), implement GugaExecutorProtocol and plug it
into GugaMatvecController — the tiling loop works unchanged.
"""

from asuka.cuguga.matvec.protocol import GugaExecutorProtocol
from asuka.cuguga.matvec.controller import GugaMatvecController

__all__ = [
    "GugaExecutorProtocol",
    "GugaMatvecController",
]
