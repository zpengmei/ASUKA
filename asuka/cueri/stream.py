from __future__ import annotations

from functools import wraps
from contextlib import nullcontext
from typing import Any, ContextManager


def stream_ctx(stream: Any) -> ContextManager[Any]:
    """Return a context manager that makes `stream` the current CuPy stream.

    Parameters
    ----------
    stream
        - None: use the existing current stream (no-op context)
        - cupy.cuda.Stream / cupy.cuda.ExternalStream: used directly
        - int: treated as a raw CUDA stream pointer (cudaStream_t) and wrapped in
          cupy.cuda.ExternalStream.
    """

    if stream is None:
        return nullcontext()

    # CuPy stream objects implement the context manager protocol.
    if hasattr(stream, "__enter__") and hasattr(stream, "__exit__"):
        return stream

    import cupy as cp

    return cp.cuda.ExternalStream(int(stream))


def stream_ptr(stream: Any) -> int:
    """Return a CUDA stream pointer (cudaStream_t) as an int.

    The pointer is suitable to pass into cuERI CUDA extension entrypoints.
    """

    if stream is None:
        import cupy as cp

        return int(cp.cuda.get_current_stream().ptr)
    return int(getattr(stream, "ptr", stream))


def with_stream(fn):
    """Decorator: execute a function under `stream_ctx(kwargs['stream'])`.

    The wrapped function must accept a `stream` keyword argument.
    """

    @wraps(fn)
    def _wrapped(*args, **kwargs):
        stream = kwargs.get("stream", None)
        with stream_ctx(stream):
            return fn(*args, **kwargs)

    return _wrapped
