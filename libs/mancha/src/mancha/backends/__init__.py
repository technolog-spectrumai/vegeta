"""Backends: ``numpy`` (the reference) and ``triton`` (GPU; imported only when asked for)."""
from __future__ import annotations

from typing import Any

from .base import Backend
from .numpy_backend import NumpyBackend

BACKENDS = ("numpy", "triton")


def create_backend(name: str, **kwargs: Any) -> Backend:
    if name == "numpy":
        return NumpyBackend(**kwargs)
    if name == "triton":
        from .triton_backend import TritonBackend   # torch and triton are optional dependencies

        return TritonBackend(**kwargs)
    raise ValueError(f"unknown backend {name!r}; choose from {BACKENDS}")


__all__ = ["BACKENDS", "Backend", "NumpyBackend", "create_backend"]
