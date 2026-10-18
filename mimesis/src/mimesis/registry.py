"""Mimics by name: ``@mimic("fenicsx.plate")`` registers a stub; ``call(name, ..., real=fn, mode=...)`` picks."""
from __future__ import annotations

from typing import Callable

from .label import label

_MIMICS: dict[str, Callable] = {}


class MimicError(LookupError):
    pass


def mimic(name: str, *, labelled: bool = True):
    """Register the decorated function as the stub of the real API ``name``. Its dict result is labelled (``label``)."""
    def deco(fn: Callable) -> Callable:
        def stub(*a, **k):
            out = fn(*a, **k)
            return label(out, name) if labelled and isinstance(out, dict) else out
        stub.__name__, stub.__doc__, stub.__wrapped__ = fn.__name__, fn.__doc__, fn
        stub.mimic_name = name
        _MIMICS[name] = stub
        return stub
    return deco


def get(name: str) -> Callable:
    try:
        return _MIMICS[name]
    except KeyError:
        raise MimicError(f"no mimic named {name!r} (registered: {', '.join(sorted(_MIMICS)) or 'none'})") from None


def available() -> list[str]:
    return sorted(_MIMICS)


def call(name: str, *args, real: Callable | None = None, mock: bool, **kwargs):
    """The stub of ``name`` when ``mock``, else ``real(*args, **kwargs)`` (which must then be given)."""
    if mock:
        return get(name)(*args, **kwargs)
    if real is None:
        raise MimicError(f"{name}: not in mock mode and no real implementation given")
    return real(*args, **kwargs)
