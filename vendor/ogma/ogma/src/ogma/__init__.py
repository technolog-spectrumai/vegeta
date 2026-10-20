"""Ogma: the physics of standard machine parts — what they are (data) and do (behaviour).

    import ogma as og
    og.frames.quat_to_R(q)

numpy only, SI units, organised by physics, never by machine (README.md). Submodules load on first use.
"""
import importlib

__version__ = "0.1.0"
_MODULES = ("units", "frames", "mass", "gears", "soil", "tracks", "cutting", "motors", "batteries", "atmosphere", "wind",
            "propulsion", "rotors", "aero")


def __getattr__(name):
    if name in _MODULES:
        try:
            return importlib.import_module(f".{name}", __name__)
        except ModuleNotFoundError as e:
            if e.name == f"{__name__}.{name}":
                raise AttributeError(f"ogma.{name} is planned but not written yet (README.md)") from None
            raise
    raise AttributeError(f"module 'ogma' has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_MODULES))
