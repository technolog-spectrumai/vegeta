"""Small helpers for workflow command lines."""
from __future__ import annotations


def parse_assignments(items) -> dict:
    """``["arm_width=16", "taper=0.55", "name=x"]`` -> ``{"arm_width": 16.0, "taper": 0.55, "name": "x"}``."""
    out = {}
    for item in items or ():
        if "=" not in item:
            raise ValueError(f"{item!r}: expected NAME=VALUE")
        k, v = item.split("=", 1)
        try:
            out[k.strip()] = float(v) if any(c in v for c in ".eE") else int(v)
        except ValueError:
            out[k.strip()] = v
    return out
