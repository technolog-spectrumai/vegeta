"""Velocity sets (only D3Q19 for now; a lattice is a frozen dataclass with ``c``, ``w``, ``opposite``, ``cs2``)."""
from .d3q19 import D3Q19

__all__ = ["D3Q19"]
