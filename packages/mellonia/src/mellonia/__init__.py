"""Mellonia — standalone 3D-print manufacturability analysis with the PrusaSlicer CLI.

Mellonia slices an STL with explicit printer, filament and print settings and an orientation chosen
by the engineer. It never optimises orientation and preserves the G-code it produces.
"""
from .gcode import GcodeInfo, parse_duration, read_gcode
from .plotting import plot_layers
from .result import CommandRecord, Result, ResultError
from .settings import REQUIRED_KEYS, Orientation, PrintSettings, read_ini
from .slicer import slice_stl

__version__ = "0.1.0"

__all__ = [
    "CommandRecord", "GcodeInfo", "Orientation", "PrintSettings", "REQUIRED_KEYS", "Result", "ResultError",
    "parse_duration", "plot_layers", "read_gcode", "read_ini", "slice_stl",
]
