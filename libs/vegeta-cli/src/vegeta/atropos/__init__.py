"""Atropos — flat parts for a GRBL diode laser: ribs from a section, the kerf, nesting on stock sheets, SVG and
G-code, time and material.

Atropos draws what the engineer specifies — the sheet, the machine, every operation's power, feed, passes and
kerf — and keeps every file it writes. It does not pick laser settings for a material: cut a test strip first.
"""
from .airfoil import arc_length, camber_z, naca4, section_area, section_outline, thickness_at
from .gcode import GcodeJob, estimate_time, read_gcode, write_gcode
from .geometry import Part, Path, circle, clip_halfplane, clip_x, parts_from_json, parts_to_json, polyline, rect
from .job import cut_job
from .nest import Nesting, Placement, nest
from .plotting import plot_nesting, plot_parts, plot_sheet
from .result import CommandRecord, Result, ResultError
from .rib import Lightening, Notch, RibSpec, Sheeting, TubeHole, make_rib
from .settings import CutSettings, Machine, Operation, Sheet
from .svg import read_svg_paths, write_svg

__version__ = "0.1.0"

__all__ = [
    "CommandRecord", "CutSettings", "GcodeJob", "Lightening", "Machine", "Nesting", "Notch", "Operation", "Part", "Path",
    "Placement", "Result", "ResultError", "RibSpec", "Sheet", "Sheeting", "TubeHole",
    "arc_length", "camber_z", "circle", "clip_halfplane", "clip_x", "cut_job", "estimate_time", "make_rib", "naca4",
    "nest", "parts_from_json", "parts_to_json", "plot_nesting", "plot_parts", "plot_sheet", "polyline", "read_gcode", "read_svg_paths",
    "rect", "section_area", "section_outline", "thickness_at", "write_gcode", "write_svg",
]
