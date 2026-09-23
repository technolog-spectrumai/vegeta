"""Talos — standalone linear static structural analysis (Gmsh + CalculiX).

Talos consumes any STEP file plus explicit engineering configuration. It never guesses supports,
loads, material properties or strengths.
"""
from .frd import FieldResults, read_dat_reactions, read_frd, von_mises
from .loads import Acceleration, Displacement, FixedSupport, Force, Pressure
from .materials import Material
from .mesh import MeshSettings
from .model import StructuralModel
from .plotting import plot_deformed, plot_along_axis, plot_von_mises_histogram
from .regions import Surfaces, SurfacesInBox, SurfacesOnPlane
from .result import CommandRecord, Result, ResultError
from .stepinfo import StepInfo, inspect_step
from .units import UNIT_SYSTEMS

__version__ = "0.1.0"

__all__ = [
    "Acceleration", "CommandRecord", "Displacement", "FieldResults", "FixedSupport", "Force", "Material",
    "MeshSettings", "Pressure", "Result", "ResultError", "StepInfo", "StructuralModel", "Surfaces",
    "SurfacesInBox", "SurfacesOnPlane", "UNIT_SYSTEMS", "inspect_step", "plot_along_axis", "plot_deformed",
    "plot_von_mises_histogram", "read_dat_reactions", "read_frd", "von_mises",
]
