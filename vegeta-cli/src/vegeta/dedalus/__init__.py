"""Dedalus — standalone parametric CAD on CadQuery.

Dedalus turns Python-defined parametric designs into geometry, measurements and STEP/STL files.
It knows nothing about FEA, CFD, slicing or Vegeta.
"""
from .design import BuildError, Design, FunctionDesign, design
from .geometry import Geometry, load_step
from .loading import load_design
from .parameters import Parameter, ParameterSet
from .plotting import measurements_table, plot_parameter_study, plot_views
from .result import CommandRecord, Result, ResultError

__version__ = "0.1.0"

__all__ = [
    "BuildError", "CommandRecord", "Design", "FunctionDesign", "Geometry", "Parameter", "ParameterSet",
    "Result", "ResultError", "design", "load_design", "load_step", "measurements_table", "plot_parameter_study", "plot_views",
]
