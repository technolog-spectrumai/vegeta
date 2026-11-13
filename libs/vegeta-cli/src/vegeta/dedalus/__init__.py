"""Dedalus — standalone parametric CAD on CadQuery.

Dedalus turns Python-defined parametric designs into geometry, measurements, STEP/STL files and Yax design
packages (``export_yax``: STL + ``yax.json`` for Piccolo, Nomios, Ketos).
It knows nothing about FEA, CFD, slicing or Vegeta.
"""
from .design import BuildError, Design, FunctionDesign, design
from .geometry import Geometry, load_step
from .loading import load_design
from .parameters import Parameter, ParameterSet
from .plotting import measurements_table, plot_parameter_study, plot_views
from .result import CommandRecord, Result, ResultError
from .yax_export import export_yax

from ._version import __version__  # noqa: F401  (the package version, read by yax_export too)

__all__ = [
    "BuildError", "CommandRecord", "Design", "FunctionDesign", "Geometry", "Parameter", "ParameterSet",
    "Result", "ResultError", "design", "export_yax", "load_design", "load_step", "measurements_table", "plot_parameter_study", "plot_views",
]
