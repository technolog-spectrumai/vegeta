"""Aeromant — standalone external aerodynamics on OpenFOAM.

Aeromant copies a known-good template case, inserts an STL and explicit values, runs the
template's pipeline when asked and extracts force coefficients. It never invents a CFD case.
"""
from .case import CFDCase, open_case, read_case_results
from .environment import OpenFOAMEnvironment
from .plotting import plot_coefficients, plot_residuals
from .result import CommandRecord, Result, ResultError
from .results import read_checkmesh, read_coefficients, read_solver_log
from .stl import Surface, read_stl, write_stl_ascii
from .templates import TEMPLATES, Flavor, TemplateParameter, TemplateSpec, get_template, list_templates

__version__ = "0.1.0"

__all__ = [
    "CFDCase", "CommandRecord", "Flavor", "OpenFOAMEnvironment", "Result", "ResultError", "Surface", "TEMPLATES",
    "TemplateParameter", "TemplateSpec", "get_template", "list_templates", "open_case", "plot_coefficients",
    "plot_residuals", "read_case_results", "read_checkmesh", "read_coefficients", "read_solver_log", "read_stl",
    "write_stl_ascii",
]
