"""Zephyr — 2-D wing sections: NACA geometry (4-digit, 5-digit plain and reflexed), coordinates from a file, plain
flaps, thin-airfoil theory, and polars from NeuralFoil (a neural surrogate of XFOIL, optional) or XFOIL (when on the
PATH), with the numbers a wing design reads from them.

Zephyr computes what the engineer names — the section, the Reynolds number, the transition criterion, the flap — and
labels every polar with its source and confidence. A surrogate's polar is a first estimate; the section an aircraft
is built on deserves a tunnel or a 2-D CFD run.
"""
from .export import load_polars, save_polars
from .polar import Polar, analyze, analyze_result, backends, check_against_thin_airfoil, flap_sweep
from .result import CommandRecord, Result, ResultError
from .section import (NACA5_PLAIN, NACA5_REFLEX, Section, camber_from_points, flap_effectiveness, from_coordinates, from_dat, naca,
                      thickness_distribution, thin_airfoil)

__version__ = "0.1.0"

__all__ = [
    "CommandRecord", "NACA5_PLAIN", "NACA5_REFLEX", "Polar", "Result", "ResultError", "Section",
    "analyze", "analyze_result", "backends", "camber_from_points", "check_against_thin_airfoil", "flap_effectiveness", "flap_sweep",
    "from_coordinates", "from_dat", "load_polars", "naca", "save_polars", "thickness_distribution", "thin_airfoil",
]
