"""Boreas — propeller and rotor performance (blade element momentum theory) with a motor/battery model.

Pure numpy; every coefficient is an explicit input. Outputs are Python objects and a JSON export
(``boreas.export``) that other tools and notebooks read.
"""
from .airfoil import Airfoil
from .bemt import OperatingPoint, rpm_for_thrust, solve
from .export import export, load, performance_map
from .motor import Battery, Motor
from .propeller import Propeller, inches
from .result import CommandRecord, Result, ResultError
from .system import Propulsion, SystemPoint, excitations, unbalance_force

__version__ = "0.1.0"
__all__ = ["Airfoil", "Battery", "CommandRecord", "Motor", "OperatingPoint", "Propeller", "Propulsion", "Result",
           "ResultError", "SystemPoint", "excitations", "export", "inches", "load", "performance_map",
           "rpm_for_thrust", "solve", "unbalance_force"]
