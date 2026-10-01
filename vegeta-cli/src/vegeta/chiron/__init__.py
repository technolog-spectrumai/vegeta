"""Chiron — Vegeta's legged-robot dynamics tool, a wrapper on MuJoCo (Chiron, the centaur who trained the heroes).

Build a robot from plain dataclasses (``Link``, ``Joint``, ``Geom``, ``PointMass``, ``Servo``, ``FootSpec``,
``Robot``) or wrap a third-party MJCF (``Robot.from_mjcf`` + ``RobotMeta``); put it on a ``Terrain``
(``Flat``, ``LongitudinalBumps``, ``AlternatingBumps``, ``CrossSlope``, ``Steps``, ``Rough``, ``Custom``); run it
in **ChironLab**, the reusable environment: ``reset``/``step``/``observe`` for interactive use, ``run`` for a
whole trial under ``FailureRules`` with ``Disturbance`` s, returning an ``Episode`` (a log dict in the shared
episode-log format, an outcome, ``save``/``load``, ``to_result``). ``PhaseGenerator`` provides fixed or
load-adaptive (Tegotae) leg phases for controllers.

Submodules, loaded on first use (``ch.metrics`` or ``from vegeta.chiron import metrics``):

* ``metrics`` — per-trial locomotion metrics on an episode log (``trial_metrics``);
* ``stats`` — paired-trial statistics (Wilson, McNemar, bootstrap, logistic fits);
* ``experiments`` — ``Trial``, ``run_trials`` (cached process pool), ``rerun_with_log``, paired-design helpers;
* ``viz`` — off-screen rendering with pyvista; ``cli`` — the ``chiron`` command.

Importing the package loads neither MuJoCo nor pyvista nor pandas. Units: SI (m, kg, s, N, N·m, rad).
"""
import importlib as _importlib

from .gaits import PhaseGenerator, cycle_to_oscillator, in_stance, oscillator_to_cycle
from .lab import ChironLab, Command, Disturbance, Episode, FailureRules, Observation
from .result import CommandRecord, Result, ResultError
from .robot import FootSpec, Geom, Joint, Link, PointMass, Robot, RobotMeta, Servo, SimOptions
from .servo import saturation, servo_torque, torque_limit
from .terrain import (AlternatingBumps, CrossSlope, Custom, Flat, LongitudinalBumps, Rough, Steps, Terrain,
                      terrain_from_spec)

__version__ = "0.1.0"
__all__ = [
    "AlternatingBumps", "ChironLab", "Command", "CommandRecord", "CrossSlope", "Custom", "Disturbance", "Episode",
    "FailureRules", "Flat", "FootSpec", "Geom", "Joint", "Link", "LongitudinalBumps", "Observation", "PhaseGenerator",
    "PointMass", "Result", "ResultError", "Robot", "RobotMeta", "Rough", "Servo", "SimOptions", "Steps", "Terrain",
    "cycle_to_oscillator", "in_stance", "oscillator_to_cycle", "saturation", "servo_torque", "terrain_from_spec",
    "torque_limit",
]

_SUBMODULES = ("metrics", "stats", "experiments", "export", "viz", "cli")


def __getattr__(name):
    """Load the submodules lazily (``ch.metrics.trial_metrics`` works without an explicit import)."""
    if name in _SUBMODULES:
        return _importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_SUBMODULES))
