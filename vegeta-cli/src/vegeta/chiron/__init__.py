"""Chiron — Vegeta's legged-robot simulation tool, a wrapper on MuJoCo (Chiron, the centaur who trained heroes).

Build a robot from plain dataclasses (``Link``, ``Joint``, ``Geom``, ``PointMass``, ``Servo``, ``FootSpec``,
``Robot``) or wrap a third-party MJCF (``Robot.from_mjcf`` + ``RobotMeta``); put it on a ``Terrain``
(``Flat``, ``LongitudinalBumps``, ``AlternatingBumps``, ``CrossSlope``, ``Steps``, ``Rough``, ``Custom``); run it
in **ChironLab**, the reusable environment: ``reset``/``step``/``observe`` for interactive use, ``run`` for a
whole trial under ``FailureRules`` with ``Disturbance`` s, returning an ``Episode`` (a log dict in the shared
episode-log format, an outcome, ``save``/``load``, ``to_result``). ``PhaseGenerator`` provides fixed or
load-adaptive (Tegotae) leg phases for controllers.

MuJoCo is imported only when a lab is built or a model compiled. Study-level analysis lives in submodules
imported explicitly: ``from vegeta.chiron import metrics, stats, viz``. Units: SI (m, kg, s, N, N·m, rad).
"""
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
