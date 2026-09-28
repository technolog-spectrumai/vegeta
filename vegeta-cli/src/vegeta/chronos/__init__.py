"""Chronos — missions, cyclic loads and long-term life.

Missions are sequences of segments with explicit load levels and vibratory excitations; a
``Structure`` (natural frequencies + damping) turns excitations into amplified cyclic loads;
``build_spectrum`` rainflow-counts the mission into blocks; ``simulate_life`` accumulates Miner
damage over a fleet usage. The spectrum JSON is the hand-off to ``talos.assess_fatigue``.
"""
from .dynamics import Structure, half_sine, shock_at_mount, srs
from .life import LifeSimulation, SNCurve, hotspot_damage, simulate_life
from .mission import Excitation, Mission, Segment
from .parachute import Body, DropResult, Parachute, Wind, crush_pulse, landing_scatter, simulate_drop
from .rainflow import rainflow, reversals
from .result import CommandRecord, Result, ResultError
from .spectrum import Block, LoadSpectrum, build_spectrum
from .turbulence import (GustField, Turbulence, alleviation_factor, frozen_field, gust_series, load_factor, rice_extreme,
                         through_mount, welch)

__version__ = "0.1.0"
__all__ = ["Block", "Body", "DropResult", "Parachute", "Wind", "crush_pulse", "half_sine", "landing_scatter", "shock_at_mount", "simulate_drop", "srs", "CommandRecord", "Excitation", "LifeSimulation", "LoadSpectrum", "Mission", "Result", "ResultError",
           "SNCurve", "Segment", "Structure", "build_spectrum", "hotspot_damage", "rainflow", "reversals", "simulate_life",
           "GustField", "Turbulence", "alleviation_factor", "frozen_field", "gust_series", "load_factor", "rice_extreme", "through_mount", "welch"]
