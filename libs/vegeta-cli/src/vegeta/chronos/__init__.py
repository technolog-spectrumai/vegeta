"""Chronos — missions, cyclic loads and long-term life.

Missions are sequences of segments with explicit load levels and vibratory excitations; a
``Structure`` (natural frequencies + damping) turns excitations into amplified cyclic loads;
``build_spectrum`` rainflow-counts the mission into blocks; ``simulate_life`` accumulates Miner
damage over a fleet usage. The spectrum JSON is the hand-off to ``talos.assess_fatigue``.
"""
from .dynamics import Structure
from .life import LifeSimulation, SNCurve, hotspot_damage, simulate_life
from .mission import Excitation, Mission, Segment
from .rainflow import rainflow, reversals
from .result import CommandRecord, Result, ResultError
from .spectrum import Block, LoadSpectrum, build_spectrum

__version__ = "0.1.0"
__all__ = ["Block", "CommandRecord", "Excitation", "LifeSimulation", "LoadSpectrum", "Mission", "Result", "ResultError",
           "SNCurve", "Segment", "Structure", "build_spectrum", "hotspot_damage", "rainflow", "reversals", "simulate_life"]
