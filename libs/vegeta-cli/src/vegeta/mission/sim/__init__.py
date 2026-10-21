"""Simulation-only parts for the mission software: birds, the simulated detector, a kinematic aircraft, the loop."""
from .birds import SPECIES, BirdField, BirdGroup, SimTruth, Species
from .detector import SimDetector, SimDetectorConfig
from .harness import SimLoop, judge_photo, run_kinematic
from .kinematic import KinematicAircraft
