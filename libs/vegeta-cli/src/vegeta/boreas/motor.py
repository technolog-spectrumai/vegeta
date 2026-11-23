"""Brushless DC motor as the usual first-order model (Kv, winding resistance, no-load current) and a battery by its
nominal numbers: Ogma's parts (``og.motors.DCMotor``, ``og.batteries.NominalPack``), under Boreas's names."""
from ogma.batteries import NominalPack as Battery
from ogma.motors import DCMotor as Motor

__all__ = ["Battery", "Motor"]
