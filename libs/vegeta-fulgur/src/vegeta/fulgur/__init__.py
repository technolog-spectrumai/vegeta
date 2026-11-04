"""Fulgur — Vegeta's cold-plasma toolkit. First physics: the corona discharge in air and its thrust.

    gas         air: density, ion mobility, Peek onset, breakdown field, net ionisation (T, p, humidity)
    climate     the design conditions: Warsaw's months, a humid day, the dry room
    corona      the 1-D corona: onset voltage, I = C V (V - V0), thrust I d / mu, the jet
    breakdown   the spark predicted: inception, propagation, uniform, surface flashover -> the spark margin
    supply      the HV supply, simply: operating point, fold-back, input power, mass, a spark's energy
    thruster    ring thrusters in the 1-D model; the Mark IV / 4B / 4C presets (estimated geometry)

Later: ``field`` (2-D axisymmetric Poisson), ``charge`` (ion drift-diffusion, Kaptzov), ``ehd`` (the body force and
the jet; the OpenFOAM export), ``dbd`` (barrier discharges), ``chem`` (ozone, NOx).
"""
from .gas import Air, DRY, INDOOR
from .corona import Gap, thrust_per_watt, jet_velocity
from .breakdown import spark_margin, SparkMargin
from .supply import HVSupply
from .thruster import RingThruster, Ring, PRESETS, MARK4, MARK4B, MARK4C
from . import climate

__all__ = ["Air", "DRY", "INDOOR", "Gap", "thrust_per_watt", "jet_velocity", "spark_margin", "SparkMargin",
           "HVSupply", "RingThruster", "Ring", "PRESETS", "MARK4", "MARK4B", "MARK4C", "climate"]
