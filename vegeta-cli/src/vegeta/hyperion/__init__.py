"""Hyperion — optics: materials (Sellmeier indices, dn/dT, conductivity), paraxial ABCD elements (spaces, thin and
thick lenses, curved and flat mirrors at incidence, interfaces, slabs, Brewster plates, thermal lenses) in a
``Train``, Gaussian beams (q, w(z), M², through a train, mode overlap), resonators (round trip, stability, eigenmode,
spots, Gouy phase, FSR, finesse, linewidth, parameter sweeps) and the four-level end-pumped laser (threshold, slope,
output, optimum coupler, thermal lens) and the fibre laser (a step-index fibre's modes and bend loss, Yb:silica, the
pump/signal power evolution of an oscillator between gratings and of an amplifier, SRS, the core's temperature) — the
numbers a laser is designed on, pure numpy.

Hyperion computes what the engineer names — the elements, their spacings, the pump, the mode, the coupler — and
labels nothing as true: a paraxial Gaussian is the first description of a beam; apertures, aberrations and the
real pump profile come after it (the physical-optics and ray-tracing stages are on the to-do list).
"""
from .beam import BeamTrace, GaussianBeam, lens_for_waist, propagate_q, q_from, spot_from, trace
from .elements import (INF, BrewsterPlate, CurvedMirror, Element, FlatMirror, FreeSpace, Interface, Slab, ThermalLens, ThickLens,
                       ThinLens, Train, abcd_power, element_from_dict, is_mirror)
from .export import load_design, save_design
from .fibre import (YB_SILICA, ActiveFibre, Dopant, Fibre, FibreAmplifier, FibreLaser, FibreOscillator, FibreSolution, core_temperature, ions_for_absorption,
                    srs_threshold)
from .gain import ND_YAG, GainMedium, Laser, Rod, effective_pump_radius, medium, photon_energy, rod_temperature_rise, thermal_lens
from .materials import AIR_STP, BK7, FUSED_SILICA, MATERIALS, SAPPHIRE, YAG, Material, get, index_of, names
from .resonator import Mode, Resonator, kogelnik_li, two_mirror
from .result import CommandRecord, Result, ResultError
from .thermal import (ALUMINIUM, BLACK_PC, BLACK_PE, BRICK, CLEAR_PC, LASER_PC, MILD_STEEL, SOLIDS, STAINLESS, HeatingResult, Solid,
                      gaussian_steady_rise, heat_plate, irradiance, semi_infinite_surface_rise, solid, steady_plate, thin_plate_time_to)
from .welding import DeliveryFibre, WeldHead, keyhole_threshold, penetration, penetration_curve, process_window, rosenthal_pool, speed_for_penetration

__version__ = "0.1.0"

__all__ = [
    "AIR_STP", "ActiveFibre", "BK7", "BeamTrace", "BrewsterPlate", "CommandRecord", "CurvedMirror", "Dopant", "Element", "FUSED_SILICA", "Fibre",
    "FibreAmplifier", "FibreLaser", "FibreOscillator", "FibreSolution", "FlatMirror", "FreeSpace", "GainMedium", "GaussianBeam", "INF", "Interface", "Laser",
    "MATERIALS", "Material", "Mode", "ND_YAG", "Resonator", "Result", "ResultError", "Rod", "SAPPHIRE", "Slab", "ThermalLens", "ThickLens",
    "ThinLens", "Train", "YAG", "YB_SILICA", "abcd_power", "core_temperature", "effective_pump_radius", "element_from_dict", "get", "index_of",
    "ions_for_absorption", "is_mirror", "kogelnik_li", "lens_for_waist", "load_design", "medium", "names", "photon_energy", "propagate_q", "q_from",
    "rod_temperature_rise", "save_design", "spot_from", "srs_threshold", "thermal_lens", "trace", "two_mirror",
    "ALUMINIUM", "BLACK_PC", "BLACK_PE", "BRICK", "CLEAR_PC", "LASER_PC", "MILD_STEEL", "SOLIDS", "STAINLESS", "HeatingResult", "Solid",
    "gaussian_steady_rise", "heat_plate", "irradiance", "semi_infinite_surface_rise", "solid", "steady_plate", "thin_plate_time_to",
    "DeliveryFibre", "WeldHead", "keyhole_threshold", "penetration", "penetration_curve", "process_window", "rosenthal_pool", "speed_for_penetration",
]
