"""The quadcopter frame's vibration and life (notebook 08 Part 3), lifted as it was: the operating points with their
propeller unbalance (cell 80), the frame with the motors and the stack as point masses and its unit load cases (cells 82,
90), the three missions (cell 93), the PETG-CF fatigue curve (cell 97), the usage mixes (cells 103, 104).

The notebook's globals are arguments here; the FEA runs in ``workflows._common.unit_fea``, the life in ``life``.
"""
from __future__ import annotations

import math

from vegeta import boreas, chronos, talos

from .quad_frame_analysis import PETG_CF

N_MODES = 8                                                                          # cell 84
ELEMENT_MM = {"smoke": 8.0, "quick": 6.0, "full": 5.0}                               # full: cell 82's 5 mm
CURVE = talos.FatigueCurve("PETG-CF (assumed)", sigma_f=80.0, b=-0.11, ultimate=55.0,
                           source="assumed Basquin fit; replace with coupon tests")    # cell 97
USAGE = {"inspection": 0.6, "freestyle": 0.15, "cruise": 0.25}                      # cell 103
N_FLIGHTS = 40000                                                                    # cell 103
MIXES = {"inspection-heavy": {"inspection": 0.6, "freestyle": 0.15, "cruise": 0.25},
         "freestyle-heavy":  {"inspection": 0.2, "freestyle": 0.6,  "cruise": 0.2},
         "cruise-only":      {"inspection": 0.0, "freestyle": 0.0,  "cruise": 1.0}}   # cell 104
BALANCE_FACTOR = 2.5                                                                 # cell 104: G 6.3 -> G 2.5


def prop_points(prop: boreas.Propeller, points: dict) -> dict:
    """Cell 80: rpm, thrust and the propeller's unbalance force at hover, cruise and full throttle (``points`` maps
    those names to boreas operating points; notebook: hover, cruise, punch)."""
    return {name: {"rpm": pt.rpm, "thrust_N": pt.thrust, "unbalance_N": boreas.excitations(prop, pt.rpm)["unbalance_force_n"]}
            for name, pt in ((n, points[n]) for n in ("hover", "cruise", "full"))}


def masses_t(parts_g_without_frame: float, motor_kg: float, prop_kg: float, motors: int = 4) -> tuple[float, float]:
    """Cell 80: the motor and propeller on each pad and everything else on the stack bolts, in tonnes. The notebook's
    parts table holds the frame too and subtracts it again: here the parts come without the frame."""
    motor_prop = (motor_kg + prop_kg) * 1e-3
    stack = (parts_g_without_frame - motors * (motor_kg + prop_kg) * 1000) * 1e-6
    return motor_prop, stack


def frame_regions(p):
    """Cell 82."""
    R, m, r = p["wheelbase"] / 2, p["motor_pattern"] / 2, p["motor_hole"] / 2 + 0.2
    regions = []
    for k in range(4):
        a = math.radians(45 + 90 * k)
        cx, cy = R * math.cos(a), R * math.sin(a)
        regions.append(talos.SurfacesInBox(f"motor{k}", (cx - m - r, cy - m - r, -0.1, cx + m + r, cy + m + r, 100.0)))
    s, r = p["stack_pattern"] / 2, p["stack_hole"] / 2 + 0.2
    for k, (sx, sy) in enumerate([(s, s), (-s, s), (s, -s), (-s, -s)]):
        regions.append(talos.SurfacesInBox(f"stack{k}", (sx - r, sy - r, -0.1, sx + r, sy + r, 100.0)))
    return regions


def models(step, p: dict, motor_prop_t: float, stack_t: float, max_thrust_N: float, element_mm: float = 5.0):
    """Cells 82 and 90: the modal model (point masses, no loads) and the unit cases ``{name: (model, level)}``."""
    regions = frame_regions(p)
    supports = [talos.FixedSupport(f"stack{k}") for k in range(4)]
    masses = [talos.PointMass(f"motor{k}", motor_prop_t) for k in range(4)] + \
             [talos.PointMass(f"stack{k}", stack_t / 4) for k in range(4)]
    mesh = talos.MeshSettings(element_size=element_mm)

    def model(loads, name):
        return talos.StructuralModel(step, "mm-N-MPa", talos.Material(**PETG_CF), regions, supports, loads, mesh, name=name,
                                     masses=masses)

    unit = {
        "thrust":    (model([talos.Force(f"motor{k}", fz=max_thrust_N) for k in range(4)], "thrust"), max_thrust_N),
        "unbalance": (model([talos.Force("motor0", fx=1.0)], "unbalance"), 1.0),
        "landing":   (model([talos.Force("motor0", fz=40.0)], "landing"), 40.0),
    }
    return model([], "modal"), unit


def missions(PROP: dict, MAX_THRUST_N: float) -> dict:
    """Cell 93 (the profile plots left out)."""
    def unb(point, rpm=None, force=None):
        rpm = rpm or PROP[point]["rpm"]; force = force if force is not None else PROP[point]["unbalance_N"]
        return chronos.Excitation(f"unbalance {point}", rpm / 60, force, "unbalance")

    SPOOL = chronos.Excitation("unbalance spool-up", 4500 / 60, 0.06, "unbalance")     # passes through the arm modes
    HOVER, CRUISE = PROP["hover"]["thrust_N"], PROP["cruise"]["thrust_N"]

    return {
        "inspection": chronos.Mission("inspection", (
            chronos.Segment("spool-up", 3, {"thrust": 0.5}, (SPOOL,)),
            chronos.Segment("take-off", 8, {"thrust": 2.0}, (unb("hover"),)),
            chronos.Segment("hover + slow moves", 660, {"thrust": HOVER}, (unb("hover"),)),
            chronos.Segment("position changes", 4, {"thrust": 1.8}, (unb("cruise"),), repeat=12),
            chronos.Segment("landing", 3, {"thrust": 0.6, "landing": 12.0}),
        ), "12 min structure inspection: mostly hover, gentle moves, soft landing"),
        "freestyle": chronos.Mission("freestyle", (
            chronos.Segment("spool-up", 3, {"thrust": 0.5}, (SPOOL,)),
            chronos.Segment("hover", 60, {"thrust": HOVER}, (unb("hover"),)),
            chronos.Segment("punch-out", 1.5, {"thrust": MAX_THRUST_N}, (unb("full", force=0.38),), repeat=40),
            chronos.Segment("hard turns", 2.0, {"thrust": 4.0}, (unb("cruise", rpm=15000, force=0.20),), repeat=60),
            chronos.Segment("cruise between tricks", 120, {"thrust": CRUISE}, (unb("cruise"),)),
            chronos.Segment("hard landing", 2, {"thrust": 0.8, "landing": 40.0}),
        ), "5 min freestyle: 40 punch-outs, 60 hard turns, one hard landing"),
        "cruise": chronos.Mission("cruise", (
            chronos.Segment("spool-up", 3, {"thrust": 0.5}, (SPOOL,)),
            chronos.Segment("climb", 20, {"thrust": 2.5}, (unb("cruise"),)),
            chronos.Segment("cruise out", 420, {"thrust": CRUISE}, (unb("cruise"),)),
            chronos.Segment("gust corrections", 2.0, {"thrust": 2.4}, (unb("cruise"),), repeat=30),
            chronos.Segment("cruise back", 420, {"thrust": CRUISE}, (unb("cruise"),)),
            chronos.Segment("landing", 3, {"thrust": 0.6, "landing": 25.0}),
        ), "15 min tilted cruise out and back with gust corrections"),
    }
