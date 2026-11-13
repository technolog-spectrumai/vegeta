"""Perseus's chassis, other manipulators, a second hand — the investigation behind ``docs/perseus_options.md``.

Perseus (``perseus_robot``) keeps its chassis: the Pekari track architecture at 1100 x 520 mm, 150 mm tracks, two
1.5 kW drives, 1.9 kWh. On it, six configurations (``CONFIGS``), each judged the same way:

* ``boom saw`` — Perseus as it is: the 4-joint boom on two electric cylinders, the 2 kW chainsaw;
* ``boom grab`` — the same boom with a two-jaw grab in place of the saw (what the cylinders lift, ``boom_capacity``);
* ``arm6 saw`` — the boom replaced by the family's 6-DOF joint-module arm (``pekari_arms.PUMBA_MAIN_ARM``) with the
  saw on its quick coupler (and the grab on it in the next job: one arm, changeable hands);
* ``boom saw + light hand`` — the boom moved 0.13 m left, Pekari-2M's 5-DOF arm (``PEKARI_ARM``) on the right;
* ``boom saw + arm6 grab`` — the boom moved left, the 6-DOF arm with the grab on the right: a holding hand;
* ``arm6 saw + arm6 grab`` — two 6-DOF arms (Pumba's main arm twice): saw left, grab right.

What is computed (``evaluate``): the mass and what it adds to Perseus; the drive on gravel (``land_comparison
.drive_numbers``: the 20 deg climb at peak, the steepest grade held continuously, the ground pressure) and the
driving endurance; the cutting arm's stall safety factor cutting outstretched with the loads x1.5 (Perseus's
measure); and for a second hand — the question — how much of the **cut zone** it reaches (branches 1.2-2.2 m ahead
of the hull centre, 0.8-2.2 m up, on the centre line, where the saw works) and what it holds there (its peak
rating at that reach), the largest branch piece it can hold (green spruce, 1 m beyond its grip), and the static
tip margin with both arms out, the saw cutting and the hand holding that piece. The roof: two pedestals side by
side need the boom off the centre line (``ROOF``).

Why a second hand at all (``branch_cases``): a cut branch falls — onto the machine, onto a line, onto people — and
a branch propped at its far end pinches the bar (``ogma.cutting.kerf_opens``); a hand that holds the piece beyond
the cut lets it be lowered instead of dropped and keeps the kerf open. Its cost: the mass on 1.5 kW drives, the
roof, the price of the modules.
"""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from ogma import cutting

import land_comparison as lc
import pekari_arms as pa
import pekari_family as pf
import pekari_rover_robot as prr
import perseus_robot as pbr
from perseus_rover import PERSEUS_CHASSIS

__all__ = ["CUT_ZONE", "ROOF", "CONFIGS", "WOOD", "STRONG_WRIST", "perseus_platform", "boom_capacity",
           "boom_tipping_load", "boom_lift", "branch_cases", "evaluate",
           "table", "markdown"]

G = 9.81
WOOD = "softwood (spruce, pine), green"
#: Where Perseus's saw works: bar-middle positions (x ahead of the hull centre, z above the ground) [m], y = 0.
CUT_ZONE = {"x": (1.2, 2.2), "z": (0.8, 2.2), "n": 11}
#: Two pedestals on the 520 mm roof: the boom (r 0.11) at y = +0.13, a second arm's (r 0.05-0.10) at y = -0.15.
ROOF = {"boom_y": 0.13, "hand_y": -0.15, "x": 0.20, "half_width": 0.26}
HOLD_BEYOND = 1.0                # the piece held extends this far beyond the grip [m]


# ----------------------------------------------------------------------------------------------- the chassis
def perseus_platform(name: str, mounts: tuple) -> pf.Platform:
    """A ``pekari_family.Platform`` on Perseus's chassis, parts, drives and battery with ``mounts`` (no deck)."""
    v = pbr.PERSEUS
    return pf.Platform(
        name=name, chassis=dict(PERSEUS_CHASSIS), cad=dict(pbr.PERSEUS_CAD), mounts=mounts, stabilizers=(),
        parts_kg=dict(v.parts_kg), drive=v.drive, materials=dict(v.materials), shell_t_mm=v.shell_t_mm,
        wheel_fill=v.wheel_fill, bus_V=48.0, battery_Wh=v.battery_Wh, electronics_W=v.electronics_W,
        payload_kg=0.0, payload_range_kg=(0.0, 0.0), v_design=2.0, sensors=("rgbd_front", "lidar_3d", "gnss", "imu"),
        track_mu=v.track_mu, internal_f0=v.internal_f0,
        assumptions={"chassis": "Perseus's chassis, parts, drives and battery unchanged (perseus_robot.PERSEUS)",
                     "battery": "re-placed by prr.masses for the new arms (as Perseus's is for its boom)"})


SAW_ARM = pa.with_tool(pa.PUMBA_MAIN_ARM, "chainsaw 2 kW")
GRAB_ARM = pa.PUMBA_MAIN_ARM


# ----------------------------------------------------------------------------------------------- the boom
def _boom_shoulder_world(boom_y: float = 0.0):
    g = pbr.arm_geometry()
    return (g["x"], boom_y, g["z_world"])


STRONG_WRIST = replace(pbr.PERSEUS, arm_drives={**pbr.PERSEUS.arm_drives, "wrist": ("rotary", "cycloidal 400 Nm, brake")})


def _boom_q(r, z, v, yaw=0.0):
    g = pbr.arm_geometry(v=v)
    z = g["z_world"] if z is None else z
    q = pbr.arm_ik((g["x"] + r - g["Lbar"] / 2, 0.0, z - g["z_hull"]), 0.0, v=v)
    return None if q is None else (yaw,) + tuple(q[1:])


def boom_tipping_load(r: float, yaw: float = 0.0, z: float | None = None, grab_kg: float = 5.0, v=pbr.PERSEUS) -> float:
    """The load [kg] in a grab on Perseus's boom at reach ``r`` and heading ``yaw`` that tips the machine
    (``perseus_robot.tip_over``: the support point reaches the tracks' footprint edge)."""
    q = _boom_q(r, z, v, yaw)
    if q is None:
        return None
    saw_kg = cutting.get_saw(v.saw).mass_kg

    def stands(m):
        t = pbr.tip_over(q, v=v, load={"total": (0.0, 0.0, -(m + grab_kg - saw_kg) * G)})
        return min(t["margin_front_m"], t["margin_rear_m"], t["margin_side_m"]) > 0
    lo, hi = 0.0, 1000.0
    for _ in range(50):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if stands(mid) else (lo, mid)
    return lo


def boom_lift(r: float, v=pbr.PERSEUS) -> dict:
    """What the boom with a grab may lift at reach ``r``: the drives' rating (``boom_capacity``), 75 % of the
    tipping load straight ahead and sideways, the smaller and what limits it."""
    arm = boom_capacity(r, v=v)
    tip = min(boom_tipping_load(r, 0.0, v=v), boom_tipping_load(r, math.pi / 2, v=v))
    cap = min(arm, 0.75 * tip)
    return {"arm_kg": arm, "tipping_kg": tip, "capacity_kg": cap, "limit": "drives" if arm <= 0.75 * tip else "stability"}


def boom_capacity(r: float, z: float | None = None, sf: float = 1.5, grab_kg: float = 5.0, v=pbr.PERSEUS) -> float:
    """What Perseus's boom lifts [kg] with a grab (``grab_kg``, in place of the saw) holding the load at the
    bar-middle point at horizontal reach ``r`` from the shoulder and height ``z`` above the ground (default: the
    shoulder's), the head level: every pitch joint at stall / ``sf`` (the cylinders on their levers in that pose,
    ``perseus_robot.arm_torques``). The saw's mass is taken off and the grab's put on at the same point."""
    q = _boom_q(r, z, v)
    if q is None:
        return None                                              # out of the boom's reach (or its joint ranges)
    saw_kg = cutting.get_saw(v.saw).mass_kg

    def ok(m):
        t = pbr.arm_torques(q, v=v, load={"total": (0.0, 0.0, -(m + grab_kg - saw_kg) * G)})
        return all(t[f"arm_{j}"]["sf_stall"] >= sf for j in ("shoulder", "elbow", "wrist"))
    if not ok(0.0):
        return 0.0
    lo, hi = 0.0, 500.0
    for _ in range(50):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if ok(mid) else (lo, mid)
    return lo


def _boom_reaches(x, z) -> bool:
    g = pbr.arm_geometry()
    return pbr.arm_ik((x - g["Lbar"] / 2, 0.0, z - g["z_hull"]), 0.0) is not None


# ----------------------------------------------------------------------------------------------- the cut zone
def _zone():
    n = CUT_ZONE["n"]
    xs = np.linspace(*CUT_ZONE["x"], n)
    zs = np.linspace(*CUT_ZONE["z"], n)
    return [(float(x), float(z)) for x in xs for z in zs]


def _hand_at(arm, shoulder, x, z):
    """The hand's reach to (x, 0, z) with the tool level or pointing 30 deg down (whichever IK solves), and its
    rating there (peak / 1.5 at that horizontal reach); None when out of reach."""
    for elev in (0.0, -math.pi / 6, math.pi / 6):
        q = pa.ik(arm, (x, 0.0, z), elev, shoulder)
        if q is not None:
            r = math.hypot(x - shoulder[0], 0.0 - shoulder[1])
            return q, pa.rated_load(arm, min(r, pa.reach(arm)))
    return None


def branch_cases() -> list:
    """The pieces a holding hand would take: green spruce, ``HOLD_BEYOND`` beyond the grip, by diameter; and the
    kerf on a branch cantilevered 1.5 m beyond the cut, cut from below (pinches) and propped (pinches from the top)."""
    out = []
    for d in (0.05, 0.08, 0.10, 0.15, 0.20, 0.30):
        m = cutting.branch_mass(d, HOLD_BEYOND + 0.3, WOOD)              # the 0.3 m between the cut and the grip
        k = cutting.kerf_opens(d, 1.5, WOOD, cut_from="bottom")
        out.append({"d_m": d, "piece_kg": m, "kerf_bottom_cut": k["verdict"].split(":")[0]})
    return out


# ----------------------------------------------------------------------------------------------- the configurations
CONFIGS = ("boom saw", "boom grab", "arm6 saw", "boom saw + light hand", "boom saw + arm6 grab", "boom saw + long arm6 grab",
           "arm6 saw + arm6 grab")
HANDS = {"light hand": pa.PEKARI_ARM, "arm6 grab": GRAB_ARM, "long arm6 grab": pa.CALYDON_MAIN_ARM}


def _drive(mass_kg):
    g = prr.geometry(pbr.chassis_params(pbr.design_params()))
    dn = lc.drive_numbers(mass_kg * G, g, pbr.PERSEUS.drive, pbr.PERSEUS.internal_f0)
    dn["drive_endurance_h"] = pbr.PERSEUS.battery_Wh * lc.USABLE / (dn["traction_W"] + pbr.PERSEUS.electronics_W)
    return dn


def _perseus_masses(boom_y: float, q=None):
    """Perseus's masses (its battery placement) with the boom shifted to ``boom_y`` (pose ``q``, default stowed)."""
    out = []
    for n, m, x, y, z in pbr.masses(q=q):
        is_boom = n in pbr.arm_masses()
        out.append((n, m, x, y + (boom_y if is_boom else 0.0), z))
    return out


def _margin(ms, load_pts=()):
    """Static tip margin [m] and angle [deg] of the masses ``ms`` plus point loads ``load_pts`` [(kg, x, y, z)] over
    Perseus's track footprint."""
    g = prr.geometry(pbr.chassis_params(pbr.design_params()))
    allm = [(m, x, y, z) for _, m, x, y, z in ms] + list(load_pts)
    M = sum(m for m, *_ in allm)
    cx = sum(m * x for m, x, _, _ in allm) / M
    cy = sum(m * y for m, _, y, _ in allm) / M
    cz = sum(m * z for m, _, _, z in allm) / M
    xs = g["wheel_x"]
    mid = sum(xs) / len(xs)
    half = g["track_center_y"] + g["b"] / 2
    d = min(mid + g["L"] / 2 - cx, cx - (mid - g["L"] / 2), half - abs(cy))
    return d, math.degrees(math.atan2(d, cz)), M


def evaluate(name: str) -> dict:
    """The numbers of configuration ``name`` (see the module)."""
    base_kg = pbr.total_mass()
    zone = _zone()
    out = {"config": name}
    work = (1.6, 0.0, 1.4)                                         # the reference cut: 1.6 m ahead, 1.4 m up
    if name.startswith("boom"):
        boom_y = 0.0 if name in ("boom saw", "boom grab") else ROOF["boom_y"]
        mass = base_kg + (cutting.get_saw(pbr.PERSEUS.saw).mass_kg * 0 if name != "boom grab" else 5.0 - 6.5)
        cutter = "boom"
        reach_cut = sum(_boom_reaches(x, z) for x, z in zone) / len(zone)
        t = pbr.arm_torques(pbr.elevations_to_q(0, 0, 0), load=pbr.PERSEUS.cut_loads(dynamic=1.5))
        saw_sf = (None if name == "boom grab" else
                  min(t[f"arm_{j}"]["sf_stall"] for j in ("shoulder", "elbow", "wrist")))
        lift = {r: boom_lift(r)["capacity_kg"] for r in (1.4, 1.8, 2.1)} if name == "boom grab" else None
        g = pbr.arm_geometry()
        root = (work[0] - g["Lbar"] / 2, 0.0, work[2] - g["z_hull"])
        ms = _perseus_masses(boom_y, pbr.arm_ik(root, 0.0))
        hand = HANDS.get(name.split(" + ")[-1]) if "+" in name else None
        cutter_ok = [_boom_reaches(x, z) for x, z in zone]
        hand_y = ROOF["hand_y"]
    else:                                                          # joint-module arms on Perseus's chassis
        boom_y = 0.0 if name == "arm6 saw" else ROOF["boom_y"]
        mounts = (pf.Mount("main", SAW_ARM, x=ROOF["x"], y=boom_y),)
        if name == "arm6 saw + arm6 grab":
            mounts += (pf.Mount("aux", GRAB_ARM, x=ROOF["x"], y=ROOF["hand_y"]),)
        pl = perseus_platform(f"Perseus chassis: {name}", mounts)
        mass = pf.total_mass(pl)
        cutter = "arm6"
        sh = pf.shoulder(pl, "main")
        cutter_ok = [_hand_at(SAW_ARM, (sh[0], sh[1], sh[2]), x, z) is not None for x, z in zone]
        reach_cut = sum(cutter_ok) / len(zone)
        c = pa.cut_check(SAW_ARM, 0.95 * pa.reach(SAW_ARM), dynamic=1.5)
        saw_sf = min(c["sf_stall"].values())
        lift = ({r: pf.lift_capacity(perseus_platform("arm6 grab", (pf.Mount("main", GRAB_ARM, x=ROOF["x"], y=0.0),)),
                                     "main", r, math.pi / 2)["capacity_kg"] for r in (1.0, 1.4)}
                if name == "arm6 saw" else None)
        qa = pa.ik(SAW_ARM, work, 0.0, sh)
        ms = [m for m in pf.masses(pl, {"main": qa} if qa else None) if not m[0].startswith("aux:")]
        hand = GRAB_ARM if name == "arm6 saw + arm6 grab" else None
        hand_y = ROOF["hand_y"]
    out.update({"mass_kg": mass, "added_kg": mass - base_kg, "cutter": cutter, "saw_sf_stall": saw_sf,
                "cut_zone_reached_by_cutter": reach_cut, "lift_kg": lift})
    # the second hand
    hold_kg = None
    if hand is not None:
        roof = (PERSEUS_CHASSIS["ground_clearance"] + PERSEUS_CHASSIS["hull_height"]) / 1000
        sh_h = (ROOF["x"], hand_y, roof + hand.pedestal[0])
        got = [_hand_at(hand, sh_h, x, z) for x, z in zone]
        cover = sum(gg is not None for gg in got) / len(zone)
        # where both work: the cut points the saw reaches and the hand reaches 0.3 m beyond (the grip on the piece)
        both = [(x, z) for (x, z), ok in zip(zone, cutter_ok) if ok and _hand_at(hand, sh_h, x + 0.3, z) is not None]
        shared = len(both) / max(sum(cutter_ok), 1)
        holds = sorted(_hand_at(hand, sh_h, x + 0.3, z)[1] for x, z in both)
        hold_kg = holds[len(holds) // 2] if holds else 0.0          # the median over the shared points
        at_work = None
        if both:                                                   # the tip check at the shared point farthest out
            xw, zw = max(both)
            at_work = _hand_at(hand, sh_h, xw + 0.3, zw)
        mass_h = pa.total_mass(hand)
        if "arm6 saw + arm6 grab" != name:
            out["mass_kg"] += mass_h
            out["added_kg"] += mass_h
        biggest = max([c["d_m"] for c in branch_cases() if c["piece_kg"] <= hold_kg], default=0.0)
        # tip: both arms out at the reference cut, the hand holding the biggest piece it can
        hand_pts = []
        if at_work:
            hand_pts = [(m, x, y, z) for _, m, x, y, z in pa.mass_points(hand, at_work[0], sh_h, hold_kg)]
        else:
            hand_pts = [(m, x, y, z) for _, m, x, y, z in pa.mass_points(hand, pa.pose(hand, "folded"), sh_h)]
        if name == "arm6 saw + arm6 grab":
            ms = [m for m in ms if not m[0].startswith("aux:")]
        d, ang, _ = _margin(ms, hand_pts)
        out.update({"hand": hand.name, "hand_kg": mass_h, "cut_zone_reached_by_hand": cover, "shared": shared,
                    "hold_at_cut_kg": hold_kg,
                    "largest_piece_d_m": biggest, "tip_margin_m": d, "tip_deg": ang})
    else:
        d, ang, _ = _margin(ms)
        out.update({"hand": None, "hand_kg": 0.0, "cut_zone_reached_by_hand": None, "shared": None, "hold_at_cut_kg": None,
                    "largest_piece_d_m": None, "tip_margin_m": d, "tip_deg": ang})
    out.update(_drive(out["mass_kg"]))
    # the price of what is added (class prices, pekari_family.PRICE_EUR): the hand's modules
    if hand is not None and name != "arm6 saw + arm6 grab":
        lo = sum(pf.PRICE_EUR[k][0] for k in hand.drives.values())
        hi = sum(pf.PRICE_EUR[k][1] for k in hand.drives.values())
        out["hand_modules_EUR"] = (lo, hi)
    elif name == "arm6 saw + arm6 grab":
        out["hand_modules_EUR"] = tuple(sum(pf.PRICE_EUR[k][i] for k in GRAB_ARM.drives.values()) for i in (0, 1))
    else:
        out["hand_modules_EUR"] = None
    return out


def table():
    """The configurations as a DataFrame (rows: metrics, columns: configurations), formatted."""
    import pandas as pd

    rows = {c: evaluate(c) for c in CONFIGS}

    def f(v, fmt="{:.0f}"):
        if v is None:
            return "—"
        if isinstance(v, bool):
            return "yes" if v else "no"
        if isinstance(v, dict):
            return ", ".join(f"{k:g} m: " + ("out of reach" if x is None else f"{x:.0f} kg") for k, x in v.items())
        if isinstance(v, tuple):
            return f"{v[0] / 1000:.1f}–{v[1] / 1000:.1f} k"
        return fmt.format(v)
    spec = [("mass_kg", "mass [kg]", "{:.0f}"), ("added_kg", "added to Perseus [kg]", "{:+.0f}"),
            ("climb20", "20° grade (peak, traction)", None), ("grade_cont_deg", "steepest grade held continuously [°]", "{:.0f}"),
            ("ground_kPa", "ground pressure [kPa]", "{:.1f}"), ("drive_endurance_h", "driving endurance [h]", "{:.1f}"),
            ("saw_sf_stall", "saw: stall SF cutting outstretched (loads x1.5)", "{:.2f}"),
            ("cut_zone_reached_by_cutter", "cut zone the cutter reaches", "{:.0%}"),
            ("lift_kg", "lift with the grab on the same arm (min of drives, 75 % tipping)", None),
            ("hand", "second hand", "{}"), ("hand_kg", "  its mass [kg]", "{:.1f}"),
            ("cut_zone_reached_by_hand", "  cut zone it reaches", "{:.0%}"),
            ("shared", "  of the saw's cuts it can hold (0.3 m beyond the cut)", "{:.0%}"),
            ("hold_at_cut_kg", "  holds there, median (peak / 1.5) [kg]", "{:.1f}"),
            ("largest_piece_d_m", "  largest branch piece it holds (1.3 m long) [m]", "{:.2f}"),
            ("tip_margin_m", "tip margin, arms out at the cut [m]", "{:.2f}"), ("tip_deg", "  tip angle [°]", "{:.0f}"),
            ("hand_modules_EUR", "second hand's modules (class prices)", None)]
    data = {c: [f(rows[c][k], fmt) if fmt else f(rows[c][k]) for k, _, fmt in spec] for c in CONFIGS}
    return pd.DataFrame(data, index=[lab for _, lab, _ in spec])


def markdown() -> str:
    df = table()
    head = "| | " + " | ".join(df.columns) + " |\n|---|" + "---|" * len(df.columns) + "\n"
    return head + "".join(f"| {i} | " + " | ".join(df.loc[i]) + " |\n" for i in df.index)
