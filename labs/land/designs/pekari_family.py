"""The Pekari family (``docs/pekari_pumba.md``): one description and one set of calculations for every tracked
platform built on the Pekari Rover's architecture with manipulators — **Pekari-2M** (``pekari_2m_robot``, the light
dual-arm response platform), **Pumba** (``pumba_robot``, the heavy dual-arm derivative at WILK-class mass) and
**Calydon** (``calydon_robot``, the first, heavier Pumba design: more lift, 444 kg). The Pekari Rover itself
(``pekari_rover_robot``) is untouched: a ``Platform`` is its chassis parameters plus arms, stabilizers and a
parts list, and every chassis number comes from ``pekari_rover_robot`` (``prr``) with the platform's keywords —
the way Perseus (``perseus_robot``) builds on it.

Frames: the vehicle frame has x forward from the road wheels' centre, y left, z up from the ground (the grouser
tips) — ``prr.masses``' frame. Arm mounts are given as (x, y) on the roof; the shoulder axis is the arm's pedestal
height above the roof. The Chiron hull frame is the hull's centre (``z_hull``).

What a platform answers (each with its assumptions named in ``Platform.assumptions``):

* ``mass_budget`` / ``total_mass``: the chassis budget (``prr.mass_budget``), the arms (``pekari_arms.masses``),
  the stabilizers.
* ``masses`` / ``cg``: every mass in 3-D for given arm poses, loads in the grippers and the deck payload; the
  battery is placed by ``prr.masses`` so the empty platform with folded arms balances over the road wheels' centre.
* ``support_polygon``, ``stability``: the static tip-over margin of the CG's ground projection to the support
  polygon (the tracks' contact patches, and the stabilizer pads when deployed) and the tip angle about each edge.
* ``tipping_load`` / ``lift_capacity``: the load one arm can hold at a reach and yaw before the platform tips, and
  the capacity = min(arm's structural rating, ``TIP_RATIO`` × tipping load) — the crane rule of rating at 75 %.
* ``drive_check``: resistance on a soil and a grade (``ogma.tracks``) against the two track drives and traction.
* ``power`` / ``endurance``: the average electrical power of a mission profile and the hours on the battery.
* ``capabilities``: what the fleet coordinator (``fleet.py``) allocates by.
* ``robot``: the Chiron (MuJoCo) model — ``prr.pekari`` with the arms and stabilizers as extra children.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

from ogma import actuators as act
from ogma import tracks

import pekari_arms as pa
import pekari_rover_robot as prr

__all__ = ["G", "TIP_RATIO", "Mount", "Stabilizer", "Platform", "chassis_params", "geometry", "roof_z", "shoulder",
           "cad_numbers", "mass_budget", "total_mass", "poses", "masses", "cg", "support_polygon", "stability",
           "tipping_load", "lift_capacity", "drive_check", "power", "endurance", "capabilities", "robot", "qpos",
           "summary", "PRICE_EUR", "bom", "bom_total"]

G = 9.81
TIP_RATIO = 0.75                       # rated load / tipping load (the mobile-crane rule)
PEKARI_LOADED_KG = 25.0                # the Pekari Rover the roller gains were tuned on (pekari_rover_robot)


@dataclass(frozen=True)
class Mount:
    """An arm on the roof: its name (the Chiron joint prefix), the arm, the pedestal's (x, y) [m]."""

    name: str
    arm: pa.ArmSpec
    x: float
    y: float


@dataclass(frozen=True)
class Stabilizer:
    """A swing-out leg with a pad: the pad's deployed (x, y) [m], its radius, the drive (``ogma.actuators``
    linear key), the stroke from stowed to the ground and the leg's mass (beam, hinge, drive, pad) [kg]."""

    name: str
    x: float
    y: float
    pad_radius: float
    drive: str
    stroke: float
    mass_kg: float


@dataclass(frozen=True)
class Platform:
    """One machine of the family."""

    name: str
    chassis: dict                         # Pekari parameters (mm) — the deck is the Pekari basket
    cad: dict                             # stored CAD measurements of the chassis (prr.CAD's keys)
    mounts: tuple                         # Mount, ...
    stabilizers: tuple                    # Stabilizer, ... (may be empty)
    parts_kg: dict                        # the chassis parts list (Pekari's name prefixes)
    drive: str                            # one per track (ogma.actuators key)
    materials: dict
    shell_t_mm: float
    wheel_fill: float
    bus_V: float
    battery_Wh: float
    electronics_W: float                  # computer, radios, sensors
    payload_kg: float                     # the deck's nominal payload
    payload_range_kg: tuple               # the target (lo, hi)
    v_design: float                       # top speed asked for [m/s]
    sensors: tuple
    track_mu: float = 0.6
    internal_f0: float = 0.035
    colors: dict = field(default_factory=dict)
    assumptions: dict = field(default_factory=dict)
    bom_extra: dict = field(default_factory=dict)  # item -> (qty, EUR low, EUR high): what is not a catalogue drive
    band_track_kg_m: float = 0.0          # > 0: a rubber band track of this mass per metre instead of PA6 links + pins

    @property
    def n_bogies(self) -> int:
        return int(self.chassis["road_wheels"]) // 2

    def mount(self, name: str) -> Mount:
        return next(m for m in self.mounts if m.name == name)


# ----------------------------------------------------------------------------------------------- geometry
def chassis_params(pl: Platform) -> dict:
    return prr.design_params(pl.chassis)


_CACHE: dict = {}


def _key(pl: Platform, *extra) -> tuple:
    return (pl.name, tuple(sorted(pl.chassis.items())), *extra)


def geometry(pl: Platform) -> dict:
    """``prr.geometry`` of the chassis [m] plus the contact patch and the outer track edge (cached per chassis:
    the belt's length is a fine polygon)."""
    k = _key(pl, "geometry")
    if k not in _CACHE:
        _CACHE[k] = _geometry(pl)
    return dict(_CACHE[k])


def _geometry(pl: Platform) -> dict:
    g = prr.geometry(chassis_params(pl))
    xs = g["wheel_x"]
    x_mid = sum(xs) / len(xs)
    g["contact_front"] = x_mid + g["L"] / 2
    g["contact_rear"] = x_mid - g["L"] / 2
    g["half_width"] = g["track_center_y"] + g["b"] / 2
    return g


def roof_z(pl: Platform) -> float:
    p = pl.chassis
    return (p["ground_clearance"] + p["hull_height"]) * 1e-3


def z_hull(pl: Platform) -> float:
    p = pl.chassis
    return (p["ground_clearance"] + p["hull_height"] / 2) * 1e-3


def shoulder(pl: Platform, m: Mount | str) -> tuple:
    """The arm's shoulder point (x, y, z) [m] in the vehicle frame."""
    m = pl.mount(m) if isinstance(m, str) else m
    return (m.x, m.y, roof_z(pl) + m.arm.pedestal[0])


def cad_numbers(pl: Platform, *, recompute: bool = False) -> dict:
    """The chassis CAD numbers (stored; ``recompute`` rebuilds them with CadQuery through ``prr.cad_numbers``)."""
    if not recompute:
        return dict(pl.cad)
    return prr.cad_numbers(chassis_params(pl), recompute=True)


# ----------------------------------------------------------------------------------------------- masses
def _budget_kw(pl: Platform) -> dict:
    return {"drive": pl.drive, "shell_t_mm": pl.shell_t_mm, "materials": pl.materials, "wheel_fill": pl.wheel_fill}


def chassis_budget(pl: Platform) -> dict:
    """``prr.mass_budget`` of the chassis (cached per platform)."""
    k = _key(pl, "budget", id(pl))
    if k not in _CACHE:
        b = prr.mass_budget(chassis_params(pl), pl.cad, pl.parts_kg, **_budget_kw(pl))
        if pl.band_track_kg_m > 0:
            # a moulded rubber band (steel cords, inner drive and guide lugs) replaces the hinged links and their pins;
            # its row keeps the "track links" prefix so prr.masses places it on the belt's centroid
            b = {k: v for k, v in b.items() if not k.startswith(("track links", "track pins"))}
            L = geometry(pl)["belt_length"]
            b[f"track links: rubber band track 2x{L:.2f} m ({pl.band_track_kg_m:g} kg/m)"] = 2 * L * pl.band_track_kg_m
        _CACHE[k] = b
    return dict(_CACHE[k])


def mass_budget(pl: Platform) -> dict:
    """Every part [kg]: the chassis (``prr.mass_budget``), each arm (one row, ``pekari_arms.masses`` in detail),
    each stabilizer."""
    out = chassis_budget(pl)
    for m in pl.mounts:
        out[f"arm '{m.name}' ({m.arm.name}, {pa.arm_dof(m.arm)} DOF + gripper)"] = pa.total_mass(m.arm)
    for s in pl.stabilizers:
        out[f"stabilizer '{s.name}' (leg, {s.drive}, pad)"] = s.mass_kg
    return out


def total_mass(pl: Platform, payload_kg: float = 0.0) -> float:
    return sum(mass_budget(pl).values()) + payload_kg


def poses(pl: Platform, name: str = "folded") -> dict:
    """Every arm in its named pose: {mount: q (rad)}."""
    return {m.name: pa.pose(m.arm, name) for m in pl.mounts}


def _stab_point(pl: Platform, s: Stabilizer, deployed: bool) -> tuple:
    """The leg's mass point: at the pad when deployed, folded against the hull side otherwise."""
    g = geometry(pl)
    if deployed:
        return (s.x, s.y, s.pad_radius + 0.05)
    side = 1.0 if s.y > 0 else -1.0
    return (s.x, side * (g["half_width"] + 0.05), z_hull(pl))


def _extras_folded(pl: Platform) -> tuple:
    """(mass, Σ m x) of the arms folded and the stabilizers stowed — what the battery placement balances."""
    m_tot = mom = 0.0
    for m in pl.mounts:
        for _, kg, x, *_ in pa.mass_points(m.arm, pa.pose(m.arm, "folded"), shoulder(pl, m)):
            m_tot += kg
            mom += kg * x
    for s in pl.stabilizers:
        x = _stab_point(pl, s, False)[0]
        m_tot += s.mass_kg
        mom += s.mass_kg * x
    return m_tot, mom


def payload_point(pl: Platform) -> tuple:
    """The deck payload's centre (x, z) [m]: on the deck (the Pekari basket) floor, 0.1 m above it."""
    p = pl.chassis
    if p["basket_height"] > 0:
        return (p["basket_x"] * 1e-3, roof_z(pl) + p["basket_wall"] * 1e-3 + 0.10)
    return (0.0, roof_z(pl) + 0.10)


def _chassis_masses(pl: Platform, payload_kg: float) -> tuple:
    """The chassis masses (and the deck payload) in 3-D, cached per platform and payload (they do not move)."""
    k = _key(pl, "chassis_masses", float(payload_kg), id(pl))
    if k not in _CACHE:
        m_x, mom = _extras_folded(pl)
        ch = prr.masses(chassis_params(pl), payload_kg, cad=pl.cad, budget=chassis_budget(pl), extra_mass_kg=m_x,
                        extra_moment_kgm=mom, payload_pos=payload_point(pl))
        _CACHE[k] = tuple((n, kg, x, 0.0, z) for n, kg, x, z in ch)
    return _CACHE[k]


def masses(pl: Platform, arm_q: dict | None = None, *, loads: dict | None = None, payload_kg: float = 0.0,
           stabilizers_deployed: bool = False) -> list:
    """Every mass [(name, kg, x, y, z)] [kg, m] in the vehicle frame: the chassis (``prr.masses``, the battery
    placed for the folded platform), the arms at ``arm_q`` ({mount: q}, default folded) with ``loads`` ({mount:
    kg} at the TCP), the stabilizers, and ``payload_kg`` on the deck."""
    arm_q = poses(pl, "folded") if arm_q is None else {**poses(pl, "folded"), **arm_q}
    loads = loads or {}
    out = list(_chassis_masses(pl, payload_kg))
    for m in pl.mounts:
        for n, kg, x, y, z in pa.mass_points(m.arm, arm_q[m.name], shoulder(pl, m), loads.get(m.name, 0.0)):
            out.append((f"{m.name}: {n}", kg, x, y, z))
    for s in pl.stabilizers:
        out.append((f"stabilizer {s.name}", s.mass_kg, *_stab_point(pl, s, stabilizers_deployed)))
    return out


def cg(pl: Platform, arm_q: dict | None = None, **kw) -> dict:
    """Mass [kg] and CG (x, y, z) [m] (``kw``: ``masses``' keywords)."""
    ms = masses(pl, arm_q, **kw)
    M = sum(k for _, k, *_ in ms)
    return {"mass_kg": M, "x": sum(k * x for _, k, x, _, _ in ms) / M, "y": sum(k * y for _, k, _, y, _ in ms) / M,
            "z": sum(k * z for _, k, _, _, z in ms) / M}


# ----------------------------------------------------------------------------------------------- stability
def _convex_hull(pts):
    pts = sorted(set(pts))
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def support_polygon(pl: Platform, stabilizers_deployed: bool = False) -> list:
    """The support polygon [(x, y)] [m], counter-clockwise: the two tracks' contact patches (road wheel span plus
    a wheel, to the outer track edge) and the stabilizer pads' centres when deployed."""
    g = geometry(pl)
    xf, xr, hw = g["contact_front"], g["contact_rear"], g["half_width"]
    pts = [(xf, hw), (xr, hw), (xr, -hw), (xf, -hw)]
    if stabilizers_deployed:
        pts += [(s.x, s.y) for s in pl.stabilizers]
    return _convex_hull(pts)


def stability(pl: Platform, arm_q: dict | None = None, **kw) -> dict:
    """Static stability [m, deg] of the CG over the support polygon: ``margin`` the shortest distance from the
    CG's ground projection to an edge (negative = outside: it tips), ``tip_deg`` the smallest tip-over angle
    (atan(distance / CG height)) and the edge it is about, and the CG. ``kw``: ``masses``' keywords."""
    c = cg(pl, arm_q, **kw)
    poly = support_polygon(pl, kw.get("stabilizers_deployed", False))
    best = None
    n = len(poly)
    for i in range(n):
        (x0, y0), (x1, y1) = poly[i], poly[(i + 1) % n]
        ex, ey = x1 - x0, y1 - y0
        L = math.hypot(ex, ey)
        d = (ex * (c["y"] - y0) - ey * (c["x"] - x0)) / L          # > 0 inside a counter-clockwise polygon
        if best is None or d < best[0]:
            best = (d, ((x0, y0), (x1, y1)))
    d, edge = best
    return {"margin_m": d, "tip_deg": math.degrees(math.atan2(d, c["z"])), "edge": edge, "cg": c,
            "stable": d > 0.0}


def _reach_pose(pl: Platform, m: Mount, r: float, yaw: float, z: float | None = None):
    """The arm at horizontal reach ``r`` from the shoulder at heading ``yaw`` (vehicle frame), the TCP at height
    ``z`` above the ground (default: the shoulder's height), tool level; None out of reach."""
    sh = shoulder(pl, m)
    z = sh[2] if z is None else z
    tgt = (sh[0] + r * math.cos(yaw), sh[1] + r * math.sin(yaw), z)
    q = pa.ik(m.arm, tgt, 0.0, sh)
    if q is None and r >= pa.reach(m.arm) - 1e-6 and abs(z - sh[2]) < 1e-9:
        q = {**pa.pose_level(m.arm), "yaw": yaw}
    return q


def tipping_load(pl: Platform, mount: str, r: float, yaw: float = 0.0, *, stabilizers_deployed: bool = False,
                 payload_kg: float = 0.0, others: str = "folded") -> float:
    """The load [kg] in ``mount``'s gripper at reach ``r`` [m] and heading ``yaw`` [rad] that brings the CG to the
    edge of the support polygon (the other arms in pose ``others``); inf when it never tips within 10 t."""
    m = pl.mount(mount)
    q = _reach_pose(pl, m, r, yaw)
    if q is None:
        return 0.0
    arm_q = {**poses(pl, others), mount: q}
    kw = {"stabilizers_deployed": stabilizers_deployed, "payload_kg": payload_kg}
    if stability(pl, arm_q, loads={mount: 0.0}, **kw)["margin_m"] <= 0:
        return 0.0
    lo, hi = 0.0, 10000.0
    if stability(pl, arm_q, loads={mount: hi}, **kw)["margin_m"] > 0:
        return math.inf
    for _ in range(60):
        mid = (lo + hi) / 2
        if stability(pl, arm_q, loads={mount: mid}, **kw)["margin_m"] > 0:
            lo = mid
        else:
            hi = mid
    return lo


def lift_capacity(pl: Platform, mount: str, r: float, yaw: float = 0.0, *, stabilizers_deployed: bool = False,
                  payload_kg: float = 0.0) -> dict:
    """What ``mount``'s arm may lift at reach ``r`` and heading ``yaw``: the arm's structural rating
    (``pekari_arms.rated_load``, peak), ``TIP_RATIO`` × the tipping load, and the smaller, with what limits it."""
    m = pl.mount(mount)
    arm = pa.rated_load(m.arm, r)
    tip = tipping_load(pl, mount, r, yaw, stabilizers_deployed=stabilizers_deployed, payload_kg=payload_kg)
    stab = TIP_RATIO * tip
    return {"arm_kg": arm, "tipping_kg": tip, "stability_kg": stab, "capacity_kg": min(arm, stab),
            "limit": "arm" if arm <= stab else "stability"}


# ----------------------------------------------------------------------------------------------- drive and power
def drive_check(pl: Platform, *, payload_kg: float | None = None, soil="gravel", grade_deg: float = 0.0,
                v_ms: float = 0.5) -> dict:
    """The track drives against ``tracks.resistance`` on ``soil`` at ``grade_deg`` and ``v_ms`` with the deck
    loaded (default the nominal payload): the force at the sprockets (stall line at v, continuous), traction,
    the top speed on the flat (no-load) and pass / fail."""
    payload = pl.payload_kg if payload_kg is None else payload_kg
    g = geometry(pl)
    W = total_mass(pl, payload) * G
    d = act.get(pl.drive)
    r = g["r_sprocket"]
    w0 = d.no_load_rpm * 2 * math.pi / 60
    F_stall = 2 * d.stall_Nm / r * max(0.0, 1 - (v_ms / r) / w0)
    F_rated = 2 * d.rated_Nm / r
    res = tracks.resistance(W, b=g["b"], L=g["L"], soil=soil, grade_deg=grade_deg, v=v_ms, f0=pl.internal_f0)
    F_trac = tracks.tractive_limit(W, b=g["b"], L=g["L"], soil=soil, grade_deg=grade_deg)
    return {"W_N": W, "resistance_N": res["total"], "drive_stall_N": F_stall, "drive_rated_N": F_rated,
            "traction_N": F_trac, "v_top_flat_m_s": w0 * r, "ok_rated": F_rated >= res["total"],
            "ok_peak": F_stall >= res["total"], "ok_traction": F_trac >= res["total"],
            "ground_pressure_kPa": tracks.ground_pressure(W, g["b"], g["L"]) / 1000.0}


#: The mission profile ``power`` averages: fractions of the time driving at ``v``, manipulating, idle (sensing).
PROFILE = {"drive": 0.5, "manipulate": 0.3, "idle": 0.2, "v": 0.8, "soil": "gravel", "grade_deg": 0.0}
DRIVE_EFF = 0.80                       # battery to sprocket: controller, motor and gearbox at part load (assumed)


def power(pl: Platform, profile: dict | None = None, payload_kg: float | None = None) -> dict:
    """Average electrical power [W] of a mission ``profile`` (default ``PROFILE``): traction (the resistance at v
    over ``DRIVE_EFF``) while driving, the arms (``pekari_arms.arm_power`` at duty 1 while manipulating, both arms),
    the electronics always."""
    pr = {**PROFILE, **(profile or {})}
    v = min(pr["v"], pl.v_design)
    dc = drive_check(pl, payload_kg=payload_kg, soil=pr["soil"], grade_deg=pr["grade_deg"], v_ms=v)
    p_drive = dc["resistance_N"] * v / DRIVE_EFF
    p_arms = sum(pa.arm_power(m.arm, duty=1.0) for m in pl.mounts)
    avg = pr["drive"] * p_drive + pr["manipulate"] * p_arms + pl.electronics_W
    return {"traction_W": p_drive, "arms_W": p_arms, "electronics_W": pl.electronics_W, "average_W": avg,
            "profile": pr}


def endurance(pl: Platform, profile: dict | None = None, usable: float = 0.85, **kw) -> float:
    """Hours on the battery (``usable`` fraction of the nominal energy) for ``profile``."""
    return pl.battery_Wh * usable / power(pl, profile, **kw)["average_W"]


# ----------------------------------------------------------------------------------------------- fleet view
def capabilities(pl: Platform) -> dict:
    """What the fleet coordinator allocates by (SI): the platform's envelope, payload, arms, speed, endurance,
    sensors and capability tags."""
    g = geometry(pl)
    ov = g["overall"]
    arms = {m.name: {"reach_m": pa.reach(m.arm), "rated_kg": m.arm.rated_payload_kg, "rated_reach_m": m.arm.rated_reach,
                     "dof": pa.arm_dof(m.arm)} for m in pl.mounts}
    h_folded = max([ov["height"]] + [max(z for *_, z in pa.mass_points(m.arm, pa.pose(m.arm, "folded"), shoulder(pl, m)))
                                      + m.arm.upper_section[1] for m in pl.mounts])
    dc = drive_check(pl)
    best = max(pl.mounts, key=lambda m: m.arm.rated_payload_kg)
    tags = {"ground", "tracked", "inspect", "map", "manipulate", "transport"}
    if len(pl.mounts) >= 2:
        tags.add("dual_arm")
    if best.arm.rated_payload_kg >= 30:
        tags.add("heavy_lift")
    if pl.payload_range_kg[1] >= 100:
        tags.add("heavy_transport")
    if ov["width"] <= 0.70:
        tags.add("confined")
    if any("thermal" in s for s in pl.sensors):
        tags.add("thermal")
    return {"name": pl.name, "mass_kg": total_mass(pl), "length_m": ov["length"], "width_m": ov["width"],
            "height_folded_m": h_folded, "payload_kg": pl.payload_range_kg[1], "arm_lift_kg": best.arm.rated_payload_kg,
            "arms": arms, "v_max_m_s": min(pl.v_design, dc["v_top_flat_m_s"]), "endurance_h": endurance(pl),
            "sensors": tuple(pl.sensors), "tags": frozenset(tags)}


def summary(pl: Platform) -> dict:
    """The headline numbers (for the docs and the tests)."""
    c_empty = cg(pl)
    c_load = cg(pl, payload_kg=pl.payload_kg)
    st = stability(pl, payload_kg=pl.payload_kg)
    cap = capabilities(pl)
    dc20 = drive_check(pl, grade_deg=20.0, v_ms=0.3)
    return {"mass_empty_kg": c_empty["mass_kg"], "mass_loaded_kg": c_load["mass_kg"],
            "cg_empty": (c_empty["x"], c_empty["y"], c_empty["z"]), "cg_loaded": (c_load["x"], c_load["y"], c_load["z"]),
            "overall_m": (cap["length_m"], cap["width_m"], cap["height_folded_m"]), "margin_folded_m": st["margin_m"],
            "v_max_m_s": cap["v_max_m_s"], "endurance_h": cap["endurance_h"], "grade20_ok": dc20["ok_peak"] and dc20["ok_traction"],
            "average_W": power(pl)["average_W"]}


# ----------------------------------------------------------------------------------------------- Chiron
def _stabilizer_link(pl: Platform, s: Stabilizer, rgba):
    """A stabilizer leg as a vertical slide (+ = down) under the hull with the pad at its foot; the leg's beam to
    the hull side is drawn (visual)."""
    from vegeta.chiron import Geom, Joint, Link, Servo

    a = act.get_linear(s.drive)
    zh = z_hull(pl)
    z_pad_stowed = s.pad_radius / 2 + s.stroke * 0.6           # the pad's centre above the ground, stowed
    side = 1.0 if s.y > 0 else -1.0
    half_w = pl.chassis["hull_width"] / 2 * 1e-3
    servo = Servo(stall_torque=a.stall_N, rated_torque=a.rated_N, no_load_speed=a.no_load_mm_s * 1e-3,
                  stall_current=a.stall_A, voltage=a.voltage_V, kp=1.0e5, kd=4.0e3, armature=5.0,
                  four_quadrant=a.self_locking, source=f"{s.drive} ({a.source})")
    beam_len = abs(s.y) - half_w
    return Link(f"stab_{s.name}", pos=(s.x, s.y, z_pad_stowed - zh),
                joints=[Joint(f"stab_{s.name}", kind="slide", axis=(0, 0, -1), range=(0.0, s.stroke), tag="stabilizer",
                              servo=servo)],
                geoms=[Geom(f"stab_{s.name}_pad", "cylinder", (s.pad_radius, s.pad_radius / 4), mass=s.mass_kg * 0.3,
                            role="link", friction=(0.8, 0.01, 0.0001), rgba=(0.15, 0.15, 0.15, 1.0)),
                       Geom(f"stab_{s.name}_leg", "box", (0.03, 0.03, 0.12), pos=(0.0, 0.0, 0.12), mass=s.mass_kg * 0.3,
                            role="visual", rgba=rgba),
                       Geom(f"stab_{s.name}_beam", "box", (0.04, beam_len / 2, 0.03),
                            pos=(0.0, -side * beam_len / 2, 0.24), mass=s.mass_kg * 0.4, role="visual", rgba=rgba)])


def stabilizer_ground_q(pl: Platform, s: Stabilizer, preload_m: float = 0.003) -> float:
    """The slide position that puts the pad on the ground plus ``preload_m`` (with the servo's kp 1e5 N/m: ~300 N a
    pad standing; a pad takes more as the machine leans onto it)."""
    z_pad_stowed = s.pad_radius / 2 + s.stroke * 0.6
    return min(s.stroke, z_pad_stowed - s.pad_radius / 4 + preload_m)


def qpos(pl: Platform, pose_name: str = "folded", grip: float = 0.0, stabilizers_deployed: bool = False) -> dict:
    """Chiron joint positions of the arms in ``pose_name`` and the stabilizers (stowed or on the ground)."""
    out = {}
    for m in pl.mounts:
        out.update(pa.qpos(m.arm, f"{m.name}_", pa.pose(m.arm, pose_name), grip))
    for s in pl.stabilizers:
        out[f"stab_{s.name}"] = stabilizer_ground_q(pl, s) if stabilizers_deployed else 0.0
    return out


def robot(pl: Platform, *, payload_kg: float | None = None, pose_name: str = "folded",
          stabilizers_deployed: bool = False, **chassis_kw):
    """The platform as a Chiron ``Robot``: ``prr.pekari`` with the platform's chassis, its arms (``pekari_arms
    .links``) and stabilizers (vertical slides) under the hull; nominal pose ``pose_name``. Joint names: the
    Pekari rollers and bogies, ``{mount}_yaw`` … ``{mount}_grip_a/b``, ``stab_{name}``."""
    payload = pl.payload_kg if payload_kg is None else payload_kg
    zh = z_hull(pl)
    rgba = pl.colors.get("arm", (0.95, 0.55, 0.05, 1.0))
    children, hull_masses, m_extra = [], [], 0.0
    for m in pl.mounts:
        sh = shoulder(pl, m)
        base, yaw_mass, moving = pa.links(m.arm, f"{m.name}_", (sh[0], sh[1], sh[2] - zh), rgba=rgba)
        children.append(base)
        hull_masses.append(yaw_mass)
        m_extra += moving + yaw_mass.mass
    for s in pl.stabilizers:
        children.append(_stabilizer_link(pl, s, pl.colors.get("stabilizer", rgba)))
        m_extra += s.mass_kg
    _, mom = _extras_folded(pl)
    # the rollers' velocity loop: Pekari's gain grown with the mass — with Pekari's 0.5 N m s/rad a 600 kg Pumba
    # lags its belt speed by ~0.4 m/s; in full proportion it overshoots 12 % on a 1 m/s² start; half of it tracks
    # within 3 % (tuned by the drive test, tests/test_pekari_family.py)
    chassis_kw.setdefault("roller_kd", 0.5 * prr.ROLLER_KD * total_mass(pl, payload) / PEKARI_LOADED_KG)
    r = prr.pekari(params=chassis_params(pl), payload_kg=payload, cad=pl.cad, parts_kg=pl.parts_kg, drive=pl.drive,
                   budget=chassis_budget(pl),
                   shell_t_mm=pl.shell_t_mm, materials=pl.materials, wheel_fill=pl.wheel_fill, n_bogies=pl.n_bogies,
                   name=pl.name, extra_children=children, extra_hull_masses=hull_masses, extra_mass_kg=m_extra,
                   extra_moment_kgm=mom, payload_pos=payload_point(pl), track_mu=pl.track_mu,
                   internal_f0=pl.internal_f0,
                   notes=f"{pl.name}: the Pekari track architecture ({pl.chassis['track_width']:g} mm tracks, "
                         f"{pl.chassis['road_wheels']} road wheels a side) with "
                         + ", ".join(f"{m.name} = {m.arm.name} ({pa.arm_dof(m.arm)} DOF)" for m in pl.mounts)
                         + (f" and {len(pl.stabilizers)} stabilizers" if pl.stabilizers else "")
                         + f"; {payload:g} kg on the deck",
                   sources={"geometry": "designs/pekari_family.py (chassis: pekari_rover.PekariRover)",
                            "masses": "pekari_family.masses", "arms": "designs/pekari_arms.py",
                            "drives": "ogma.actuators", "assumptions": f"{pl.name} Platform.assumptions"},
                   **chassis_kw)
    r.nominal_qpos.update(qpos(pl, pose_name, stabilizers_deployed=stabilizers_deployed))
    r.validate()
    r.platform = pl
    return r


def lab(pl: Platform, terrain=None, robot_=None, **kwargs):
    """A ChironLab for the platform (1 ms steps, 2 ms control, 20 ms log; ``kwargs`` override)."""
    from vegeta.chiron import ChironLab

    opts = {"timestep": 0.001, "control_dt": 0.002, "log_dt": 0.02}
    opts.update(kwargs)
    return ChironLab(robot_ or robot(pl), terrain, **opts)


def with_arm_pose(pl: Platform, mount: str, pose_name: str, deg: dict) -> Platform:
    """A copy of ``pl`` whose arm ``mount`` has an extra named pose (``deg``: {joint: deg})."""
    ms = []
    for m in pl.mounts:
        if m.name == mount:
            m = replace(m, arm=replace(m.arm, poses={**m.arm.poses, pose_name: deg}))
        ms.append(m)
    return replace(pl, mounts=tuple(ms))


def stand_test(pl: Platform, pose_name: str = "folded", *, stabilizers_deployed: bool = False, payload_kg: float | None = None,
               loads: dict | None = None, duration: float = 1.5, terrain=None) -> dict:
    """Stand the Chiron model still for ``duration`` s with the arms holding ``pose_name`` (and ``loads`` {mount:
    kg} as point masses in the grippers' palms — added to the robot before the run): the hull's tilt [deg], its
    height change [m], the share of the weight on the tracks (the rest on the stabilizer pads), the largest arm
    joint error [deg] and whether the hull touched the ground. Needs MuJoCo."""
    import numpy as np
    from vegeta import chiron as ch
    from vegeta.chiron import PointMass

    r = robot(pl, payload_kg=payload_kg, pose_name=pose_name, stabilizers_deployed=stabilizers_deployed)
    for name, kg in (loads or {}).items():
        hand = next(lk for lk in r.links() if lk.name == f"{name}_hand")
        hand.masses.append(PointMass(f"{name}_load", kg, (pl.mount(name).arm.tool * 0.65, 0.0, 0.0)))
    lb = lab(pl, terrain or ch.Flat(), r)
    obs = lb.reset(seed=0)
    target = dict(r.nominal_qpos)
    cmd = ch.Command(q_target={j: v for j, v in target.items() if j in lb.actuated_joints})
    z0 = None
    n = int(round(duration / lb.control_dt))
    for i in range(n):
        obs = lb.step(cmd)
        if z0 is None:
            z0 = float(obs.base_pos[2])
    w, x, y, z = (float(v) for v in obs.base_quat)
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (x * x + y * y)))))
    fz = float(np.asarray(obs.foot_force)[:, 2].sum())
    M = r.total_mass()
    arm_err = 0.0
    for m in pl.mounts:
        for j in pa.joint_names(m.arm, f"{m.name}_"):
            if j.endswith("grip"):
                continue
            arm_err = max(arm_err, abs(obs.joint(j) - target[j]))
    return {"mass_kg": M, "tilt_deg": tilt, "dz_m": float(obs.base_pos[2]) - z0, "track_share": fz / (M * G),
            "arm_err_deg": math.degrees(arm_err), "belly": bool(np.asarray(obs.belly_contact).any()),
            "base_pos": tuple(float(v) for v in obs.base_pos)}


# ----------------------------------------------------------------------------------------------- BOM and cost
#: Class prices of the catalogue drives [EUR, (low, high)], 2026, one-off quantities: estimates for budgeting, not
#: quotes (no supplier was asked); the drives' own entries are class values too (ogma.actuators).
PRICE_EUR = {
    "perseus mini drive 20:1 planetary": (250.0, 600.0),
    "calydon track drive 4 kW, 48 V": (2500.0, 5000.0),
    "pumba track drive 2.5 kW, 48 V": (1800.0, 3800.0),
    "stick cylinder 8 kN": (1200.0, 2500.0),
    "harmonic 12 Nm, brake": (600.0, 1400.0),
    "harmonic 30 Nm, brake": (900.0, 2000.0),
    "harmonic 60 Nm, brake": (1500.0, 3000.0),
    "harmonic 150 Nm, brake": (2000.0, 4000.0),
    "cycloidal 400 Nm, brake": (3000.0, 6000.0),
    "cycloidal 800 Nm, brake": (5000.0, 9000.0),
    "cycloidal 1600 Nm, brake": (8000.0, 15000.0),
    "smart servo 12 Nm": (100.0, 250.0),
    "smart servo 25 Nm": (200.0, 450.0),
    "jaw screw 6 kN": (900.0, 2000.0),
    "boom cylinder 12 kN": (1500.0, 3000.0),
}


def bom(pl: Platform) -> list:
    """The bill of materials [(item, qty, EUR low, EUR high)] per unit: the track drives, every arm drive and
    stabilizer drive from the catalogue (``PRICE_EUR``), and the platform's ``bom_extra`` (chassis, battery,
    computers, sensors, grippers, wiring). Prices are class estimates; assembly labour is not included."""
    count: dict = {pl.drive: 2}
    for m in pl.mounts:
        for key in m.arm.drives.values():
            count[key] = count.get(key, 0) + 1
    for st in pl.stabilizers:
        count[st.drive] = count.get(st.drive, 0) + 1
    rows = [(f"drive: {k}", n, *PRICE_EUR[k]) for k, n in count.items()]
    rows += [(k, q, lo, hi) for k, (q, lo, hi) in pl.bom_extra.items()]
    return rows


def bom_total(pl: Platform) -> tuple:
    """(EUR low, EUR high) of one unit's parts."""
    rows = bom(pl)
    return sum(q * lo for _, q, lo, _ in rows), sum(q * hi for _, q, _, hi in rows)
