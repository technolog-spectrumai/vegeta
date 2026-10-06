"""Drongo (notebook 08b) as a Chiron robot: notebook 08's quadcopter with a landing gear of two skids and a pincer,
and what it carries — a potato and a cup of cream.

* **the airframe** — ``QuadFrame`` from notebook 08 (250 mm X-frame, PETG-CF), its parts list (2306 motors, 5x4.3
  three-blade propellers, 4S 1500 mAh), plus the skids and the pincer of ``drongo.Drongo`` (CAD volumes in ``CAD``).
  The root link is the frame: centre plate, arms, battery and gripper housing are shells (``role='body'``: touching
  the ground is a crash), the skids are its feet (``skid_L``, ``skid_R``), motors and propeller discs are drawn and
  weigh but do not collide;
* **the pincer** — a parallel gripper: one hobby servo turns a pinion between two racks, the jaws slide along x on a
  rail (``jaw_front``, ``jaw_rear``: slide joints, + opens). Each jaw is a Chiron ``Servo`` on its slide with the force
  the pinion gives one rack, ``τ / (2 r)`` (both racks share the servo's torque when both press on an item): 23 N at
  stall — below the cream cup's crushing load, so even a servo stuck at stall cannot crush the cream;
* **the rotors** — ``Rotors``, a scene hook (``ChironLab.add_hook``): each rotor's thrust follows its command with a
  first-order lag and stays between 0 and the maximum thrust; the four thrusts along the frame's z axis, their
  moments about the frame's centre of mass and the propellers' drag torques go onto the frame with
  ``lab.body_force``, with the airframe's drag; the electrical power comes from the propulsion (``Propulsion``) and
  is integrated into the energy used;
* **the propulsion** — ``propulsion()``: notebook 08's Boreas export (``_runs/quadcopter/propeller/quad_5x43.json``:
  the operating points hover / cruise / full, thrust and electrical power) when notebook 08 has been run; otherwise the
  assumptions in ``ASSUMED_PROPULSION`` (nothing is solved here);
* **the items** — ``POTATO`` (a 200 g sphere) and ``CREAM`` (a 400 g cup of 18 % cream as a cylinder), with the
  loads they take before they are spoiled (``squeeze_limit_N``, ``impact_limit_m_s``, ``catch_limit_N``), and the
  ``NET`` the hungry people hold.

Units: SI (m, kg, s, N). x forward (between two arms), y left, z up; the root frame's origin is the frame's underside
at the centre plate (z = 0 of the CAD), ``skid_height`` above the ground when landed.

    import drongo_robot as dr
    robot = dr.drongo()                       # 0.59 kg
    prop = dr.propulsion()                    # notebook 08's export, or the assumptions
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from vegeta.chiron import FootSpec, Geom, Joint, Link, PointMass, Robot, Servo

import actuators as act

__all__ = ["DRONGO", "CAD", "PARTS_G", "GRIPPER_PARTS_G", "MATERIALS", "FILL", "JAW_ACTUATOR", "PINION_RADIUS_M",
           "JAW_GAINS", "ASSUMPTIONS", "POTATO", "CREAM", "ITEMS", "NET", "GRIP_N", "G", "RHO_AIR", "DRAG_AREA_M2",
           "QUAD_EXPORT", "ASSUMED_PROPULSION", "Propulsion", "propulsion", "design_params", "cad_numbers", "geometry",
           "mass_budget", "item_geometry", "grip_needed", "grip_holds", "net_catch", "jaw_servo", "rotor_layout",
           "drongo", "LAB_OPTIONS", "Rotors", "JAWS", "PADS", "JAW_JOINTS", "pad_gap"]

G = 9.81
RHO_AIR = 1.2

#: Drongo's parameters beyond notebook 08's QuadFrame (``drongo.Drongo`` defaults; mm). The frame's own are its defaults.
DRONGO = {
    "wheelbase": 250.0, "arm_width": 12.0, "arm_height": 6.0, "taper": 0.7, "plate_size": 70.0, "plate_thickness": 6.0,
    "pad_diameter": 30.0,
    "skid_height": 120.0, "skid_y": 75.0, "skid_length": 180.0, "skid_diameter": 8.0, "strut_x": 50.0,
    "strut_y_top": 22.0, "strut_diameter": 6.0, "housing_x": 44.0, "housing_y": 34.0, "housing_z": 16.0,
    "rail_length": 190.0, "rail_width": 12.0, "rail_height": 8.0, "carriage_x": 16.0, "carriage_y": 20.0,
    "carriage_z": 8.0, "finger_thickness": 5.0, "finger_width": 16.0, "pad_thickness": 6.0, "pad_width": 40.0,
    "pad_height": 30.0, "grip_height": 36.5, "closed_gap": 30.0, "jaw_travel": 60.0, "jaw_opening": 40.0,
    "max_item_height": 75.0,
}

#: CAD volumes of the default Drongo [mm³] (``drongo.Drongo().generate(part=...).measure()``); ``cad_numbers(recompute=
#: True)`` rebuilds them. ``frame`` is notebook 08's QuadFrame; ``gear`` both skids with their struts; ``gripper`` the
#: servo housing and the rail (solid); ``jaw`` one carriage and finger; ``pad`` one TPU pad.
CAD = {"frame": 57016.4, "gear": 32101.5, "gripper": 42176.0, "jaw": 7880.0, "pad": 7200.0}
MATERIALS = {"PETG-CF": 1.25e-3, "TPU 95A": 1.21e-3}              # g/mm³ (notebook 08: PETG-CF 1.25 g/cm³)
#: Share of the solid CAD volume that is printed: the frame solid (as in notebook 08), the skid tubes and struts with
#: thin walls, the servo housing a 2 mm shell around the servo.
FILL = {"frame": 1.0, "gear": 0.6, "gripper": 0.35, "jaw": 1.0, "pad": 1.0}

#: Notebook 08 §1: the quad's bought parts [g].
PARTS_G = {"motors 2306 (4x)": 120.0, "propellers 5x4.3 (4x)": 20.0, "4-in-1 ESC": 15.0, "flight controller": 10.0,
           "battery 4S 1500 mAh": 180.0, "camera + VTX + antenna": 35.0, "receiver, wiring, bolts": 30.0}
JAW_ACTUATOR = "micro-metal 20 g"                                # actuators.py: 0.55 N·m stall, 90 rpm, 7.4 V
GRIPPER_PARTS_G = {"pinion, racks, rail screws": 6.0}
PINION_RADIUS_M = 0.012
#: Jaw slide servo: kp [N/m], kd [N·s/m], reflected rotor inertia at the rack [kg] (ASSUMPTIONS).
JAW_GAINS = (1500.0, 40.0, 0.5)
#: Airframe drag area C_d·A [m²] with an item in the pincer (frame, battery, gear, pincer, item; assumed).
DRAG_AREA_M2 = 0.015

ASSUMPTIONS = {
    "frame": "notebook 08's QuadFrame at its default parameters (the baseline revision); notebook 08's preferred revision "
             "can be passed as overrides",
    "jaw_drive": "one 20 g metal-gear servo turns a 12 mm pinion between two racks: each rack gets τ/(2r) when both press "
                 "on an item (23 N at stall); modelled as two slides, each with that force (the racks are not coupled "
                 "here: each jaw closes on its own until it meets the item)",
    "jaw_gains": "kp 1500 N/m (a squeeze of F needs a position target F/kp inside the item: 8 mm for 12 N), kd 40 N·s/m "
                 "for ζ ≈ 0.7 with 0.5 kg reflected at the rack (a ~0.05 g·cm² rotor through ~250:1 and the pinion)",
    "rotors": "thrust along the frame's z axis, first-order lag (motor and propeller spin-up), 0 ≤ T ≤ T_max per rotor; "
              "no ground effect, no blade flapping, no gyroscopic moments; propeller drag torque = k_q T",
    "drag": f"C_d·A = {DRAG_AREA_M2} m² on the whole airframe with its item, at the frame's centre of mass",
    "sensing": "the flight controller knows the frame's position, velocity, attitude and rates exactly (ideal GNSS-RTK "
               "and IMU) and where the items, the net and the drop zone are",
    "masses": "bought parts from notebook 08; printed parts from the CAD volumes × density × FILL; the servo from "
              "actuators.py",
}

# ----------------------------------------------------------------------------------------------- what Drongo carries
#: The potato: a sphere of 200 g (the user's request). Its numbers are the usual ones for a table potato (pl.wikipedia
#: "Ziemniak": a tuber of mostly water and starch); the limits are assumptions stated here.
POTATO = {
    "name": "potato", "shape": "sphere", "mass_kg": 0.200,
    "density_kg_m3": 1080.0,            # a tuber is ~78 % water and ~18 % starch: 1.06–1.10 g/cm³ (it sinks in water)
    "mu": 0.6,                          # moist skin on the TPU pads (also on the grass)
    "torsional_m": 0.01, "rolling_m": 0.003,   # a lumpy tuber: it does not roll away like a ball
    "squeeze_limit_N": 60.0,            # bruising under a two-point squeeze (whole tubers split at ~150–250 N)
    "impact_limit_m_s": 1.7,            # blackspot bruising: drops above ~15 cm onto a hard surface (v = √(2 g 0.15))
    "catch_limit_N": 60.0,              # the net's arrest force on it (spread over the skin; the squeeze limit kept)
    "rgba": (0.72, 0.55, 0.30, 1.0),
}
#: The cream: a 400 g cup of 18 % cream (the user's product page: a 400 g cup), as a cylinder. The cup is a typical
#: 400 g polypropylene cup with a foil lid — check the diameter and height against the real one.
CREAM = {
    "name": "cream", "shape": "cylinder", "content_kg": 0.400, "cup_kg": 0.012,
    "density_kg_m3": 1010.0,            # 18 % cream
    "diameter_m": 0.092, "height_m": 0.074,
    "mu": 0.5,                          # polypropylene on TPU
    "torsional_m": 0.02, "rolling_m": 0.0,
    "squeeze_limit_N": 25.0,            # the thin PP wall buckles and the foil seal lifts under a two-pad side squeeze
    "impact_limit_m_s": 2.4,            # ~30 cm onto a hard floor without the seal bursting
    "catch_limit_N": 100.0,             # in a soft net the load spreads over the cup: ~10 kPa on the lid (seal ~40 kPa)
    "rgba": (0.97, 0.97, 0.94, 1.0),
}
CREAM["mass_kg"] = CREAM["content_kg"] + CREAM["cup_kg"]
POTATO["diameter_m"] = 2 * (3 * POTATO["mass_kg"] / (4 * math.pi * POTATO["density_kg_m3"])) ** (1 / 3)
POTATO["height_m"] = POTATO["diameter_m"]
ITEMS = {"potato": POTATO, "cream": CREAM}
#: The net the hungry people hold over the drop zone (not simulated: a catch at its plane, see the scenario).
NET = {"height_m": 1.0,                # held at the people's waist
       "size_m": 2.0,                   # a 2 m × 2 m square
       "max_drop_m": 5.0,               # the user's rule: an item falling more than 5 m onto it tears it
       "stretch_m": 0.4}                # how far it gives when it catches (assumed): the arrest distance
#: Squeeze the pincer holds each item with [N per pad] (notebook 08b §3: the grip window).
GRIP_N = {"potato": 6.0, "cream": 12.0}


# ----------------------------------------------------------------------------------------------- propulsion
QUAD_EXPORT = Path(__file__).resolve().parents[1] / "_runs" / "quadcopter" / "propeller" / "quad_5x43.json"

#: Used when notebook 08's export is absent. Nothing is computed from these: a 5-inch 2306/2400KV/4S class, as
#: test-stand tables give it.
ASSUMED_PROPULSION = {
    "max_thrust_N": 8.0,                 # notebook 08's MAX_THRUST_PER_MOTOR_N: ~815 gf at full throttle on 4S
    "spin_up_s": 0.035,                  # thrust time constant of a 5-inch rotor (motor + propeller)
    "torque_per_thrust_m": 0.011,        # propeller drag torque / thrust (C_P / (2π C_T) · D for a 5x4.3 three-blade)
    "efficiency_g_per_W": ((100.0, 7.0), (200.0, 5.0), (300.0, 4.2), (500.0, 3.2), (815.0, 2.4)),   # thrust [g], g/W
    "battery_Wh": 4 * 3.7 * 1.5,         # 4S 1500 mAh
    "usable_fraction": 0.8,
    "source": "assumed: a 5-inch 2306 2400KV 4S test-stand class; notebook 08's maximum thrust",
}


@dataclass
class Propulsion:
    """One rotor's numbers for the flight simulation: maximum thrust [N], spin-up time constant [s], drag torque per
    thrust [m], and the electrical power [W] at a thrust as a table (``thrust_N``, ``power_W``; log-log
    interpolation, extended with its end slopes). ``battery_Wh`` × ``usable_fraction`` is the energy available."""

    max_thrust_N: float
    spin_up_s: float
    torque_per_thrust_m: float
    thrust_N: np.ndarray
    power_W: np.ndarray
    battery_Wh: float
    usable_fraction: float
    source: str
    points: dict = field(default_factory=dict)

    def electrical_power(self, thrust):
        """Electrical power [W] of one rotor at ``thrust`` [N] (array or scalar); 0 at no thrust."""
        T = np.asarray(thrust, dtype=float)
        lt, lp = np.log(self.thrust_N), np.log(self.power_W)
        x = np.log(np.maximum(T, 1e-6))
        lo = lp[0] + (x - lt[0]) * (lp[1] - lp[0]) / (lt[1] - lt[0])
        hi = lp[-1] + (x - lt[-1]) * (lp[-1] - lp[-2]) / (lt[-1] - lt[-2])
        y = np.where(x < lt[0], lo, np.where(x > lt[-1], hi, np.interp(x, lt, lp)))
        return np.where(T > 0, np.exp(y), 0.0)

    @property
    def usable_Wh(self) -> float:
        return self.battery_Wh * self.usable_fraction

    def table(self) -> dict:
        return {"max thrust per rotor [N]": self.max_thrust_N, "spin-up time constant [s]": self.spin_up_s,
                "drag torque / thrust [m]": self.torque_per_thrust_m, "battery [Wh]": self.battery_Wh,
                "usable [Wh]": self.usable_Wh, "source": self.source}


def propulsion(path=QUAD_EXPORT) -> Propulsion:
    """The rotors from notebook 08's Boreas export (``path``) when it exists, else ``ASSUMED_PROPULSION``."""
    a = ASSUMED_PROPULSION
    path = None if path is None else Path(path)
    if path is not None and path.exists():
        doc = json.loads(path.read_text())
        pts = doc.get("points") or {}
        rows = sorted((float(r["aero"]["thrust"]), float(r["electrical_power"])) for r in pts.values()
                      if r.get("aero") and r.get("electrical_power"))
        if len(rows) >= 2 and "full" in pts:
            hover = pts.get("hover") or next(iter(pts.values()))
            kq = float(hover["aero"]["torque"]) / float(hover["aero"]["thrust"])
            bat = doc.get("battery") or {}
            wh = (bat.get("cells", 4) * 3.7 * bat.get("capacity_ah", 1.5)) if bat else a["battery_Wh"]
            return Propulsion(float(pts["full"]["aero"]["thrust"]), a["spin_up_s"], kq, np.array([r[0] for r in rows]),
                              np.array([r[1] for r in rows]), wh, float(bat.get("usable_fraction", a["usable_fraction"])),
                              f"notebook 08's Boreas export ({path.name}: {', '.join(pts)}); spin-up assumed",
                              {k: {"thrust_N": float(v["aero"]["thrust"]), "electrical_W": float(v["electrical_power"]),
                                   "rpm": float(v["rpm"])} for k, v in pts.items()})
    tg = np.array([t for t, _ in a["efficiency_g_per_W"]])
    eff = np.array([e for _, e in a["efficiency_g_per_W"]])
    return Propulsion(a["max_thrust_N"], a["spin_up_s"], a["torque_per_thrust_m"], tg * G / 1000.0, tg / eff,
                      a["battery_Wh"], a["usable_fraction"], a["source"])


# ----------------------------------------------------------------------------------------------- design numbers
def design_params(overrides: dict | None = None) -> dict:
    """Drongo's parameters [mm] with ``overrides``."""
    p = dict(DRONGO)
    p.update({k: float(v) for k, v in (overrides or {}).items()})
    return p


def cad_numbers(p: dict | None = None, *, recompute: bool = False) -> dict:
    """The CAD volumes [mm³]: the stored ``CAD`` for the default design, else rebuilt with ``drongo.Drongo``."""
    p = design_params() if p is None else dict(p)
    if not recompute and p == design_params():
        return dict(CAD)
    import drongo

    d = drongo.Drongo()
    over = {k: v for k, v in p.items() if k in {prm.name for prm in d.parameters}}
    return {part: d.generate(part=part, **over).measure()["volume"] for part in CAD}


def geometry(p: dict | None = None) -> dict:
    """Drongo's geometry in metres (frame origin: the frame's underside at the centre)."""
    p = design_params() if p is None else p
    mm = 1e-3
    return {"R": p["wheelbase"] / 2 * mm, "plate_r": p["plate_size"] / 2 * mm, "plate_t": p["plate_thickness"] * mm,
            "arm_h": p["arm_height"] * mm, "arm_w": p["arm_width"] * (1 + p["taper"]) / 2 * mm,
            "skid_h": p["skid_height"] * mm, "skid_y": p["skid_y"] * mm, "skid_L": p["skid_length"] * mm,
            "skid_r": p["skid_diameter"] / 2 * mm, "strut_x": p["strut_x"] * mm, "strut_y_top": p["strut_y_top"] * mm,
            "strut_r": p["strut_diameter"] / 2 * mm, "housing": (p["housing_x"] * mm, p["housing_y"] * mm, p["housing_z"] * mm),
            "rail": (p["rail_length"] * mm, p["rail_width"] * mm, p["rail_height"] * mm),
            "carriage": (p["carriage_x"] * mm, p["carriage_y"] * mm, p["carriage_z"] * mm),
            "finger": (p["finger_thickness"] * mm, p["finger_width"] * mm),
            "pad": (p["pad_thickness"] * mm, p["pad_width"] * mm, p["pad_height"] * mm),
            "pad_z": (p["grip_height"] - p["skid_height"]) * mm, "grip_height": p["grip_height"] * mm,
            "closed_gap": p["closed_gap"] * mm, "travel": p["jaw_travel"] * mm, "ready": p["jaw_opening"] * mm,
            "carriage_z": -(p["housing_z"] + p["rail_height"] + p["carriage_z"]) * mm,
            "prop_r": 5 * 25.4 / 2 * mm, "motor_r": 0.014, "motor_h": 0.016}


def mass_budget(p: dict | None = None, cad: dict | None = None) -> dict:
    """Every part [g]: notebook 08's bought parts, the printed parts (CAD volume × density × FILL) and the pincer."""
    cad = cad_numbers(p) if cad is None else cad
    out = dict(PARTS_G)
    out["frame (QuadFrame, PETG-CF, CAD)"] = cad["frame"] * MATERIALS["PETG-CF"] * FILL["frame"]
    out["landing gear (PETG-CF, CAD)"] = cad["gear"] * MATERIALS["PETG-CF"] * FILL["gear"]
    out["servo housing and rail (PETG-CF, CAD)"] = cad["gripper"] * MATERIALS["PETG-CF"] * FILL["gripper"]
    out[f"jaw servo ({JAW_ACTUATOR})"] = act.get(JAW_ACTUATOR).mass_g
    out.update(GRIPPER_PARTS_G)
    out["jaws 2x (PETG-CF, CAD)"] = 2 * cad["jaw"] * MATERIALS["PETG-CF"] * FILL["jaw"]
    out["pads 2x (TPU, CAD)"] = 2 * cad["pad"] * MATERIALS["TPU 95A"] * FILL["pad"]
    return out


def item_geometry(item: dict) -> dict:
    """Radius, half-height and the gap between the pads [m] when the jaws close on the item's middle."""
    r = item["diameter_m"] / 2
    return {"r": r, "half_h": item["height_m"] / 2, "gap": item["diameter_m"]}


# ----------------------------------------------------------------------------------------------- the grip and the net
def grip_needed(item: dict, accel_h: float, accel_v: float, sf: float = 1.0) -> float:
    """Squeeze per pad [N] for two pads to hold ``item`` by friction while Drongo accelerates ``accel_h`` sideways and
    ``accel_v`` upwards [m/s²]: the inertial and gravity load, all of it tangential to the pads (the worst direction),
    shared by two pads at friction μ, × ``sf``."""
    load = item["mass_kg"] * math.hypot(G + accel_v, accel_h)
    return sf * load / (2 * item["mu"])


def grip_holds(item: dict, squeeze: float) -> float:
    """The largest acceleration [m/s²] (sideways, at 1 g up) a squeeze per pad [N] holds the item through."""
    cap = 2 * item["mu"] * squeeze / item["mass_kg"]
    return math.sqrt(max(0.0, cap * cap - G * G))


def net_catch(item: dict, fall_m: float, net: dict = NET) -> dict:
    """An item falling ``fall_m`` onto the net: its speed there (no air drag), the arrest over the net's stretch, the
    load on it, and whether the net and the item survive."""
    v = math.sqrt(2 * G * max(fall_m, 0.0))
    a = v * v / (2 * net["stretch_m"])
    load = item["mass_kg"] * (G + a)
    return {"fall [m]": fall_m, "speed at the net [m/s]": v, "arrest [g]": a / G, "load on the item [N]": load,
            "net holds": fall_m <= net["max_drop_m"], "item survives": load <= item["catch_limit_N"]}


# ----------------------------------------------------------------------------------------------- the Chiron robot
JAWS = ("jaw_front", "jaw_rear")
JAW_JOINTS = ("jaw_front_slide", "jaw_rear_slide")
PADS = ("pad_front", "pad_rear")


def jaw_servo() -> Servo:
    """One jaw's drive as a Chiron servo on its slide [N, m/s]: the servo's torque through the pinion, shared by the
    two racks."""
    a = act.get(JAW_ACTUATOR)
    r2 = 2 * PINION_RADIUS_M
    kp, kd, arm = JAW_GAINS
    return Servo(a.stall_Nm / r2, a.rated_Nm / r2, a.no_load_rpm * 2 * math.pi / 60 * PINION_RADIUS_M, a.stall_A,
                 a.voltage_V, kp, kd, arm, source=f"{a.key} through a {PINION_RADIUS_M * 1000:.0f} mm pinion, two racks: "
                                                   f"{a.source}")


def pad_gap(q, g: dict | None = None) -> float:
    """The gap between the pads [m] at jaw slide positions ``q`` (front, rear)."""
    g = geometry() if g is None else g
    return g["closed_gap"] + float(q[0]) + float(q[1])


def rotor_layout(p: dict | None = None) -> tuple:
    """Rotor positions (4, 3) [m, frame] — arms at 45°, 135°, 225°, 315° — and spin signs (+1: the propeller's drag
    torque on the frame is +z)."""
    g = geometry(p)
    pos = np.array([[g["R"] * math.cos(math.radians(45 + 90 * k)), g["R"] * math.sin(math.radians(45 + 90 * k)),
                     g["arm_h"] + 0.020] for k in range(4)])
    return pos, np.array([1.0, -1.0, 1.0, -1.0])


def _quat_z(angle: float) -> tuple:
    return (math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2))


def drongo(overrides: dict | None = None, *, cad: dict | None = None, name: str = "Drongo") -> Robot:
    """Drongo as a Chiron ``Robot`` (see the module). Feet: the two skids; actuated joints: the two jaw slides."""
    p = design_params(overrides)
    cad = cad_numbers(p) if cad is None else cad
    g = geometry(p)
    m = {k: v / 1000.0 for k, v in mass_budget(p, cad).items()}          # kg
    frame_rgba, dark, pad_rgba = (0.20, 0.22, 0.25, 1.0), (0.08, 0.08, 0.09, 1.0), (0.85, 0.35, 0.10, 1.0)
    # the frame's CAD mass on the plate and the arms by volume
    v_plate = math.pi * g["plate_r"] ** 2 * g["plate_t"]
    v_arm = g["R"] * g["arm_w"] * g["arm_h"]
    m_frame = m["frame (QuadFrame, PETG-CF, CAD)"]
    m_plate = m_frame * v_plate / (v_plate + 4 * v_arm)
    m_arm = (m_frame - m_plate) / 4
    geoms = [Geom("plate", "cylinder", (g["plate_r"], g["plate_t"] / 2), pos=(0, 0, g["plate_t"] / 2), mass=m_plate,
                  role="body", rgba=frame_rgba)]
    pos, _ = rotor_layout(p)
    for k in range(4):
        a = math.radians(45 + 90 * k)
        geoms.append(Geom(f"arm{k}", "box", (g["R"] / 2, g["arm_w"] / 2, g["arm_h"] / 2),
                          pos=(g["R"] / 2 * math.cos(a), g["R"] / 2 * math.sin(a), g["arm_h"] / 2), quat=_quat_z(a),
                          mass=m_arm, role="body", rgba=frame_rgba))
        geoms.append(Geom(f"motor{k}", "cylinder", (g["motor_r"], g["motor_h"] / 2),
                          pos=(pos[k, 0], pos[k, 1], g["arm_h"] + g["motor_h"] / 2), mass=m["motors 2306 (4x)"] / 4,
                          role="visual", rgba=dark))
        geoms.append(Geom(f"prop{k}", "cylinder", (g["prop_r"], 0.0008), pos=tuple(pos[k]),
                          mass=m["propellers 5x4.3 (4x)"] / 4, role="visual", rgba=(0.55, 0.75, 0.95, 0.35)))
    geoms.append(Geom("battery", "box", (0.0375, 0.0175, 0.015), pos=(0, 0, g["plate_t"] + 0.015),
                      mass=m["battery 4S 1500 mAh"], role="body", rgba=(0.95, 0.75, 0.10, 1.0)))
    hx, hy, hz = g["housing"]
    rl, rw, rh = g["rail"]
    m_grip = m["servo housing and rail (PETG-CF, CAD)"]
    v_h, v_r = hx * hy * hz, rl * rw * rh
    geoms += [Geom("gripper_housing", "box", (hx / 2, hy / 2, hz / 2), pos=(0, 0, -hz / 2),
                   mass=m_grip * v_h / (v_h + v_r), role="body", rgba=frame_rgba),
              Geom("rail", "box", (rl / 2, rw / 2, rh / 2), pos=(0, 0, -hz - rh / 2), mass=m_grip * v_r / (v_h + v_r),
                   role="link", rgba=frame_rgba)]
    # landing gear: the skids are the feet; the struts weigh and are drawn
    m_gear = m["landing gear (PETG-CF, CAD)"]
    L_strut = math.hypot(g["skid_y"] - g["strut_y_top"], g["skid_h"] - g["skid_r"])
    m_skid = m_gear * g["skid_L"] / (2 * g["skid_L"] + 4 * L_strut)
    m_strut = m_gear * L_strut / (2 * g["skid_L"] + 4 * L_strut)
    feet = []
    for side, s in (("L", 1), ("R", -1)):
        z = -g["skid_h"] + g["skid_r"]
        geoms.append(Geom(f"skid_{side}", "capsule", (g["skid_r"],),
                          fromto=(-g["skid_L"] / 2 + g["skid_r"], s * g["skid_y"], z, g["skid_L"] / 2 - g["skid_r"], s * g["skid_y"], z),
                          mass=m_skid, role="foot", friction=(0.8, 0.005, 0.0001), rgba=dark))
        for sx in (g["strut_x"], -g["strut_x"]):
            geoms.append(Geom(f"strut_{side}{'f' if sx > 0 else 'r'}", "capsule", (g["strut_r"],),
                              fromto=(sx, s * g["strut_y_top"], 0.0, sx, s * g["skid_y"], z), mass=m_strut,
                              role="visual", rgba=dark))
        feet.append(FootSpec(f"skid_{side}", f"skid_{side}", [], "drongo"))
    masses = [PointMass("esc", m["4-in-1 ESC"], (0, 0, g["plate_t"] * 0.5)),
              PointMass("flight_controller", m["flight controller"], (0, 0, g["plate_t"] * 0.5)),
              PointMass("camera_vtx", m["camera + VTX + antenna"], (0.04, 0, g["plate_t"])),
              PointMass("wiring", m["receiver, wiring, bolts"], (0, 0, 0.0)),
              PointMass("jaw_servo", m[f"jaw servo ({JAW_ACTUATOR})"] + m["pinion, racks, rail screws"], (0, 0, -hz / 2))]
    # the jaws: carriage under the rail, finger, TPU pad; closed (q = 0) the pads' inner faces are closed_gap apart
    cx, cy, cz = g["carriage"]
    ft, fw = g["finger"]
    pt, pw, ph = g["pad"]
    z_c, z_pad = g["carriage_z"], g["pad_z"]
    z_bot = z_pad - ph / 2
    m_jaw, m_pad = m["jaws 2x (PETG-CF, CAD)"] / 2, m["pads 2x (TPU, CAD)"] / 2
    jaws = []
    for jaw, joint, pad, s in zip(JAWS, JAW_JOINTS, PADS, (1, -1)):
        x0 = s * g["closed_gap"] / 2
        jaws.append(Link(jaw, joints=[Joint(joint, kind="slide", axis=(s, 0.0, 0.0), range=(-0.003, g["travel"] + 0.003),
                                            tag="jaw", servo=jaw_servo())],
                         geoms=[Geom(f"{jaw}_carriage", "box", (cx / 2, cy / 2, cz / 2),
                                     pos=(x0 + s * (pt + cx / 2), 0.0, z_c + cz / 2), mass=m_jaw * 0.3, role="link",
                                     rgba=frame_rgba),
                                Geom(f"{jaw}_finger", "box", (ft / 2, fw / 2, (z_c - z_bot) / 2),
                                     pos=(x0 + s * (pt + ft / 2), 0.0, (z_c + z_bot) / 2), mass=m_jaw * 0.7, role="link",
                                     rgba=frame_rgba),
                                Geom(pad, "box", (pt / 2, pw / 2, ph / 2), pos=(x0 + s * pt / 2, 0.0, z_pad), mass=m_pad,
                                     role="link", friction=(0.8, 0.01, 0.0001), rgba=pad_rgba)]))
    root = Link("drongo", geoms=geoms, masses=masses, children=jaws, log=True)
    robot = Robot(name, root=root, feet=feet, nominal_qpos={j: g["ready"] for j in JAW_JOINTS},
                  notes="Drongo: notebook 08's quadcopter with skids and a rack-and-pinion pincer; rotors by the Rotors hook",
                  sources={"geometry": "designs/drongo.py (QuadFrame from notebook 08)", "masses": "drongo_robot.mass_budget",
                           "jaw servo": f"designs/actuators.py: {JAW_ACTUATOR}", "assumptions": "drongo_robot.ASSUMPTIONS"})
    robot.validate()
    robot.params, robot.geometry, robot.cad = p, g, cad
    return robot


#: 1 ms physics, the flight controller at 500 Hz, a log sample every 20 ms; elliptic friction cones with a high
#: impedance ratio and MuJoCo's no-slip solver: a squeezed item sticks to the pads instead of creeping down them
#: (soft-contact friction alone lets a potato held with three times the friction it needs slide 0.6 mm/s).
LAB_OPTIONS = dict(timestep=0.001, control_dt=0.002, log_dt=0.02, cone="elliptic", impratio=10.0, noslip_iterations=5)


class Rotors:
    """Scene hook: Drongo's four rotors and its drag (see the module). The flight controller writes ``command`` (4
    thrusts [N]); every control step the thrusts follow it with the spin-up lag, clipped to [0, T_max], and act on
    the frame through ``lab.body_force``. ``energy_Wh`` is the electrical energy used; ``history`` holds
    [t, T1..T4, P_el] every ``log_every`` calls."""

    def __init__(self, robot: Robot, prop: Propulsion, *, drag_area: float = DRAG_AREA_M2, rho: float = RHO_AIR,
                 log_every: int = 10):
        self.robot, self.prop, self.drag_area, self.rho, self.log_every = robot, prop, float(drag_area), float(rho), log_every
        self.pos, self.spin = rotor_layout(robot.params)

    def reset(self, lab):
        self.root = lab._body_id(self.robot.root.name)
        self.qv = int(lab.model.jnt_dofadr[lab._root_jnt])
        self.command = np.zeros(4)
        self.thrust = np.zeros(4)
        self.energy_J, self.n, self.history = 0.0, 0, []
        self.alpha = 1.0 - math.exp(-lab.control_dt / self.prop.spin_up_s)

    def __call__(self, lab):
        d = lab.data
        dt = lab.control_dt
        cmd = np.clip(self.command, 0.0, self.prop.max_thrust_N)
        self.thrust += (cmd - self.thrust) * self.alpha
        T = self.thrust
        R = d.xmat[self.root].reshape(3, 3)
        zb = R[:, 2]
        arms = (d.xpos[self.root] - d.xipos[self.root]) + self.pos @ R.T          # rotor hubs from the frame's COM
        torque = np.cross(arms, np.outer(T, zb)).sum(axis=0) + zb * float(np.dot(self.spin, T)) * self.prop.torque_per_thrust_m
        v = d.qvel[self.qv:self.qv + 3]
        force = zb * T.sum() - 0.5 * self.rho * self.drag_area * float(np.linalg.norm(v)) * v
        lab.body_force(self.robot.root.name, force, torque)
        P = float(self.prop.electrical_power(T).sum())
        self.energy_J += P * dt
        if self.n % self.log_every == 0:
            self.history.append([lab.time, *T, P])
        self.n += 1

    @property
    def energy_Wh(self) -> float:
        return self.energy_J / 3600.0
