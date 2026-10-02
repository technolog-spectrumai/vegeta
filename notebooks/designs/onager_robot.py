"""Onager Sentinel SX-1 (notebook 20) as a Chiron robot (``vegeta.chiron``): a 380 kg wheel-leg hybrid built from
Chiron's building blocks — hull, four two-link legs (shoulder pitch, knee pitch) each carrying a hub-motor wheel.

Every number comes from the design (``onager.py`` parameters, mm and degrees), from notebook 20's mass budget
(CAD areas and volumes × material densities, plus the listed parts) or from the shared actuator catalogue
(``actuators.py``: ``cycloidal 800 Nm, brake`` at the leg joints, ``hub motor 3 kW`` in the wheels). Inputs the
notebook does not fix are named in ``ASSUMPTIONS`` with the reason for each value. SI at the Chiron boundary.

Frames (Chiron's convention): x forward, y left, z up; the hull frame at the hull box centre. Per leg:

* **shoulder** (tag ``hip_pitch``) — axis +y through the shoulder point; the angle is the upper leg behind
  vertical (``a1``, knee back), ``q_shoulder = a1``;
* **knee** — axis +y at the knee; the lower leg's angle ``a2`` is *absolute* (ahead of vertical), so the
  physical knee angle is ``q_knee = −(a1 + a2)`` (the same convention as the robot dog);
* **wheel** (tag ``wheel``) — an unlimited hinge about +y at the axle, driven by the hub motor as a *velocity*
  servo (``kp = 0``: τ = kd (ω_target − ω), clipped to the motor's torque–speed line). Rolling resistance is a
  joint friction loss ``C_RR · W/4 · r`` (MuJoCo's condim-3 contacts have none).

The wheels are Chiron 'feet' (cylinder geoms, role 'foot'), so the lab's foot forces, slip and support metrics
apply to them unchanged.

    import onager_robot as orb
    robot = orb.onager()                       # Robot, ~390 kg
    lab = orb.onager_lab(chiron.Flat())        # ChironLab(robot, terrain, **orb.LAB_OPTIONS)
"""
from __future__ import annotations

import math

from vegeta.chiron import ChironLab, FootSpec, Geom, Joint, Link, PointMass, Robot, Servo

import actuators as act

__all__ = ["DESIGN", "CAD", "MATERIALS", "PARTS_KG", "ASSUMPTIONS", "LEGS", "LEG_NAMES", "LAB_OPTIONS", "G",
           "C_RR", "LEG_ACTUATOR", "WHEEL_MOTOR", "design_params", "cad_numbers", "geometry", "mass_budget",
           "leg_servo", "wheel_servo", "leg_joints", "wheel_joint", "angles_to_q", "q_to_angles", "nominal_qpos",
           "onager", "onager_lab"]

G = 9.81

# ---- the design: onager.py's parameter defaults (mm, deg; tests check they match OnagerSentinel.parameters) ----
DESIGN = {
    "hull_length": 2000.0, "hull_width": 780.0, "hull_height": 520.0, "hull_chamfer": 60.0, "hull_bottom": 760.0,
    "shoulder_x": 900.0, "shoulder_boss_diameter": 220.0, "shoulder_boss_length": 90.0,
    "upper_leg_length": 520.0, "upper_leg_width": 140.0, "upper_leg_thickness": 50.0, "upper_leg_taper": 0.7,
    "lower_leg_length": 540.0, "lower_leg_width": 120.0, "lower_leg_thickness": 44.0, "lower_leg_taper": 0.75,
    "hip_angle_deg": 40.0, "knee_angle_deg": 55.0, "pin_diameter": 40.0,
    "wheel_diameter": 560.0, "wheel_width": 180.0, "hub_diameter": 260.0, "wheel_offset": 30.0, "axle_diameter": 45.0,
    "turret_length": 500.0, "turret_width": 420.0, "turret_height": 220.0, "mast_height": 450.0, "mast_diameter": 60.0,
    "sensor_head": 150.0,
}

#: CAD measurements of the default design (Dedalus/CadQuery, notebook 20 §1; mm², mm³, mm). ``cad_numbers(
#: recompute=True)`` rebuilds them. Centres of mass are in each part's own frame.
CAD = {
    "hull_surface_area": 6678000.0,
    "hull_com": (-2.0, 0.0, 1044.0),
    "upper_leg_volume": 2710000.0,
    "upper_leg_com": (0.0, 0.0, -229.0),
    "lower_leg_volume": 2120000.0,
    "lower_leg_com": (0.0, 0.0, -251.0),
    "wheel_volume": 45130000.0,
}

#: Materials and the listed parts of notebook 20 §1 (kg; densities in kg/mm³).
MATERIALS = {"Al 5083 armour plate": 2.66e-6, "Al 7075-T6 legs": 2.81e-6}
SHELL_T_MM = 3.0                                     # armour shell thickness over a welded frame (the CAD hull is solid)
PARTS_KG = {
    "frame, mounts, hatches": 28.0,
    "wheels: tyres + rims 4x": 4 * 11.0,
    "battery 48 V LFP 6 kWh": 55.0,
    "sensors: E/O-IR head, lidar, acoustic, mast": 22.0,
    "computer, radios, relay, IMU": 12.0,
    "wiring, connectors, cooling": 15.0,
}
LEG_ACTUATOR = "cycloidal 800 Nm, brake"             # notebook 20 §3: shoulders and knees (8×)
WHEEL_MOTOR = "hub motor 3 kW"                       # notebook 20 §2: one per wheel (4×)
C_RR = 0.03                                          # rolling-resistance coefficient, knobbly tyre on gravel (input)

#: Inputs notebook 20 does not fix, with the reason for each value.
ASSUMPTIONS = {
    "leg_joint_ranges_deg": "shoulder (−20°, 100°) behind vertical, knee flexion a1 + a2 in (20°, 150°): the actuator "
                            "flange allows ±100° (actuators.py 'onager shoulder / knee pin'); the ranges keep the "
                            "wheel off the hull",
    "actuator_positions": "shoulder modules in the hull's shoulder bosses; knee modules at the knee on the upper "
                          "leg; hub motors inside the wheels (their mass rotates with the wheel in this model)",
    "actuator_inertia": "leg modules: reflected inertia 0.05 kg·m² (a 1 kg·cm² rotor through 1:60); hub motor "
                        "rotor 0.05 kg·m² (direct drive) added to the wheel's own ½ m r²",
    "battery_position": "on the hull floor, at the x that would put the standing CG over the centre of the four "
                        "tyre contact patches (x = axle_x, 108 mm ahead of the hull centre: the lower legs slant "
                        "forward) but no further forward than BATTERY_X_MAX = 0.6 m (the front compartment): the "
                        "knee modules sit 0.33 m behind the shoulders, so the CG ends ~50 mm behind the patch "
                        "centre and the rear pair carries ~53 % (notebook 20 §1)",
    "sensor_positions": "turret and mast head masses at their CAD centres (the mast head 1.9 m up)",
    "shell_inertia": "the 3 mm armour shell is given the inertia of a solid box of the hull's size with its mass "
                     "(a hollow shell's would be ~1.6× larger)",
    "leg_links": "machined H-section plates as boxes of their mean width and depth with the CAD mass at the CAD "
                 "centre of mass; the lower leg carries the knee module's partner flange only",
    "friction": "tyres μ = 0.8 (knobbly rubber on dry gravel/concrete, input); hull μ = 0.5",
    "servo_gains": "leg joints kp = 1600 N·m/rad, kd = 120 N·m·s/rad: the knee's vertical rate at the wheel is "
                   "kp / (L2 sin a2)² ≈ 8 kN/m, a 1.5 Hz heave on the 95 kg corner (an active suspension, "
                   "notebook 20 §5); ζ ≈ 0.45. Wheels kd = 30 N·m·s/rad: a 10 rad/s speed error asks for 300 N·m, "
                   "more than the stall torque — the hub motor runs on its torque–speed line until near the target",
    "rolling_resistance": "joint friction loss C_RR·(W/4)·r on each wheel hinge: 0.03 × 955 N × 0.28 m = 8 N·m",
}

#: ChironLab settings (everything else: ChironLab's defaults — 1 ms implicit-fast, pyramidal cones). The wheels
#: are stiff tyres: contact time constant 0.01 s keeps the 95 kg corner's tyre sink ≈ 5 mm (a foam-filled tyre).
LAB_OPTIONS = {"contact_solref": (0.01, 1.0), "heightfield_cell": 0.05, "course_extent": (-3.0, 40.0, -4.0, 4.0)}

LEGS = {"FL": (1, 1), "FR": (1, -1), "RL": (-1, 1), "RR": (-1, -1)}           # wheel name -> (sx, sy)
LEG_NAMES = {"FL": "front-left", "FR": "front-right", "RL": "rear-left", "RR": "rear-right"}
JOINTS = ("shoulder", "knee", "wheel")
LEG_KP, LEG_KD, WHEEL_KD = 1600.0, 120.0, 30.0
LEG_ARMATURE, WHEEL_ARMATURE = 0.05, 0.05
BATTERY_X_MAX = 0.6                                  # m ahead of the hull centre: the front compartment


# ----------------------------------------------------------------------------------------------- the design
def design_params(overrides: dict | None = None) -> dict:
    """``DESIGN`` with ``overrides`` (mm, deg)."""
    p = dict(DESIGN)
    for k, v in (overrides or {}).items():
        if k not in DESIGN:
            raise KeyError(f"unknown OnagerSentinel parameter {k!r}")
        p[k] = float(v)
    return p


def cad_numbers(p: dict | None = None, *, recompute: bool = False) -> dict:
    """The CAD areas, volumes and centres of mass the mass budget needs (mm units, see ``CAD``); the stored values
    for the default design, else rebuilt with ``onager.OnagerSentinel`` (needs CadQuery and vegeta.dedalus)."""
    p = design_params() if p is None else dict(p)
    if not recompute and all(p.get(k) == v for k, v in DESIGN.items()):
        return dict(CAD)
    import onager

    design = onager.OnagerSentinel()
    over = {k: v for k, v in p.items() if k in DESIGN}
    hull = design.generate(part="hull", **over).measure()
    upper = design.generate(part="upper_leg", **over).measure()
    lower = design.generate(part="lower_leg", **over).measure()
    wheel = design.generate(part="wheel", **over).measure()
    return {"hull_surface_area": hull["surface_area"], "hull_com": tuple(hull["center_of_mass"]),
            "upper_leg_volume": upper["volume"], "upper_leg_com": tuple(upper["center_of_mass"]),
            "lower_leg_volume": lower["volume"], "lower_leg_com": tuple(lower["center_of_mass"]),
            "wheel_volume": wheel["volume"]}


def geometry(p: dict | None = None) -> dict:
    """Derived standing geometry [m, rad] (``OnagerSentinel.axle_height`` / ``axle_x`` / ``shoulder_height`` and the
    leg planes of ``OnagerSentinel.build``): ``h_axle`` shoulder to axle, ``axle_x`` axle ahead of the shoulder,
    ``y_upper``/``y_lower``/``y_wheel`` the leg and wheel planes, ``shoulder_x``, ``L1``, ``L2``, ``r_wheel``,
    ``shoulder_z`` (shoulder below the hull frame), ``hull_z`` (hull centre above the ground), ``a1``/``a2``."""
    p = design_params() if p is None else p
    a1, a2 = math.radians(p["hip_angle_deg"]), math.radians(p["knee_angle_deg"])
    L1, L2 = p["upper_leg_length"], p["lower_leg_length"]
    y_upper = p["hull_width"] / 2 + p["shoulder_boss_length"] + p["upper_leg_thickness"] / 2 + 1.0
    y_lower = y_upper + p["upper_leg_thickness"] / 2 + p["lower_leg_thickness"] / 2 + 1.0
    y_wheel = y_lower + p["lower_leg_thickness"] / 2 + p["wheel_offset"] + p["wheel_width"] / 2
    h_axle = L1 * math.cos(a1) + L2 * math.cos(a2)
    shoulder_z_world = p["wheel_diameter"] / 2 + h_axle
    hull_z = p["hull_bottom"] + p["hull_height"] / 2
    return {"h_axle": h_axle / 1000.0, "axle_x": (-L1 * math.sin(a1) + L2 * math.sin(a2)) / 1000.0,
            "y_upper": y_upper / 1000.0, "y_lower": y_lower / 1000.0, "y_wheel": y_wheel / 1000.0,
            "shoulder_x": p["shoulder_x"] / 1000.0, "L1": L1 / 1000.0, "L2": L2 / 1000.0,
            "r_wheel": p["wheel_diameter"] / 2000.0, "w_wheel": p["wheel_width"] / 1000.0,
            "shoulder_z": (shoulder_z_world - hull_z) / 1000.0, "hull_z": hull_z / 1000.0,
            "shoulder_height": shoulder_z_world / 1000.0, "a1": a1, "a2": a2,
            "wheelbase": (2 * p["shoulder_x"]) / 1000.0, "track": 2 * y_wheel / 1000.0}


def mass_budget(p: dict | None = None, cad: dict | None = None) -> dict:
    """Notebook 20 §1's parts list [kg]: part -> mass (actuators from the catalogue)."""
    p = design_params() if p is None else p
    cad = cad_numbers(p) if cad is None else cad
    la, wm = act.get(LEG_ACTUATOR), act.get(WHEEL_MOTOR)
    out = {
        f"armour shell, {SHELL_T_MM:.0f} mm Al 5083 (from CAD area)": cad["hull_surface_area"] * SHELL_T_MM * MATERIALS["Al 5083 armour plate"],
        "upper legs 4x (Al 7075-T6, from CAD)": 4 * cad["upper_leg_volume"] * MATERIALS["Al 7075-T6 legs"],
        "lower legs 4x (Al 7075-T6, from CAD)": 4 * cad["lower_leg_volume"] * MATERIALS["Al 7075-T6 legs"],
        f"leg actuators 8x ({LEG_ACTUATOR}; {la.mass_g / 1000:.1f} kg each)": 8 * la.mass_g / 1000.0,
        f"hub motors 4x ({WHEEL_MOTOR}; {wm.mass_g / 1000:.1f} kg each)": 4 * wm.mass_g / 1000.0,
    }
    out.update(PARTS_KG)
    return out


# ----------------------------------------------------------------------------------------------- joints
def leg_joints(leg: str) -> list:
    """The leg's actuated joints, shoulder to wheel: ``[f'{leg}_shoulder', f'{leg}_knee', f'{leg}_wheel']``."""
    return [f"{leg}_{j}" for j in JOINTS]


def wheel_joint(leg: str) -> str:
    return f"{leg}_wheel"


def angles_to_q(a1_deg: float, a2_deg: float) -> tuple:
    """(upper leg behind vertical, lower leg ahead of vertical) [deg] → (q_shoulder, q_knee) [rad]."""
    r = math.pi / 180.0
    return a1_deg * r, -(a1_deg + a2_deg) * r


def q_to_angles(q_shoulder: float, q_knee: float) -> tuple:
    """(q_shoulder, q_knee) [rad] → (a1, a2) [deg]."""
    d = 180.0 / math.pi
    return q_shoulder * d, -(q_knee + q_shoulder) * d


def nominal_qpos(p: dict | None = None) -> dict:
    """The standing pose (hip_angle_deg, knee_angle_deg) for every leg, wheels at 0 [rad]."""
    p = design_params() if p is None else p
    qs, qk = angles_to_q(p["hip_angle_deg"], p["knee_angle_deg"])
    out = {}
    for leg in LEGS:
        js, jk, jw = leg_joints(leg)
        out[js], out[jk], out[jw] = qs, qk, 0.0
    return out


def leg_servo(kp: float = LEG_KP, kd: float = LEG_KD, actuator: str = LEG_ACTUATOR) -> Servo:
    """A leg joint module as a Chiron position servo (catalogue data + the gains of ``ASSUMPTIONS['servo_gains']``)."""
    return Servo.from_actuator(act.get(actuator), kp=kp, kd=kd, armature=LEG_ARMATURE)


def wheel_servo(kd: float = WHEEL_KD, motor: str = WHEEL_MOTOR) -> Servo:
    """The hub motor as a velocity servo: ``kp = 0``, τ = kd (ω_target − ω) on the torque–speed line."""
    return Servo.from_actuator(act.get(motor), kp=0.0, kd=kd, armature=WHEEL_ARMATURE)


# ----------------------------------------------------------------------------------------------- the robot
def onager(overrides: dict | None = None, *, kp: float = LEG_KP, kd: float = LEG_KD, wheel_kd: float = WHEEL_KD,
           tyre_mu: float = 0.8, hull_mu: float = 0.5, leg_collision: bool = True, rolling_resistance: float = C_RR,
           shoulder_range_deg: tuple = (-20.0, 100.0), knee_flexion_range_deg: tuple = (20.0, 150.0),
           cad: dict | None = None, name: str = "Onager Sentinel SX-1") -> Robot:
    """Onager Sentinel as a Chiron ``Robot``.

    ``overrides``: OnagerSentinel parameters [mm, deg] (other than the defaults, the CAD is rebuilt for the
    masses); ``kp``/``kd``: leg servo gains, ``wheel_kd``: the hub motors' velocity gain; ``tyre_mu``/``hull_mu``:
    sliding friction; ``leg_collision``: the leg plates collide with the terrain (False = visual only, faster);
    ``rolling_resistance``: C_RR of the tyres (a wheel-hinge friction loss). Logged body: ``hull``; feet (wheels)
    ``FL``, ``FR``, ``RL``, ``RR``.
    """
    p = design_params(overrides)
    g = geometry(p)
    cad = cad_numbers(p) if cad is None else cad
    budget = mass_budget(p, cad)
    la, wm = act.get(LEG_ACTUATOR), act.get(WHEEL_MOTOR)
    m_leg_act, m_hub = la.mass_g / 1000.0, wm.mass_g / 1000.0
    servo, wservo = leg_servo(kp, kd), wheel_servo(wheel_kd)
    mm = 1.0 / 1000.0
    Lh, Wh, Hh = p["hull_length"] * mm, p["hull_width"] * mm, p["hull_height"] * mm
    m_shell = budget[f"armour shell, {SHELL_T_MM:.0f} mm Al 5083 (from CAD area)"]
    m_upper = budget["upper legs 4x (Al 7075-T6, from CAD)"] / 4
    m_lower = budget["lower legs 4x (Al 7075-T6, from CAD)"] / 4
    m_tyre = PARTS_KG["wheels: tyres + rims 4x"] / 4
    m_total = sum(budget.values())
    weight_corner = m_total * G / 4
    tau_rr = rolling_resistance * weight_corner * g["r_wheel"]
    m_batt = PARTS_KG["battery 48 V LFP 6 kWh"]
    m_sens = PARTS_KG["sensors: E/O-IR head, lidar, acoustic, mast"]
    m_elec = PARTS_KG["computer, radios, relay, IMU"]

    hull_rgba, leg_rgba, tyre_rgba = (0.36, 0.37, 0.30, 1.0), (0.22, 0.23, 0.22, 1.0), (0.08, 0.08, 0.08, 1.0)
    leg_role = "link" if leg_collision else "visual"
    hull_fr, tyre_fr = (hull_mu, 0.005, 0.0001), (tyre_mu, 0.01, 0.0001)
    half_w_upper = (p["upper_leg_width"] * (1 + p["upper_leg_taper"]) / 4) * mm
    half_w_lower = (p["lower_leg_width"] * (1 + p["lower_leg_taper"]) / 4) * mm
    quat_y = (math.cos(math.pi / 4), -math.sin(math.pi / 4), 0.0, 0.0)      # cylinder axis z -> y
    sh_rng = tuple(math.radians(v) for v in shoulder_range_deg)
    knee_rng = (-math.radians(knee_flexion_range_deg[1]), -math.radians(knee_flexion_range_deg[0]))
    # wheel inertia: tyre + rim as the cylinder geom (MuJoCo computes ½ m r² from the mass), hub motor as a point mass
    # at the axle, its rotor as armature on the wheel hinge
    legs, feet = [], []
    for leg, (sx, sy) in LEGS.items():
        js, jk, jw = leg_joints(leg)
        shoulder = (sx * g["shoulder_x"], sy * g["y_upper"], g["shoulder_z"])
        wheel = Link(f"{leg}_wheel_link", pos=(0.0, sy * (g["y_wheel"] - g["y_lower"]), -g["L2"]),
                     joints=[Joint(jw, axis=(0, 1, 0), range=None, tag="wheel", servo=wservo, leg=leg,
                                   frictionloss=tau_rr)],
                     geoms=[Geom(f"{leg}_wheel", "cylinder", (g["r_wheel"], g["w_wheel"] / 2), quat=quat_y,
                                 mass=m_tyre, role="foot", friction=tyre_fr, rgba=tyre_rgba)],
                     masses=[PointMass(f"{leg}_hub_motor", m_hub, (0.0, 0.0, 0.0))])
        shank = Link(f"{leg}_lower_leg", pos=(0.0, sy * (g["y_lower"] - g["y_upper"]), -g["L1"]),
                     joints=[Joint(jk, axis=(0, 1, 0), range=knee_rng, tag="knee", servo=servo, leg=leg)],
                     geoms=[Geom(f"{leg}_lower_leg", "box", (half_w_lower, p["lower_leg_thickness"] / 2 * mm, g["L2"] / 2),
                                 pos=tuple(c * mm for c in cad["lower_leg_com"]), mass=m_lower, role=leg_role,
                                 friction=hull_fr, rgba=leg_rgba)],
                     children=[wheel])
        thigh = Link(f"{leg}_upper_leg", pos=shoulder,
                     joints=[Joint(js, axis=(0, 1, 0), range=sh_rng, tag="hip_pitch", servo=servo, leg=leg)],
                     geoms=[Geom(f"{leg}_upper_leg", "box", (half_w_upper, p["upper_leg_thickness"] / 2 * mm, g["L1"] / 2),
                                 pos=tuple(c * mm for c in cad["upper_leg_com"]), mass=m_upper, role=leg_role,
                                 friction=hull_fr, rgba=leg_rgba)],
                     masses=[PointMass(f"{leg}_knee_actuator", m_leg_act, (0.0, 0.0, -g["L1"]))],
                     children=[shank])
        legs.append(thigh)
        feet.append(FootSpec(leg, f"{leg}_wheel", [js, jk, jw], "hull"))

    tur_l, tur_w, tur_h = p["turret_length"] * mm, p["turret_width"] * mm, p["turret_height"] * mm
    mast_h, head = p["mast_height"] * mm, p["sensor_head"] * mm
    z_roof = Hh / 2
    x_tur = -Lh * 0.1
    x_mast = x_tur + tur_l * 0.3
    # battery x: puts the whole robot's standing CG over the centre of the tyre contact patches (x = axle_x in the
    # hull frame); everything else is placed first and its moment taken out
    a1, a2 = g["a1"], g["a2"]
    ux, _, uz = (c * mm for c in cad["upper_leg_com"])
    lx, _, lz = (c * mm for c in cad["lower_leg_com"])
    x_upper = ux * math.cos(a1) + uz * math.sin(a1)                         # Ry(a1) applied to the CAD COM
    x_knee = -g["L1"] * math.sin(a1)
    x_lower = x_knee + (lx * math.cos(a2) - lz * math.sin(a2))             # Ry(-a2) applied to the CAD COM
    x_axle = x_knee + g["L2"] * math.sin(a2)
    moment_legs = 4 * (m_upper * x_upper + m_lower * x_lower + (m_tyre + m_hub) * x_axle + m_leg_act * x_knee)
    moment_hull = m_sens * (0.6 * x_tur + 0.4 * x_mast) + m_elec * 0.5
    x_batt = (m_total * g["axle_x"] - moment_legs - moment_hull) / m_batt
    x_batt = min(x_batt, BATTERY_X_MAX)                                    # the knee modules sit 0.33 m behind the
    # shoulders: the exact balance would put the battery past the front wall, so it is capped (ASSUMPTIONS)
    hull = Link("hull", log=True, children=legs,
                geoms=[Geom("hull", "box", (Lh / 2, Wh / 2, Hh / 2), mass=m_shell + PARTS_KG["frame, mounts, hatches"],
                            role="body", friction=hull_fr, rgba=hull_rgba),
                       Geom("turret", "box", (tur_l / 2, tur_w / 2, tur_h / 2), pos=(x_tur, 0.0, z_roof + tur_h / 2),
                            mass=m_sens * 0.6, role="body", friction=hull_fr, rgba=hull_rgba),
                       Geom("mast", "cylinder", (p["mast_diameter"] / 2000.0, mast_h / 2),
                            pos=(x_mast, 0.0, z_roof + tur_h + mast_h / 2), mass=m_sens * 0.15, role="visual",
                            rgba=leg_rgba),
                       Geom("sensor_head", "box", (head / 2, head / 2, head / 2),
                            pos=(x_mast, 0.0, z_roof + tur_h + mast_h + head / 2), mass=m_sens * 0.25, role="visual",
                            rgba=leg_rgba)],
                masses=[PointMass("battery", m_batt, (x_batt, 0.0, -Hh / 2 + 0.08)),
                        PointMass("computer, radios, relay, IMU", m_elec, (0.5, 0.0, 0.0)),
                        PointMass("wiring, connectors, cooling", PARTS_KG["wiring, connectors, cooling"], (0.0, 0.0, 0.0))]
                + [PointMass(f"{leg}_shoulder_actuator", m_leg_act, (sx * g["shoulder_x"], sy * (Wh / 2 + 0.045), g["shoulder_z"]))
                   for leg, (sx, sy) in LEGS.items()])
    robot = Robot(name, hull, feet=feet, nominal_qpos=nominal_qpos(p), nominal_base_height=g["hull_z"],
                  nominal_hip_height=g["shoulder_height"],
                  notes=f"Onager Sentinel SX-1: hull + 4 wheel-legs; {LEG_ACTUATOR} ×8, {WHEEL_MOTOR} ×4; "
                        f"wheel hinge friction loss {tau_rr:.1f} N·m (C_RR {rolling_resistance})",
                  sources={"geometry": "designs/onager.py", "masses": "notebook 20 §1 mass budget",
                           "actuators": "designs/actuators.py", "assumptions": "onager_robot.ASSUMPTIONS"})
    robot.validate()
    robot.params = p
    robot.geometry = g
    robot.battery_x = x_batt
    return robot


def onager_lab(terrain=None, robot: Robot | None = None, **kwargs) -> ChironLab:
    """``ChironLab(onager(), terrain, **LAB_OPTIONS)`` (``kwargs`` override the options)."""
    opts = dict(LAB_OPTIONS)
    opts.update(kwargs)
    return ChironLab(robot or onager(), terrain, **opts)
