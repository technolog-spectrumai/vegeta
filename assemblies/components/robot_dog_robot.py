"""Cerberus, notebook 16's robot dog, as a Chiron robot (``vegeta.chiron``) — built from Chiron's building blocks
(``Link``, ``Joint``, ``Geom``, ``PointMass``, ``Servo``, ``FootSpec``, ``Robot``), no MJCF of its own.

Every number comes from the dog's design (``robot_dog.py`` parameters, in mm and degrees), from notebook 16's
mass budget (§1: CAD areas and volumes × material densities, plus the listed parts) or from the shared actuator
catalogue (``actuators.py``: ``qdd 24 Nm``). The few inputs the notebook does not fix are named in
``ASSUMPTIONS`` with the reason for each value. Units at the Chiron boundary: SI (m, kg, rad).

Frames (Chiron's convention): x forward, y left, z up. The trunk frame sits at the body centre, which is
``body_centre_above_hip`` (8 mm, notebook 16 / ``RobotDog._body``) above the hip axes. Each leg has three
actuated joints, as notebook 16's mass budget lists them and as ``gait.ik_dog`` / ``gait.fk_dog`` model them:

* **hip roll** — axis +x through the hip point; positive swings the foot to +y (``gait.Rx``);
* **hip pitch** — axis +y through the hip point; the angle is ``gait``'s upper-leg angle behind vertical
  (``a1``, knee back), so ``q_pitch = a1``;
* **knee** — axis +y at the knee; ``gait``'s lower-leg angle ``a2`` is *absolute* (ahead of vertical), so the
  physical knee joint is ``q_knee = −(a1 + a2)`` (≤ 0: the knee folds backwards).

``angles_to_q`` / ``q_to_angles`` convert between ``gait.ik_dog``'s (roll, a1, a2) [deg] and joint angles [rad].

    from . import robot_dog_robot as rdr
    robot = rdr.dog_robot()                 # 13.07 kg, 12 × qdd 24 Nm
    lab = rdr.dog_lab(chiron.Flat())         # = ChironLab(robot, terrain, **rdr.LAB_OPTIONS)

Promoted from ``notebooks/designs/robot_dog_robot.py`` as it was proven there; the notebook copy may move on.
"""
from __future__ import annotations

import math

from vegeta.chiron import ChironLab, FootSpec, Geom, Joint, Link, PointMass, Robot, Servo

from . import actuators as act

__all__ = ["DOG", "CAD", "ASSUMPTIONS", "LEGS", "LEG_NAMES", "JOINTS", "DENSITY_G_MM3", "SHELL_T_MM",
           "OTHER_PARTS_G", "LAB_OPTIONS", "design_params", "cad_numbers", "geometry", "mass_budget", "dog_servo",
           "dog_robot", "dog_lab", "leg_joints", "angles_to_q", "q_to_angles", "nominal_qpos"]

# ---- the design: robot_dog.py's parameter defaults (mm, deg; tests check they match RobotDog.parameters) ----
DOG = {
    "body_length": 520.0, "body_width": 240.0, "body_height": 130.0, "shell_fillet": 28.0, "hip_x": 190.0,
    "hip_boss_diameter": 64.0, "hip_boss_length": 30.0, "upper_leg_length": 200.0, "upper_leg_width": 44.0,
    "upper_leg_thickness": 16.0, "upper_leg_taper": 0.65, "lower_leg_length": 210.0, "lower_leg_diameter": 24.0,
    "foot_diameter": 34.0, "hip_angle_deg": 35.0, "knee_angle_deg": 40.0, "pin_diameter": 10.0,
    "head_length": 150.0, "head_height": 95.0, "head_drop": 25.0, "deck_length": 300.0, "deck_width": 200.0,
    "deck_depth": 6.0, "deck_hole_diameter": 5.5, "deck_hole_pitch": 200.0,
}

#: CAD measurements of the default design (Dedalus/CadQuery, notebook 16 §1: ``dog_design.generate(part=...)``,
#: mm, mm², mm³; ``cad_numbers(recompute=True)`` rebuilds them). ``body`` is shell + deck + head + hip bosses +
#: tail; centres of mass are in each part's own frame (the body: world, hips at the standing height).
CAD = {
    "body_surface_area": 495514.2529971391,
    "body_com": (30.84521336968371, 0.0, 328.81858007346796),
    "upper_leg_volume": 117650.76171336346,
    "upper_leg_com": (0.0, 0.0, -89.05230456368481),
    "lower_leg_volume": 87472.60688409615,
    "lower_leg_com": (-3.855207001559973, 0.0, -104.94395338785986),
}

#: Notebook 16 §1 mass budget inputs.
SHELL_T_MM = 3.0                                     # the body is a printed shell of this thickness over a frame
DENSITY_G_MM3 = {"PA12-CF": 1.1e-3, "CF-nylon": 1.15e-3}   # shell and upper legs; lower legs
OTHER_PARTS_G = {                                    # notebook 16 §1, listed parts [g]
    "aluminium frame + deck plate": 1400.0,
    "battery 10S 10 Ah": 2200.0,
    "computer, radios, IMU, cameras": 650.0,
    "wiring, bearings, pins, feet": 500.0,
}
ACTUATOR_KEY = "qdd 24 Nm"                           # notebook 16 §2: every leg joint (12 × 480 g)
X_CG_MM = 25.0                                       # notebook 16 §2: the dog's CG ahead of the body centre (input)

#: Inputs notebook 16 does not fix, with the reason for each value (all in SI unless named).
ASSUMPTIONS = {
    "body_centre_above_hip": "8 mm: RobotDog._body puts the shell centre at the standing height + 8 mm",
    "hip_roll_axis": "through the hip point in the leg plane (y = ±y_leg), as gait.ik_dog / fk_dog model it; "
                     "the CAD has no hip-roll joint yet (notebook 16 §11 lists it as a next step)",
    "hip_roll_range_deg": "±45°: not given in notebook 16; wide enough for gait.ik_dog over rough ground",
    "hip_pitch_range_deg": "(−30°, 110°): notebook 16 §3 HIP_RANGE (upper leg behind vertical)",
    "knee_range_deg": "flexion a1 + a2 in (0°, 150°) = q_knee in (−150°, 0°): notebook 16 §3 KNEE_RANGE, read as "
                      "the relative knee angle (the notebook applies it to the lower-leg angle)",
    "actuator_positions": "hip-roll actuators in the body at (±hip_x, ±(body_width/2 − 40 mm)), at hip height; "
                          "hip-pitch and knee actuators coaxial in the hip boss on the hip-roll link (the knee "
                          "driven 1:1 through a linkage; the usual QDD-dog layout that keeps the leg light). "
                          "Notebook 16 gives masses, not positions.",
    "actuator_inertia": "actuators are point masses; reflected rotor inertia (armature) 0 — the catalogue has no "
                        "rotor inertia for the qdd 24 Nm",
    "battery_x": "chosen so the dog's CG lies X_CG = 25 mm ahead of the body centre (notebook 16 §2 input)",
    "electronics_position": "computer, radios, IMU, cameras in the head (notebook 16: 'head, computer up front')",
    "frame_and_wiring_position": "aluminium frame + deck and wiring/bearings/pins/feet at the body centre",
    "shell_inertia": "the 3 mm shell's mass is split between the trunk and head boxes by their surface areas and "
                     "given the inertia of solid boxes (a hollow shell's would be ~1.7× larger)",
    "leg_links": "upper leg: a box of the plate's mean width/thickness with its mass at the CAD centre of mass; "
                 "lower leg: a capsule (mean rod radius) knee → foot centre with its mass at the capsule centre "
                 "(CAD: 3.9 mm behind it); the foot sphere is part of the lower-leg CAD volume (massless here)",
    "friction": "feet μ = 0.7 (notebook 16 §3 MU_FEET, dry concrete); shell/head μ = 0.5 (printed PA12-CF on "
                "concrete; not in notebook 16)",
    "servo_gains": "kp = 100 N·m/rad, kd = 1.0 N·m·s/rad on every joint: not in notebook 16. kp: a stance knee "
                   "carrying a third of the weight (≈ 6 N·m) sags ≈ 3°; kd: the free lower leg (≈ 1.5e-3 kg·m² "
                   "about the knee) is overdamped (ζ ≈ 1.3), and kd ≲ 1.2 keeps ChironLab's default implicit-fast "
                   "integrator out of a contact limit cycle: with damping integrated implicitly (implicit-fast, or "
                   "Euler with joint damping) kd ≥ 2 makes the standing dog bounce at ≈ 10 Hz (summed foot force "
                   "0.4–1.6 m g); explicit damping (Euler) is stable standing but unstable on a free lower leg for "
                   "kd ≥ 2",
}

#: ChironLab settings for the dog (everything else: ChironLab's defaults — 1 ms implicit-fast, pyramidal cones).
#: contact_solref: MuJoCo's default contact time constant (0.02 s) lets a loaded foot sink several mm (≈ 10 mm at a
#: walking touchdown of this 13 kg dog) — on a 5 mm-cell height field the 17 mm foot sphere then meets prism sides
#: (nearly horizontal contact normals), gets trapped below the surface and sees kN spikes; 0.005 s (≥ 2 time
#: steps, MuJoCo's lower bound) keeps the sink < 1 mm: a 0.3 m/s walk on a flat 5 mm height field covers 0.89 m in
#: 4 s (0.39 m at the default; 0.97 m on a plane).
LAB_OPTIONS = {"contact_solref": (0.005, 1.0)}

LEGS = {"FL": (1, 1), "FR": (1, -1), "RL": (-1, 1), "RR": (-1, -1)}           # foot name -> (sx, sy)
LEG_NAMES = {"FL": "front-left", "FR": "front-right", "RL": "rear-left", "RR": "rear-right"}   # gait.LEGS names
JOINTS = ("hip_roll", "hip_pitch", "knee")
DEFAULT_KP, DEFAULT_KD = 100.0, 1.0


# ----------------------------------------------------------------------------------------------- the design
def design_params(overrides: dict | None = None) -> dict:
    """``DOG`` with ``overrides`` (mm, deg)."""
    p = dict(DOG)
    for k, v in (overrides or {}).items():
        if k not in DOG:
            raise KeyError(f"unknown RobotDog parameter {k!r}")
        p[k] = float(v)
    return p


def cad_numbers(p: dict | None = None, *, recompute: bool = False) -> dict:
    """The CAD areas, volumes and centres of mass the mass budget needs (mm units, see ``CAD``).

    The stored ``CAD`` values for the default design; for other parameters (or ``recompute=True``) the parts
    are rebuilt with ``robot_dog.RobotDog`` (needs CadQuery and vegeta.dedalus)."""
    p = design_params() if p is None else dict(p)
    if not recompute and all(p.get(k) == v for k, v in DOG.items()):
        return dict(CAD)
    from . import robot_dog

    design = robot_dog.RobotDog()
    over = {k: v for k, v in p.items() if k in DOG}
    body = design.generate(part="body", **over).measure()
    upper = design.generate(part="upper_leg", **over).measure()
    lower = design.generate(part="lower_leg", **over).measure()
    return {"body_surface_area": body["surface_area"], "body_com": tuple(body["center_of_mass"]),
            "upper_leg_volume": upper["volume"], "upper_leg_com": tuple(upper["center_of_mass"]),
            "lower_leg_volume": lower["volume"], "lower_leg_com": tuple(lower["center_of_mass"])}


def geometry(p: dict | None = None) -> dict:
    """Derived standing geometry [m, rad] (the formulas of ``RobotDog.standing_height`` / ``foot_x`` and notebook
    16's ``y_leg``): ``h_stand`` hip to foot centre, ``foot_x`` foot ahead of the hip, ``y_leg`` the leg plane,
    ``hip_x``, ``L1``, ``L2``, ``foot_r``, ``hip_z`` (hip below the trunk frame), standing angles ``a1``/``a2``."""
    p = design_params() if p is None else p
    a1, a2 = math.radians(p["hip_angle_deg"]), math.radians(p["knee_angle_deg"])
    L1, L2 = p["upper_leg_length"], p["lower_leg_length"]
    y_leg = p["body_width"] / 2 + p["hip_boss_length"] + p["upper_leg_thickness"] / 2 + 1.0
    return {"h_stand": (L1 * math.cos(a1) + L2 * math.cos(a2)) / 1000.0,
            "foot_x": (-L1 * math.sin(a1) + L2 * math.sin(a2)) / 1000.0,
            "y_leg": y_leg / 1000.0, "hip_x": p["hip_x"] / 1000.0, "L1": L1 / 1000.0, "L2": L2 / 1000.0,
            "foot_r": p["foot_diameter"] / 2000.0, "hip_z": -8.0 / 1000.0, "a1": a1, "a2": a2,
            "wheelbase": 2 * p["hip_x"] / 1000.0}


def mass_budget(p: dict | None = None, cad: dict | None = None) -> dict:
    """Notebook 16 §1's parts list [kg]: part -> mass (12 actuators from the catalogue)."""
    p = design_params() if p is None else p
    cad = cad_numbers(p) if cad is None else cad
    a = act.get(ACTUATOR_KEY)
    out = {
        "body shell, 3 mm PA12-CF (from CAD area)": cad["body_surface_area"] * SHELL_T_MM * DENSITY_G_MM3["PA12-CF"],
        "aluminium frame + deck plate": OTHER_PARTS_G["aluminium frame + deck plate"],
        "upper legs 4x (from CAD)": 4 * cad["upper_leg_volume"] * DENSITY_G_MM3["PA12-CF"],
        "lower legs 4x (from CAD, CF-nylon)": 4 * cad["lower_leg_volume"] * DENSITY_G_MM3["CF-nylon"],
        f"actuators 12x (hip roll, hip pitch, knee; {a.mass_g:.0f} g each)": 12 * a.mass_g,
        "battery 10S 10 Ah": OTHER_PARTS_G["battery 10S 10 Ah"],
        "computer, radios, IMU, cameras": OTHER_PARTS_G["computer, radios, IMU, cameras"],
        "wiring, bearings, pins, feet": OTHER_PARTS_G["wiring, bearings, pins, feet"],
    }
    return {k: v / 1000.0 for k, v in out.items()}


# ----------------------------------------------------------------------------------------------- joints
def leg_joints(leg: str) -> list:
    """The leg's actuated joints, hip to foot: ``[f'{leg}_hip_roll', f'{leg}_hip_pitch', f'{leg}_knee']``."""
    return [f"{leg}_{j}" for j in JOINTS]


def angles_to_q(roll_deg: float, a1_deg: float, a2_deg: float) -> tuple:
    """``gait.ik_dog``'s (hip roll, upper leg behind vertical, lower leg ahead of vertical) [deg] → joint angles
    (q_roll, q_pitch, q_knee) [rad]."""
    r = math.pi / 180.0
    return roll_deg * r, a1_deg * r, -(a1_deg + a2_deg) * r


def q_to_angles(q_roll: float, q_pitch: float, q_knee: float) -> tuple:
    """Joint angles [rad] → ``gait.fk_dog``'s (roll, a1, a2) [deg]."""
    d = 180.0 / math.pi
    return q_roll * d, q_pitch * d, -(q_knee + q_pitch) * d


def nominal_qpos(p: dict | None = None) -> dict:
    """Notebook 16's standing pose (hip_angle_deg, knee_angle_deg; no roll) for every leg [rad]."""
    p = design_params() if p is None else p
    q = angles_to_q(0.0, p["hip_angle_deg"], p["knee_angle_deg"])
    return {jn: v for leg in LEGS for jn, v in zip(leg_joints(leg), q)}


def dog_servo(kp: float = DEFAULT_KP, kd: float = DEFAULT_KD, actuator: str = ACTUATOR_KEY) -> Servo:
    """The leg actuator as a Chiron servo: catalogue data (stall, rated, no-load speed, stall current, voltage)
    with the position-loop gains ``kp`` [N·m/rad], ``kd`` [N·m·s/rad] (``ASSUMPTIONS['servo_gains']``)."""
    return Servo.from_actuator(act.get(actuator), kp=kp, kd=kd)


# ----------------------------------------------------------------------------------------------- the robot
def dog_robot(overrides: dict | None = None, *, kp: float = DEFAULT_KP, kd: float = DEFAULT_KD,
              foot_mu: float = 0.7, shell_mu: float = 0.5, leg_collision: bool = True,
              hip_roll_range_deg: tuple = (-45.0, 45.0), hip_pitch_range_deg: tuple = (-30.0, 110.0),
              knee_flexion_range_deg: tuple = (0.0, 150.0), cad: dict | None = None, name: str = "Cerberus"
              ) -> Robot:
    """Cerberus as a Chiron ``Robot`` (13.07 kg with the default design).

    ``overrides``: RobotDog parameters [mm, deg] (other than the defaults, the CAD is rebuilt for the masses);
    ``kp``/``kd``: servo gains of every joint; ``foot_mu``/``shell_mu``: sliding friction of the feet and of the
    shell and head (belly contacts); ``leg_collision``: upper and lower legs collide with the terrain (role
    'link'; False = visual only, faster); joint ranges in degrees (see ``ASSUMPTIONS``). Logged body: ``trunk``
    (the body with the head welded on); feet ``FL``, ``FR``, ``RL``, ``RR`` (sphere pads).
    """
    p = design_params(overrides)
    g = geometry(p)
    cad = cad_numbers(p) if cad is None else cad
    budget = mass_budget(p, cad)
    a = act.get(ACTUATOR_KEY)
    m_act = a.mass_g / 1000.0
    servo = dog_servo(kp, kd)
    mm = 1.0 / 1000.0
    Lb, Wb, Hb = p["body_length"] * mm, p["body_width"] * mm, p["body_height"] * mm
    Lh, Hh, Wh = p["head_length"] * mm, p["head_height"] * mm, p["body_width"] * 0.55 * mm
    head_pos = (Lb / 2 + Lh / 2 - 15 * mm, 0.0, -p["head_drop"] * mm)
    # shell mass: split between the trunk box and the head box by surface area
    area_trunk = 2 * (Lb * Wb + Lb * Hb + Wb * Hb)
    area_head = 2 * (Lh * Wh + Lh * Hh + Wh * Hh)
    m_shell = budget["body shell, 3 mm PA12-CF (from CAD area)"]
    m_shell_trunk = m_shell * area_trunk / (area_trunk + area_head)
    m_shell_head = m_shell - m_shell_trunk
    m_upper = budget["upper legs 4x (from CAD)"] / 4
    m_lower = budget["lower legs 4x (from CAD, CF-nylon)"] / 4
    m_frame = OTHER_PARTS_G["aluminium frame + deck plate"] * mm
    m_batt = OTHER_PARTS_G["battery 10S 10 Ah"] * mm
    m_elec = OTHER_PARTS_G["computer, radios, IMU, cameras"] * mm
    m_wire = OTHER_PARTS_G["wiring, bearings, pins, feet"] * mm
    # battery x: puts the whole dog's CG X_CG ahead of the body centre in the standing pose (actuators are
    # fore-aft symmetric; the legs, knees back, are not)
    m_total = sum(budget.values())
    a1, a2 = g["a1"], g["a2"]
    ux, _, uz = (c * mm for c in cad["upper_leg_com"])
    x_upper = ux * math.cos(a1) + uz * math.sin(a1)                         # Ry(a1) applied to the CAD COM
    x_knee = -g["L1"] * math.sin(a1)
    x_lower = x_knee + g["L2"] / 2 * math.sin(a2)
    moment_others = (m_shell_head + m_elec) * head_pos[0] + 4 * (m_upper * x_upper + m_lower * x_lower)
    x_batt = (m_total * X_CG_MM * mm - moment_others) / m_batt

    shell_rgba, leg_rgba, foot_rgba = (0.81, 0.84, 0.87, 1.0), (0.25, 0.26, 0.28, 1.0), (0.1, 0.1, 0.1, 1.0)
    leg_role = "link" if leg_collision else "visual"
    shell_fr = (shell_mu, 0.005, 0.0001)
    foot_fr = (foot_mu, 0.005, 0.0001)
    k_taper = p["upper_leg_taper"]
    half_w_upper = (p["upper_leg_width"] + p["upper_leg_width"] * k_taper) / 4 * mm
    rod_r = (p["lower_leg_diameter"] + 0.6 * p["lower_leg_diameter"]) / 4 * mm
    boss_r, boss_half = p["hip_boss_diameter"] / 2 * mm, p["hip_boss_length"] / 2 * mm
    quat_y = (math.cos(math.pi / 4), -math.sin(math.pi / 4), 0.0, 0.0)      # cylinder axis z -> y
    roll_rng = tuple(math.radians(v) for v in hip_roll_range_deg)
    pitch_rng = tuple(math.radians(v) for v in hip_pitch_range_deg)
    knee_rng = (-math.radians(knee_flexion_range_deg[1]), -math.radians(knee_flexion_range_deg[0]))

    legs, feet, trunk_masses, trunk_geoms = [], [], [], []
    for leg, (sx, sy) in LEGS.items():
        jr, jp, jk = leg_joints(leg)
        hip = (sx * g["hip_x"], sy * g["y_leg"], g["hip_z"])
        boss_y = sy * (Wb / 2 + boss_half) - hip[1]                         # hip boss centre, in the hip frame
        shank = Link(f"{leg}_lower_leg", pos=(0.0, 0.0, -g["L1"]),
                     joints=[Joint(jk, axis=(0, 1, 0), range=knee_rng, tag="knee", servo=servo, leg=leg)],
                     geoms=[Geom(f"{leg}_lower_leg", "capsule", (rod_r,), fromto=(0, 0, 0, 0, 0, -g["L2"]),
                                 mass=m_lower, role=leg_role, friction=shell_fr, rgba=leg_rgba),
                            Geom(f"{leg}_foot", "sphere", (g["foot_r"],), pos=(0.0, 0.0, -g["L2"]),
                                 role="foot", friction=foot_fr, rgba=foot_rgba)])
        thigh = Link(f"{leg}_upper_leg",
                     joints=[Joint(jp, axis=(0, 1, 0), range=pitch_rng, tag="hip_pitch", servo=servo, leg=leg)],
                     geoms=[Geom(f"{leg}_upper_leg", "box", (half_w_upper, p["upper_leg_thickness"] / 2 * mm,
                                                             g["L1"] / 2),
                                 pos=tuple(c * mm for c in cad["upper_leg_com"]), mass=m_upper, role=leg_role,
                                 friction=shell_fr, rgba=leg_rgba)],
                     children=[shank])
        hip_link = Link(f"{leg}_hip", pos=hip,
                        joints=[Joint(jr, axis=(1, 0, 0), range=roll_rng, tag="hip_roll", servo=servo, leg=leg)],
                        masses=[PointMass(f"{leg}_hip_pitch_actuator", m_act, (0.0, boss_y, 0.0)),
                                PointMass(f"{leg}_knee_actuator", m_act, (0.0, boss_y, 0.0))],
                        children=[thigh])
        legs.append(hip_link)
        feet.append(FootSpec(leg, f"{leg}_foot", [jr, jp, jk], "trunk"))
        trunk_masses.append(PointMass(f"{leg}_hip_roll_actuator", m_act,
                                      (sx * g["hip_x"], sy * (Wb / 2 - 0.040), g["hip_z"])))
        trunk_geoms.append(Geom(f"{leg}_hip_boss", "cylinder", (boss_r, boss_half),
                                pos=(sx * g["hip_x"], sy * (Wb / 2 + boss_half), g["hip_z"]), quat=quat_y,
                                role="body", friction=shell_fr, rgba=shell_rgba))
    trunk = Link(
        "trunk", log=True, group="body",
        geoms=[Geom("trunk_shell", "box", (Lb / 2, Wb / 2, Hb / 2), mass=m_shell_trunk, role="body",
                    friction=shell_fr, rgba=shell_rgba),
               Geom("head", "box", (Lh / 2, Wh / 2, Hh / 2), pos=head_pos, mass=m_shell_head, role="body",
                    friction=shell_fr, rgba=shell_rgba)] + trunk_geoms,
        masses=[PointMass("frame_and_deck", m_frame, (0.0, 0.0, 0.0)),
                PointMass("battery", m_batt, (x_batt, 0.0, 0.0)),
                PointMass("electronics", m_elec, head_pos),
                PointMass("wiring_bearings_pins_feet", m_wire, (0.0, 0.0, 0.0))] + trunk_masses,
        children=legs)
    sources = {
        "geometry": "notebooks/designs/robot_dog.py (RobotDog parameter defaults, standing_height, foot_x)",
        "masses": "notebook 16 §1 mass budget (CAD area/volumes × densities + listed parts), 13.07 kg",
        "actuators": f"notebooks/designs/actuators.py: {a.key} ({a.source})",
        "standing pose": "notebook 16: hip_angle_deg, knee_angle_deg (foot 20 mm ahead of the hip, 325 mm below)",
        "kinematics": "gait.ik_dog / fk_dog (roll axis through the hip point)",
        "assumptions": "robot_dog_robot.ASSUMPTIONS",
    }
    notes = (f"Cerberus (notebook 16): {m_total:.2f} kg, 12 × {a.key}; hips at x = ±{g['hip_x']:.3f} m, "
             f"leg planes at y = ±{g['y_leg']:.3f} m; standing hip-to-foot-centre {g['h_stand']:.4f} m.")
    return Robot(name, trunk, feet=feet, nominal_qpos=nominal_qpos(p), notes=notes, sources=sources)


def dog_lab(terrain=None, *, robot: Robot | None = None, **lab_kwargs) -> ChironLab:
    """A ChironLab with the dog (``dog_robot()`` unless ``robot`` is given) on ``terrain`` (default flat) with
    ``LAB_OPTIONS``; ``lab_kwargs`` are ChironLab keywords and win over ``LAB_OPTIONS``."""
    kw = dict(LAB_OPTIONS)
    kw.update(lab_kwargs)
    return ChironLab(robot if robot is not None else dog_robot(), terrain, **kw)
