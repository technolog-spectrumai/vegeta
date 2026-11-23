"""Notebook 11's rover as a Chiron robot: the ``Rover`` design's tub chassis, four trailing arms on sprung pivots and
a driven wheel on each (``rover.py`` parameters, mm and degrees; notebook 11's parts list and suspension).

* **chassis** — the root link (``role='body'``), its CAD mass and the bought parts as point masses;
* **arms** — a hinge about +y at each pivot, the torsion spring and damper of notebook 11 (``K_SUSP`` N/m at the
  wheel, ``ZETA``) as joint stiffness and damping (× the arm length²); front arms lead, rear arms trail;
* **wheels** — an unlimited hinge about +y at the axle, a ``Servo`` with notebook 11's stall torque at the wheel.

Units: SI. x forward, y left, z up; the root frame at the chassis plate's top centre (z = 0 of the CAD).

    import rover_robot as rr
    robot = rr.rover()                        # 2.1 kg (notebook 11's parts list and CAD)
"""
from __future__ import annotations

import math

from vegeta.chiron import FootSpec, Geom, Joint, Link, PointMass, Robot, Servo

#: Notebook 11: PA12-CF (printed) density [g/mm³], bought parts [g], the suspension and the drive.
RHO_PA12CF = 1.1e-3
PARTS_G = {"wheels + tyres 4x": 4 * 120.0, "motors + gearboxes 4x": 4 * 95.0, "battery 3S 5000 mAh": 380.0,
           "controller, radio, camera": 180.0, "wiring, bolts, bearings": 120.0}
K_SUSP = 520.0          # N/m at the wheel
ZETA = 0.3
T_STALL = 1.2           # N·m at the wheel (motor × gearbox)
ASSUMPTIONS = {
    "no_load_speed": "30 rad/s at the wheel (1.6 m/s with the 110 mm wheel): notebook 11 drives at up to 1.5 m/s and "
                     "states no motor speed",
    "electrics": "3S (11.1 V), 5 A stall per motor: notebook 11 states the battery only",
    "wheel_kd": "0.05 N·m·s/rad: a velocity-driven wheel, as the other Vegeta rovers",
}


def design_params(overrides: dict | None = None) -> dict:
    from rover import Rover
    p = {prm.name: prm.default for prm in Rover.parameters}
    p.update(overrides or {})
    return p


def cad_volumes(p: dict) -> dict:
    """The chassis' and one arm's CAD volumes [mm³] (``rover.Rover``, as notebook 11 measures them)."""
    from rover import Rover
    over = {k: v for k, v in p.items() if k != "part"}
    return {part: Rover().generate(part=part, **over).measure()["volume"] for part in ("chassis", "arm")}


def rover(overrides: dict | None = None, *, cad: dict | None = None, name: str = "Rover", tyre_mu: float = 0.9) -> Robot:
    p = design_params(overrides)
    cad = cad_volumes(p) if cad is None else cad
    mm = 1e-3
    m_chassis = cad["chassis"] * RHO_PA12CF / 1000
    m_arm = cad["arm"] * RHO_PA12CF / 1000
    m_wheel = (PARTS_G["wheels + tyres 4x"] + PARTS_G["motors + gearboxes 4x"]) / 4 / 1000
    Lc, Wc, tp, rh = p["length"] * mm, p["width"] * mm, p["plate_thickness"] * mm, p["rail_height"] * mm
    L, h, w = p["arm_length"] * mm, p["arm_height"] * mm, p["wheel_width"] * mm
    ang = math.radians(p["arm_angle_deg"])
    px, pz = Lc / 2 - p["pivot_x"] * mm, -tp - rh / 2
    r_wheel = p["wheel_diameter"] / 2 * mm
    k_rot, c_rot = K_SUSP * L * L, 2 * ZETA * math.sqrt(K_SUSP * 0.6) * L * L      # 0.6 kg: notebook 11's sprung corner
    quat_y = (math.cos(math.pi / 4), -math.sin(math.pi / 4), 0.0, 0.0)          # cylinder axis z -> y
    servo = Servo(stall_torque=T_STALL, rated_torque=T_STALL / 3, no_load_speed=30.0, stall_current=5.0, voltage=11.1,
                  kp=0.0, kd=0.05, source="notebook 11 T_STALL; speed and electrics: ASSUMPTIONS")
    rgba, tyre = (0.20, 0.22, 0.25, 1.0), (0.08, 0.08, 0.08, 1.0)
    children, feet = [], []
    for sx, end in ((1, "front"), (-1, "rear")):
        for sy, side in ((1, "L"), (-1, "R")):
            y_arm = sy * (Wc / 2 + h / 2 + 1.0 * mm)
            ax = (sx * L * math.cos(ang), 0.0, -L * math.sin(ang))
            y_wheel = sy * (h / 2 + p["wheel_offset"] * mm + w / 2)
            wn = f"{end}_{side}_wheel"
            wheel = Link(f"{wn}_link", pos=(ax[0], y_wheel, ax[2]),
                         joints=[Joint(wn, axis=(0, 1, 0), tag="wheel", servo=servo, leg=wn)],
                         geoms=[Geom(wn, "cylinder", (r_wheel, w / 2), quat=quat_y, mass=m_wheel, role="foot",
                                     friction=(tyre_mu, 0.01, 0.0001), rgba=tyre)])
            an = f"{end}_{side}_arm"
            arm = Link(f"{an}_link", pos=(sx * px, y_arm, pz),
                       joints=[Joint(an, axis=(0, 1, 0), range=(-0.6, 0.6), stiffness=k_rot, damping=c_rot, tag="suspension",
                                     leg=wn)],
                       geoms=[Geom(an, "box", (L / 2, h / 2, p["arm_width"] * mm / 2),
                                   pos=(ax[0] / 2, 0.0, ax[2] / 2), quat=(math.cos(sx * ang / 2), 0.0, math.sin(sx * ang / 2), 0.0),
                                   mass=m_arm, role="link", rgba=rgba)],
                       children=[wheel])
            children.append(arm)
            feet.append(FootSpec(wn, wn, [an, wn], "chassis"))
    geoms = [Geom("plate", "box", (Lc / 2, Wc / 2, tp / 2), pos=(0, 0, -tp / 2), mass=m_chassis, role="body", rgba=rgba)]
    for sy in (1, -1):
        geoms.append(Geom(f"rail_{'L' if sy > 0 else 'R'}", "box", ((Lc - 20 * mm) / 2, p["rail_thickness"] * mm / 2, rh / 2),
                          pos=(0, sy * (Wc / 2 - p["rail_thickness"] * mm / 2 - 2 * mm), -tp - rh / 2), role="visual",
                          rgba=rgba))
    masses = [PointMass(k, v / 1000, (0.0, 0.0, -tp - rh / 2)) for k, v in PARTS_G.items()
              if k not in ("wheels + tyres 4x", "motors + gearboxes 4x")]
    robot = Robot(name, Link("chassis", log=True, geoms=geoms, masses=masses, children=children), feet=feet,
                  notes="notebook 11's four-wheel rover: tub chassis, trailing arms on sprung pivots, driven wheels",
                  sources={"geometry": "designs/rover.py", "masses": "notebook 11 parts list, CAD volumes × PA12-CF",
                           "suspension": "notebook 11 K_SUSP, ZETA", "drive": "notebook 11 T_STALL; ASSUMPTIONS"})
    robot.validate()
    robot.params = p
    return robot
