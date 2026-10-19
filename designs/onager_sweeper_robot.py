"""Onager Sweeper (notebook 23) as a Chiron robot: the Sentinel chassis (``onager_robot.onager``) set low in a fixed
stance, the Manus's two arms (``onager_manus_robot._arm``, longer links, the pedestals turning freely so a pincer
reaches the basket behind it), and the sweeper's own parts: the rotary disc broom under the hull centre (a hinge about z
driven by the catalogue's ``broom drive 1.5 kW`` as a velocity servo), the suction hood behind it (hull geoms: its
lips 25 mm over the road, the hood collides with litter), the basket on the back of the roof (a welded link of five
plates: whatever the pincers drop in stays in) and the display post on the nose. The suction itself is not a
MuJoCo contact: the scene (``onager_sweeper_scenario.Vacuum``) applies the CFD's air drag to the litter under the
hood (``onager_sweeper_cfd.SUCTION``).

=================  ==========================  ==========================================================================
joint              drive                       convention
=================  ==========================  ==========================================================================
``broom``          broom drive 1.5 kW          hinge about +z at the disc centre, unlimited; a velocity servo (``kp = 0``)
                                               like the hub motors: the controller sets ω [rad/s], + is counter-clockwise
                                               seen from above
``<s>_*``          the Manus's arm joints      ``onager_manus_robot``; the yaw is unlimited (slip ring)
``<leg>_*``        the Sentinel's legs/wheels  ``onager_robot``; leg modules stiff and braked (the fixed stance)
=================  ==========================  ==========================================================================

    import onager_sweeper_robot as osr
    robot = osr.sweeper()                     # ~640 kg
    lab = osr.sweeper_lab(terrain, props=...)
"""
from __future__ import annotations


from vegeta.chiron import ChironLab, Geom, Joint, Link, PointMass, Robot, Servo

import actuators as act
import onager_manus_robot as omr
import onager_robot as orb

__all__ = ["SWEEPER", "CHASSIS", "ARMS", "CAD", "MATERIALS", "PARTS_KG", "SWEEPER_KG", "BROOM_DRIVE", "FAN_MOTOR",
           "BROOM_RPM", "FAN_FLOW", "LEG_KP", "LEG_KD", "YAW_RANGE_DEG", "ASSUMPTIONS", "LAB_OPTIONS", "design_params",
           "cad_numbers", "sweeper_geometry", "sweeper_masses", "mass_budget", "broom_servo", "nominal_qpos", "sweeper",
           "sweeper_lab", "STOW"]

#: Sweeper parameters beyond the Manus's (onager_sweeper.py defaults, mm).
SWEEPER = {
    "display_x": 900.0, "broom_x": 350.0, "broom_diameter": 600.0, "broom_disc_thickness": 20.0, "bristle_length": 60.0,
    "broom_housing_diameter": 200.0, "hood_x": -100.0, "hood_width": 500.0, "hood_length": 250.0, "hood_height": 170.0,
    "hood_gap": 30.0, "hood_wall": 6.0, "duct_inner": 140.0, "duct_wall": 10.0, "basket_x": -650.0,
    "basket_length": 700.0, "basket_width": 900.0, "basket_height": 400.0, "basket_wall": 2.0,
}
#: The Sentinel parameters the Sweeper changes (onager_sweeper.SWEEPER_CHASSIS) ...
CHASSIS = {"hull_length": 2200.0, "hull_width": 1000.0, "hull_height": 650.0, "hull_chamfer": 150.0, "hull_bottom": 300.0,
           "shoulder_x": 950.0, "upper_leg_length": 420.0, "lower_leg_length": 420.0, "hip_angle_deg": 40.0,
           "knee_angle_deg": 50.0, "turret_length": 0.0, "turret_height": 0.0, "mast_height": 250.0, "sensor_head": 150.0}
#: ... and the Manus arm parameters it changes.
ARMS = {"arm_x": 600.0, "arm_y": 300.0, "upper_arm_length": 700.0, "forearm_length": 650.0}

#: CAD of the default Sweeper (notebook 23 §1; mm², mm³). ``cad_numbers(recompute=True)`` rebuilds them.
CAD = {"hull_surface_area": 13225120.0, "upper_arm_volume": 1729814.0, "forearm_volume": 1148420.0, "jaw_volume": 127929.0,
       "broom_disc_volume": 5273100.0, "hood_volume": 2755648.0, "basket_volume": 3997632.0}
MATERIALS = {"PA6 broom disc": 1.15e-6, "Al 5754 sheet (hood, duct, basket)": 2.66e-6}

PARTS_KG = dict(orb.PARTS_KG)
PARTS_KG.pop("sensors: E/O-IR head, lidar, acoustic, mast")
PARTS_KG["sensors: display head, lidar, cameras"] = 10.0
PARTS_KG["frame, mounts, hatches"] = 36.0                   # the longer body
PARTS_KG["wiring, connectors, cooling"] = 18.0
#: The sweeper's listed parts (kg); the disc, hood and basket come from the CAD.
SWEEPER_KG = {"bristle ring (PP tufts)": 3.0, "broom mount and bearing": 4.0, "fan impeller and housing": 9.0,
              "ducting to the fan": 4.0, "hopper 60 L with filter": 14.0, "basket mounts": 3.0}
BROOM_DRIVE, FAN_MOTOR = "broom drive 1.5 kW", "suction fan 4 kW"
BROOM_RPM = 150.0                                          # the broom's working speed (input: brush-disc sweepers 100-200 rpm)
FAN_FLOW = 0.35                                            # m³/s the fan draws (input: 1260 m³/h, a compact sweeper class)
LEG_KP, LEG_KD = 6000.0, 400.0
YAW_RANGE_DEG = None                                       # unlimited: slip-ring pedestals
BROOM_CLEARANCE = 0.035                                    # m: the bristle cylinder over the road in MuJoCo (small litter passes under it)

ASSUMPTIONS = {
    "fixed_stance": "no active suspension: the leg modules' brakes hold the stance; in MuJoCo the leg servos are "
                    "stiff (kp 6000 N·m/rad: a leg's compliance at the wheel ~30 kN/m, the structure's and the "
                    "pins', not a control loop) and four-quadrant (braking without current)",
    "broom": "the disc and its bristles are one cylinder 35 mm over the road: the bristles' compliance and their "
             "combing of small litter are not modelled — the cylinder sweeps what stands taller than 35 mm and small "
             "litter passes under it to the hood, as it does through a bristle skirt; μ 0.8 on litter (PP bristles); the drive is a "
             "velocity servo kd 5 N·m·s/rad with the disc's ½ m r² and the rotor through 1:12 (0.02 kg·m²)",
    "hood": "five Al plates (four walls, the roof) under the hull, the front lip a rubber flap that bulky litter "
            "pushes through (it does not collide); the duct inside the hull is a listed mass; the suction is the "
            "scene's air drag on the litter (onager_sweeper_scenario.Vacuum), from the CFD",
    "basket": "a welded sheet-metal box on the roof as five plates; the basket mounts are a hull mass",
    "fan_and_hopper": "point masses in the rear of the hull (the duct rises behind the hood)",
    "shell": "the series' 3 mm Al 5083 skin on the CAD's area (104 kg): a 2 mm skin on a frame would save ~35 kg",
    "yaw_range": "unlimited (a slip ring in the pedestal: the arm swings through 180° to the basket); the arms do "
                 "not collide with each other, the basket or the display post (Chiron's robots have no "
                 "self-collision — the mission's poses keep clear of them)",
}

LAB_OPTIONS = dict(omr.LAB_OPTIONS)
STOW = omr.STOW


def design_params(overrides: dict | None = None) -> dict:
    """The Manus's parameters with the Sweeper's chassis and arm changes, plus the sweeper parts [mm, deg]."""
    over = dict(overrides or {})
    sw = {k: over.pop(k) for k in list(over) if k in SWEEPER}
    p = omr.design_params({**CHASSIS, **ARMS, **over})
    q = dict(SWEEPER)
    q.update({k: float(v) for k, v in sw.items()})
    return {**p, **q}


def cad_numbers(p: dict | None = None, *, recompute: bool = False) -> dict:
    """Hull area and part volumes (mm², mm³): the stored ``CAD`` for the default design, else rebuilt with
    ``onager_sweeper.OnagerSweeper`` (CadQuery)."""
    p = design_params() if p is None else dict(p)
    sentinel = orb.cad_numbers(orb.design_params({k: p[k] for k in orb.DESIGN}), recompute=recompute)
    if not recompute and p == design_params():
        return {**sentinel, **CAD}
    import onager_sweeper

    d = onager_sweeper.OnagerSweeper()
    over = {k: v for k, v in p.items() if k in {prm.name for prm in d.parameters}}
    vol = lambda part: d.generate(part=part, **over).measure()["volume"]          # noqa: E731
    return {**sentinel, "hull_surface_area": d.generate(part="hull", **over).measure()["surface_area"],
            "upper_arm_volume": vol("upper_arm"), "forearm_volume": vol("forearm"), "jaw_volume": vol("jaw"),
            "broom_disc_volume": vol("broom_disc"), "hood_volume": vol("hood_shell"), "basket_volume": vol("basket")}


def sweeper_geometry(p: dict | None = None) -> dict:
    """[m] in the hull frame (origin at the hull centre, standing): the broom, the hood box, the basket box, the
    roof; ``hull_z`` the hull centre over the road."""
    p = design_params() if p is None else p
    g = orb.geometry({k: p[k] for k in orb.DESIGN})
    mm = 1e-3
    hz = g["hull_z"]
    hb = (p["hood_x"] - p["hood_length"] / 2, -p["hood_width"] / 2, p["hood_gap"],
          p["hood_x"] + p["hood_length"] / 2, p["hood_width"] / 2, p["hood_height"])
    roof = p["hull_bottom"] + p["hull_height"]
    bb = (p["basket_x"] - p["basket_length"] / 2, -p["basket_width"] / 2, roof,
          p["basket_x"] + p["basket_length"] / 2, p["basket_width"] / 2, roof + p["basket_height"])
    return {"hull_z": hz, "roof_z": roof * mm - hz, "broom_x": p["broom_x"] * mm, "broom_r": p["broom_diameter"] / 2 * mm,
            "broom_h": (p["bristle_length"] + p["broom_disc_thickness"]) * mm, "broom_clearance": BROOM_CLEARANCE,
            "housing_r": p["broom_housing_diameter"] / 2 * mm, "floor_z": p["hull_bottom"] * mm - hz,
            "hood_world": tuple(v * mm for v in hb), "hood": tuple(v * mm - (hz if k % 3 == 2 else 0.0) for k, v in enumerate(hb)),
            "hood_wall": p["hood_wall"] * mm, "duct_inner": p["duct_inner"] * mm,
            "basket_world": tuple(v * mm for v in bb), "basket": tuple(v * mm - (hz if k % 3 == 2 else 0.0) for k, v in enumerate(bb)),
            "basket_wall": p["basket_wall"] * mm, "display_x": p["display_x"] * mm}


def sweeper_masses(p: dict | None = None, cad: dict | None = None) -> dict:
    """The sweeper's own parts [kg]: CAD volumes × densities, the catalogue's drives, the listed parts."""
    p = design_params() if p is None else p
    cad = cad_numbers(p) if cad is None else cad
    out = {"broom disc (PA6, CAD)": cad["broom_disc_volume"] * MATERIALS["PA6 broom disc"],
           "hood and duct (Al 5754, CAD)": cad["hood_volume"] * MATERIALS["Al 5754 sheet (hood, duct, basket)"],
           "basket (Al 5754, CAD)": cad["basket_volume"] * MATERIALS["Al 5754 sheet (hood, duct, basket)"],
           f"broom drive ({BROOM_DRIVE})": act.get(BROOM_DRIVE).mass_g / 1000.0,
           f"fan motor ({FAN_MOTOR})": act.get(FAN_MOTOR).mass_g / 1000.0}
    out.update(SWEEPER_KG)
    return out


def mass_budget(p: dict | None = None, cad: dict | None = None) -> dict:
    """The whole Sweeper [kg]: the chassis (``onager_robot.mass_budget`` with the Sweeper's parts), two arms, the
    sweeping gear."""
    p = design_params() if p is None else p
    cad = cad_numbers(p) if cad is None else cad
    out = orb.mass_budget({k: p[k] for k in orb.DESIGN}, cad, PARTS_KG)
    out["arms 2x (see onager_manus_robot.arm_masses)"] = 2 * sum(omr.arm_masses(p, cad).values())
    out["sweeping gear (see sweeper_masses)"] = sum(sweeper_masses(p, cad).values())
    return out


def broom_servo(kd: float = 5.0) -> Servo:
    """The broom drive as a velocity servo (``kp = 0``) on its torque–speed line, with the rotor's reflected inertia."""
    return Servo.from_actuator(act.get(BROOM_DRIVE), kp=0.0, kd=kd, armature=0.02)


def nominal_qpos(p: dict | None = None) -> dict:
    q = omr.nominal_qpos(p or design_params())
    q["broom"] = 0.0
    return q


def _sweeping_gear(p: dict, cad: dict, g: dict, litter_mu: float) -> tuple:
    """The broom link, the basket link, the hood geoms and the hull point masses; (links, hull geoms, hull masses,
    total mass, Σ m x)."""
    m = sweeper_masses(p, cad)
    broom_rgba, hood_rgba, basket_rgba = (0.12, 0.12, 0.14, 1.0), (0.3, 0.32, 0.34, 1.0), (0.78, 0.8, 0.82, 1.0)
    fr = (litter_mu, 0.005, 0.0001)
    # the broom: one cylinder (disc + bristles), its bottom BROOM_CLEARANCE over the road, spinning about z
    z_bot = -g["hull_z"] + g["broom_clearance"]
    m_broom = m["broom disc (PA6, CAD)"] + m["bristle ring (PP tufts)"]
    broom = Link("broom_link", pos=(g["broom_x"], 0.0, z_bot + g["broom_h"] / 2),
                 joints=[Joint("broom", axis=(0, 0, 1), range=None, tag="broom", servo=broom_servo())],
                 geoms=[Geom("broom", "cylinder", (g["broom_r"], g["broom_h"] / 2), mass=m_broom, role="link",
                             friction=fr, rgba=broom_rgba)])
    # the basket: floor and four walls, welded to the hull
    x0, y0, z0, x1, y1, z1 = g["basket"]
    t = g["basket_wall"]
    L, W, H = x1 - x0, y1 - y0, z1 - z0
    m_bask = m["basket (Al 5754, CAD)"]
    a_floor, a_wall_x, a_wall_y = L * W, W * H, L * H
    a_tot = a_floor + 2 * a_wall_x + 2 * a_wall_y
    bg = [Geom("basket_floor", "box", (L / 2, W / 2, t), pos=(0.0, 0.0, t), mass=m_bask * a_floor / a_tot, role="link",
               friction=fr, rgba=basket_rgba),
          Geom("basket_front", "box", (t, W / 2, H / 2), pos=(L / 2 - t, 0.0, H / 2), mass=m_bask * a_wall_x / a_tot, role="link",
               friction=fr, rgba=basket_rgba),
          Geom("basket_back", "box", (t, W / 2, H / 2), pos=(-L / 2 + t, 0.0, H / 2), mass=m_bask * a_wall_x / a_tot, role="link",
               friction=fr, rgba=basket_rgba),
          Geom("basket_left", "box", (L / 2, t, H / 2), pos=(0.0, W / 2 - t, H / 2), mass=m_bask * a_wall_y / a_tot, role="link",
               friction=fr, rgba=basket_rgba),
          Geom("basket_right", "box", (L / 2, t, H / 2), pos=(0.0, -W / 2 + t, H / 2), mass=m_bask * a_wall_y / a_tot, role="link",
               friction=fr, rgba=basket_rgba)]
    basket = Link("basket", pos=((x0 + x1) / 2, 0.0, z0), geoms=bg)
    # the hood: four walls and the roof under the hull floor (hull geoms), the lips hood_gap over the road
    hx0, hy0, hz0, hx1, hy1, hz1 = g["hood"]
    hw = g["hood_wall"]
    hL, hW, hH = hx1 - hx0, hy1 - hy0, hz1 - hz0
    m_hood = m["hood and duct (Al 5754, CAD)"]
    hc = ((hx0 + hx1) / 2, 0.0)
    hood_geoms = [Geom("hood_roof", "box", (hL / 2, hW / 2, hw / 2), pos=(hc[0], 0.0, hz1 - hw / 2), mass=m_hood * 0.4, role="link",
                       friction=fr, rgba=hood_rgba),
                  # the front lip is a rubber flap: litter pushes through it (no collision), the side and back lips are rigid
                  Geom("hood_front", "box", (hw / 2, hW / 2, hH / 2), pos=(hx1 - hw / 2, 0.0, (hz0 + hz1) / 2), mass=m_hood * 0.15,
                       role="visual", rgba=(0.1, 0.1, 0.1, 1.0)),
                  Geom("hood_back", "box", (hw / 2, hW / 2, hH / 2), pos=(hx0 + hw / 2, 0.0, (hz0 + hz1) / 2), mass=m_hood * 0.15,
                       role="link", friction=fr, rgba=hood_rgba),
                  Geom("hood_left", "box", (hL / 2, hw / 2, hH / 2), pos=(hc[0], hy1 - hw / 2, (hz0 + hz1) / 2), mass=m_hood * 0.15,
                       role="link", friction=fr, rgba=hood_rgba),
                  Geom("hood_right", "box", (hL / 2, hw / 2, hH / 2), pos=(hc[0], hy0 + hw / 2, (hz0 + hz1) / 2), mass=m_hood * 0.15,
                       role="link", friction=fr, rgba=hood_rgba),
                  Geom("broom_housing", "cylinder", (g["housing_r"], (g["floor_z"] - (z_bot + g["broom_h"])) / 2),
                       pos=(g["broom_x"], 0.0, (g["floor_z"] + z_bot + g["broom_h"]) / 2), role="visual", rgba=hood_rgba)]
    fz = g["floor_z"]
    masses = [PointMass("broom drive", m[f"broom drive ({BROOM_DRIVE})"], (g["broom_x"], 0.0, fz + 0.1)),
              PointMass("broom mount and bearing", m["broom mount and bearing"], (g["broom_x"], 0.0, fz + 0.02)),
              PointMass("fan motor", m[f"fan motor ({FAN_MOTOR})"], (hc[0] - 0.25, 0.0, fz + 0.25)),
              PointMass("fan impeller and housing", m["fan impeller and housing"], (hc[0] - 0.25, 0.0, fz + 0.25)),
              PointMass("ducting to the fan", m["ducting to the fan"], (hc[0] - 0.1, 0.0, fz + 0.15)),
              PointMass("hopper 60 L with filter", m["hopper 60 L with filter"], (hc[0] - 0.55, 0.0, fz + 0.3)),
              PointMass("basket mounts", m["basket mounts"], ((x0 + x1) / 2, 0.0, z0 - 0.02))]
    total = sum(m.values())
    moment = (m_broom * g["broom_x"] + m_bask * (x0 + x1) / 2 + m_hood * hc[0] + sum(pm.mass * pm.pos[0] for pm in masses))
    return [broom, basket], hood_geoms, masses, total, moment


def sweeper(overrides: dict | None = None, *, jaw_mu: float = 0.6, litter_mu: float = 0.8, cad: dict | None = None,
            name: str = "Onager Sweeper", four_quadrant: bool = True, kp: float = LEG_KP, kd: float = LEG_KD,
            **chassis_kw) -> Robot:
    """Onager Sweeper as a Chiron ``Robot``: ``onager_robot.onager`` (low fixed stance, no turret, the display post
    on the nose) with two Manus arms, the broom, the hood and the basket. ``litter_mu``: the broom's and the
    hood's friction on litter."""
    p = design_params(overrides)
    cad = cad_numbers(p) if cad is None else cad
    g = sweeper_geometry(p)
    arms, hull_masses, m_extra, moment = [], [], 0.0, 0.0
    for side in omr.SIDES:
        base, yaw_mass, m_arm, mom = omr._arm(side, p, cad, jaw_mu, yaw_range_deg=YAW_RANGE_DEG)
        arms.append(base)
        hull_masses.append(yaw_mass)
        m_extra += m_arm
        moment += mom
    links, hood_geoms, masses, m_gear, mom_gear = _sweeping_gear(p, cad, g, litter_mu)
    robot = orb.onager({k: p[k] for k in orb.DESIGN if p[k] != orb.DESIGN[k]}, cad=cad, name=name, parts_kg=PARTS_KG,
                       extra_children=arms + links, extra_mass_kg=m_extra + m_gear, extra_moment_kgm=moment + mom_gear,
                       extra_hull_masses=hull_masses + masses, extra_hull_geoms=hood_geoms, mast_x=g["display_x"],
                       kp=kp, kd=kd, four_quadrant=four_quadrant,
                       notes=f"Onager Sweeper: the Sentinel chassis low in a fixed stance, two Manus arms, a disc broom "
                             f"({BROOM_DRIVE}), a suction hood (the fan: {FAN_MOTOR}, {FAN_FLOW} m³/s), a basket on the roof",
                       sources={"geometry": "designs/onager_sweeper.py", "masses": "notebook 23 §1 mass budget",
                                "actuators": "designs/actuators.py", "assumptions": "onager_sweeper_robot.ASSUMPTIONS",
                                "suction": "onager_sweeper_cfd.SUCTION (notebook 23 §4, OpenFOAM)"},
                       **chassis_kw)
    robot.nominal_qpos = nominal_qpos(p)
    robot.validate()
    robot.params = p
    robot.arm = omr.arm_geometry(p)
    robot.sweeper = g
    return robot


def sweeper_lab(terrain=None, robot: Robot | None = None, **kwargs) -> ChironLab:
    """``ChironLab(sweeper(), terrain, **LAB_OPTIONS)``; ``kwargs`` override (``props``, ``log_geoms``...)."""
    opts = dict(LAB_OPTIONS)
    opts.update(kwargs)
    return ChironLab(robot or sweeper(), terrain, **opts)
