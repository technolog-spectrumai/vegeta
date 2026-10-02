"""Onager Atlas (notebook 21) as a Chiron robot: the Sentinel's chassis (``onager_robot.onager``) in the logistics
stance (shoulders 25°, knees 15°: the legs nearly straight under the load), without the turret, with a rear
counterweight and a forklift on the nose.

=================  ==========================  =========================================================================
joint              drive                       convention
=================  ==========================  =========================================================================
``mast_tilt``      tilt screw 12 kN on a       hinge about −y at the pivot under the nose: + tilts the mast back (the
                   0.30 m lever (3.6 kN·m)     load leans against the carriage), range −5° … +12°
``lift``           lift screw 8 kN             slide along the mast: the carriage's height above its lowest position
                                               [m], 0 … ``lift_max``; the forks' undersides are ``fork_ground`` above
                                               the ground at 0 (mast vertical, standing)
=================  ==========================  =========================================================================

The screws are linear drives (``actuators.LINEAR``); the tilt screw acts on the mast 0.30 m above the pivot, so it
is given to Chiron as a hinge servo with the equivalent torque and speed (12 kN × 0.30 m, 60 mm/s ÷ 0.30 m). The
lift is a slide joint whose servo works in newtons and m/s. Both have holding brakes (self-locking): standing still
the load costs no current. The forks collide with props (the pallet); the mast and carriage do too.

    import onager_atlas_robot as oar
    robot = oar.atlas()
    lab = oar.atlas_lab(terrain, props=..., )
"""
from __future__ import annotations

import math

from vegeta.chiron import ChironLab, Geom, Joint, Link, PointMass, Robot, Servo

import actuators as act
import onager_robot as orb

__all__ = ["ATLAS", "CHASSIS", "CAD", "PARTS_KG", "FORKLIFT_KG", "COUNTERWEIGHT", "LIFT", "TILT", "TILT_LEVER",
           "ASSUMPTIONS", "LAB_OPTIONS", "design_params", "cad_numbers", "forklift_geometry", "forklift_masses",
           "mass_budget", "atlas", "atlas_lab", "nominal_qpos", "lift_servo", "tilt_servo"]

#: Atlas parameters beyond the Sentinel's (onager_atlas.py defaults, mm and degrees).
ATLAS = {
    "pivot_x": 1200.0, "pivot_z": 1065.0, "upright_y": 320.0, "upright_width": 60.0, "upright_depth": 100.0,
    "upright_wall": 6.0, "mast_bottom": -1005.0, "mast_top": 1020.0, "carriage_offset": 80.0, "carriage_width": 760.0,
    "carriage_height": 450.0, "carriage_thickness": 30.0, "fork_y": 175.0, "fork_length": 1000.0, "fork_width": 80.0,
    "fork_thickness": 30.0, "fork_shank": 450.0, "fork_ground": 40.0, "lift_max": 1200.0, "lift": 150.0,
    "tilt_deg": 0.0,
}
#: The Sentinel parameters the Atlas changes (onager_atlas.ATLAS_CHASSIS).
CHASSIS = {"turret_length": 0.0, "turret_height": 0.0, "hip_angle_deg": 25.0, "knee_angle_deg": 15.0,
           "hull_bottom": 1045.0}

#: CAD of the default Atlas (notebook 21 §1; mm², mm³). ``cad_numbers(recompute=True)`` rebuilds them.
CAD = {"hull_surface_area": 6291129.0, "fork_volume": 3444715.0, "mast_volume": 10409363.0}
STEEL = 7.85e-6                                      # kg/mm³: S355 mast, S690 forks

PARTS_KG = dict(orb.PARTS_KG)
PARTS_KG.pop("sensors: E/O-IR head, lidar, acoustic, mast")
PARTS_KG["sensors: mast head, lidar, fork cameras"] = 14.0
#: The forklift's listed parts (kg); the forks and the mast come from the CAD.
FORKLIFT_KG = {"carriage frame (welded)": 30.0, "mast mount arms": 12.0, "lift screw 8 kN": 12.0,
               "tilt screws 2x (12 kN)": 12.0, "lift chain, rollers, guards": 8.0}
#: A cast-iron slab under the rear of the hull: a forklift's counterweight (kg, position in the hull frame [m]).
COUNTERWEIGHT = {"mass_kg": 120.0, "x": -0.85, "z": -0.22}
LIFT, TILT = "lift screw 8 kN", "tilt screw 12 kN"
TILT_LEVER = 0.30                                    # m: the tilt screws act 0.30 m above the pivot

ASSUMPTIONS = {
    "lift_servo": "kp 4e5 N/m (2.5 mm sag under a 200 kg pallet without feed-forward), kd 2e4 N·s/m, reflected mass "
                  "50 kg (a 2 kg·cm² rotor through a 10 mm lead); the controller adds the carriage's weight",
    "tilt_servo": "kp 2e5 N·m/rad, kd 1e4, armature 20 kg·m² (rotor through a 5 mm lead on the 0.30 m lever)",
    "four_quadrant": "every Atlas drive (screws, leg modules, hub motors) brakes with its full force when backdriven "
                     "(braked screws, regenerating motors: chiron.Servo.four_quadrant); the original law lets a "
                     "jolted mast fall once it passes the screw's no-load speed (notebook 21 §7)",
    "counterweight": "120 kg cast iron under the rear of the hull, chosen so the unloaded robot's CG sits near the "
                     "hull centre and a 200 kg pallet at 500 mm leaves a tipping margin ≥ 1.5 (notebook 21 §2)",
    "forks": "S690 steel tines, boxes of the CAD's section; the tip taper is not modelled in MuJoCo",
    "carriage": "a 30 kg welded frame (the CAD's plate is its envelope)",
}

LAB_OPTIONS = dict(orb.LAB_OPTIONS, timestep=0.0005, control_dt=0.002, log_dt=0.02)


def design_params(overrides: dict | None = None) -> dict:
    over = dict(overrides or {})
    fl = {k: over.pop(k) for k in list(over) if k in ATLAS}
    p = orb.design_params({**CHASSIS, **over})
    q = dict(ATLAS)
    q.update({k: float(v) for k, v in fl.items()})
    return {**p, **q}


def cad_numbers(p: dict | None = None, *, recompute: bool = False) -> dict:
    p = design_params() if p is None else dict(p)
    sentinel = orb.cad_numbers(orb.design_params({k: p[k] for k in orb.DESIGN}), recompute=recompute)
    if not recompute and p == design_params():
        return {**sentinel, **CAD}
    import onager_atlas

    d = onager_atlas.OnagerAtlas()
    over = {k: v for k, v in p.items() if k in {prm.name for prm in d.parameters}}
    return {**sentinel, "hull_surface_area": d.generate(part="hull", **over).measure()["surface_area"],
            "fork_volume": d.generate(part="fork", **over).measure()["volume"],
            "mast_volume": d.generate(part="mast", **over).measure()["volume"]}


def forklift_geometry(p: dict | None = None) -> dict:
    """[m] in the hull frame (origin at the hull centre, standing): the pivot, the carriage, the forks."""
    p = design_params() if p is None else p
    g = orb.geometry({k: p[k] for k in orb.DESIGN})
    mm = 1e-3
    xc = (p["upright_depth"] / 2 + p["carriage_offset"]) * mm
    heel = xc + (p["carriage_thickness"] / 2 + p["fork_thickness"]) * mm
    return {"pivot": (p["pivot_x"] * mm, 0.0, p["pivot_z"] * mm - g["hull_z"]), "pivot_z_world": p["pivot_z"] * mm,
            "carriage_x": xc, "heel_x": heel, "fork_y": p["fork_y"] * mm, "fork_L": p["fork_length"] * mm,
            "fork_w": p["fork_width"] * mm, "fork_t": p["fork_thickness"] * mm, "shank": p["fork_shank"] * mm,
            "fork_z0": (p["fork_ground"] - p["pivot_z"]) * mm, "lift_max": p["lift_max"] * mm,
            "mast_z": (p["mast_bottom"] * mm, p["mast_top"] * mm), "upright_y": p["upright_y"] * mm,
            "upright": (p["upright_depth"] * mm, p["upright_width"] * mm),
            "carriage": (p["carriage_thickness"] * mm, p["carriage_width"] * mm, p["carriage_height"] * mm),
            "heel_world_x": p["pivot_x"] * mm + heel}


def forklift_masses(p: dict | None = None, cad: dict | None = None) -> dict:
    p = design_params() if p is None else p
    cad = cad_numbers(p) if cad is None else cad
    out = {"forks 2x (S690, CAD)": 2 * cad["fork_volume"] * STEEL, "mast (S355 uprights, CAD)": cad["mast_volume"] * STEEL}
    out.update(FORKLIFT_KG)
    return out


def mass_budget(p: dict | None = None, cad: dict | None = None) -> dict:
    p = design_params() if p is None else p
    cad = cad_numbers(p) if cad is None else cad
    out = orb.mass_budget({k: p[k] for k in orb.DESIGN}, cad, PARTS_KG)
    out["forklift (see forklift_masses)"] = sum(forklift_masses(p, cad).values())
    out["counterweight (cast iron, rear)"] = COUNTERWEIGHT["mass_kg"]
    return out


def lift_servo() -> Servo:
    a = act.get_linear(LIFT)
    return Servo(stall_torque=a.stall_N, rated_torque=a.rated_N, no_load_speed=a.no_load_mm_s / 1000.0,
                 stall_current=a.stall_A, voltage=a.voltage_V, kp=4e5, kd=2e4, armature=50.0,
                 source=f"{a.key}: {a.source} (a slide joint: N and m/s)", four_quadrant=True)


def tilt_servo() -> Servo:
    a = act.get_linear(TILT)
    return Servo(stall_torque=a.stall_N * TILT_LEVER, rated_torque=a.rated_N * TILT_LEVER,
                 no_load_speed=a.no_load_mm_s / 1000.0 / TILT_LEVER, stall_current=a.stall_A, voltage=a.voltage_V,
                 kp=2e5, kd=1e4, armature=20.0, source=f"{a.key} on a {TILT_LEVER:.2f} m lever: {a.source}",
                 four_quadrant=True)


def nominal_qpos(p: dict | None = None, lift: float = 0.0, tilt: float = 0.0) -> dict:
    q = orb.nominal_qpos({k: v for k, v in (p or design_params()).items() if k in orb.DESIGN})
    q.update({"mast_tilt": tilt, "lift": lift})
    return q


def _forklift(p: dict, cad: dict, fork_mu: float) -> tuple:
    f = forklift_geometry(p)
    m = forklift_masses(p, cad)
    steel, fork_rgba = (0.85, 0.62, 0.12, 1.0), (0.25, 0.25, 0.25, 1.0)
    fr = (0.5, 0.005, 0.0001)
    fork_fr = (fork_mu, 0.005, 0.0001)
    L, w, t, h = f["fork_L"], f["fork_w"], f["fork_t"], f["shank"]
    m_fork = m["forks 2x (S690, CAD)"] / 2
    m_tine, m_shank = m_fork * L / (L + h), m_fork * h / (L + h)
    ct, cw, ch_ = f["carriage"]
    z0 = f["fork_z0"]                                                  # forks' underside, mast frame, lift 0
    geoms = [Geom("carriage", "box", (ct / 2, cw / 2, ch_ / 2), pos=(f["carriage_x"], 0.0, z0 + ch_ / 2),
                  mass=m["carriage frame (welded)"], role="link", friction=fr, rgba=steel)]
    for side, sy in (("L", 1), ("R", -1)):
        y = sy * f["fork_y"]
        xs = f["heel_x"] - t / 2
        geoms.append(Geom(f"fork_{side}_shank", "box", (t / 2, w / 2, (h + t) / 2), pos=(xs, y, z0 + (h + t) / 2),
                          mass=m_shank, role="link", friction=fork_fr, rgba=fork_rgba))
        geoms.append(Geom(f"fork_{side}", "box", (L / 2, w / 2, t / 2), pos=(f["heel_x"] + L / 2, y, z0 + t / 2),
                          mass=m_tine, role="link", friction=fork_fr, rgba=fork_rgba))
    carriage = Link("carriage", joints=[Joint("lift", kind="slide", axis=(0, 0, 1), range=(0.0, f["lift_max"]),
                                              tag="lift", servo=lift_servo())],
                    geoms=geoms, masses=[PointMass("lift chain, rollers, guards", m["lift chain, rollers, guards"],
                                                   (0.0, 0.0, z0 + ch_ / 2))])
    zb, zt = f["mast_z"]
    ud, uw = f["upright"]
    mast_geoms = [Geom(f"upright_{s}", "box", (ud / 2, uw / 2, (zt - zb) / 2), pos=(0.0, sy * f["upright_y"], (zb + zt) / 2),
                       mass=m["mast (S355 uprights, CAD)"] / 2, role="link", friction=fr, rgba=steel)
                  for s, sy in (("L", 1), ("R", -1))]
    mast_geoms.append(Geom("mast_crossbar", "box", (ud * 0.4, f["upright_y"] + uw / 2, 0.01), pos=(0.0, 0.0, zt - 0.04),
                           role="visual", rgba=steel))
    mast = Link("mast", pos=f["pivot"],
                joints=[Joint("mast_tilt", axis=(0, -1, 0), range=(math.radians(-5), math.radians(12)), tag="tilt",
                              servo=tilt_servo())],
                geoms=mast_geoms, masses=[PointMass("lift screw", m["lift screw 8 kN"], (0.0, 0.0, 0.0))],
                children=[carriage])
    total = sum(m.values()) - m["mast mount arms"] - m["tilt screws 2x (12 kN)"]       # those two sit on the hull
    moment = (sum(m.values()) - m["mast mount arms"] - m["tilt screws 2x (12 kN)"]) * (f["pivot"][0] + 0.15)
    hull_masses = [PointMass("mast mount arms", m["mast mount arms"], (f["pivot"][0] - 0.1, 0.0, f["pivot"][2])),
                   PointMass("tilt screws", m["tilt screws 2x (12 kN)"], (f["pivot"][0] - 0.15, 0.0, f["pivot"][2] + 0.3)),
                   PointMass("counterweight", COUNTERWEIGHT["mass_kg"], (COUNTERWEIGHT["x"], 0.0, COUNTERWEIGHT["z"]))]
    extra_mass = total + m["mast mount arms"] + m["tilt screws 2x (12 kN)"] + COUNTERWEIGHT["mass_kg"]
    moment += (m["mast mount arms"] * (f["pivot"][0] - 0.1) + m["tilt screws 2x (12 kN)"] * (f["pivot"][0] - 0.15)
               + COUNTERWEIGHT["mass_kg"] * COUNTERWEIGHT["x"])
    return mast, hull_masses, extra_mass, moment


def atlas(overrides: dict | None = None, *, fork_mu: float = 0.4, cad: dict | None = None,
          name: str = "Onager Atlas", four_quadrant: bool = True, **chassis_kw) -> Robot:
    """Onager Atlas as a Chiron ``Robot``: ``onager_robot.onager`` (logistics stance, no turret) with the forklift
    and the counterweight. ``fork_mu``: steel forks on pallet wood."""
    p = design_params(overrides)
    cad = cad_numbers(p) if cad is None else cad
    mast, hull_masses, extra_mass, moment = _forklift(p, cad, fork_mu)
    robot = orb.onager({k: p[k] for k in orb.DESIGN if p[k] != orb.DESIGN[k]}, cad=cad, name=name, parts_kg=PARTS_KG,
                       extra_children=[mast], extra_mass_kg=extra_mass, extra_moment_kgm=moment,
                       extra_hull_masses=hull_masses,
                       notes=f"Onager Atlas: the Sentinel chassis in the logistics stance (25°/15°), no turret, "
                             f"{COUNTERWEIGHT['mass_kg']:.0f} kg counterweight, a forklift ({LIFT}, {TILT})",
                       sources={"geometry": "designs/onager_atlas.py", "masses": "notebook 21 §1 mass budget",
                                "actuators": "designs/actuators.py", "assumptions": "onager_atlas_robot.ASSUMPTIONS"},
                       four_quadrant=four_quadrant, **chassis_kw)
    robot.nominal_qpos = nominal_qpos(p)
    robot.validate()
    robot.params = p
    robot.forklift = forklift_geometry(p)
    return robot


def atlas_lab(terrain=None, robot: Robot | None = None, **kwargs) -> ChironLab:
    opts = dict(LAB_OPTIONS)
    opts.update(kwargs)
    return ChironLab(robot or atlas(), terrain, **opts)
