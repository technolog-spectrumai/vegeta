"""Pekari Rover (notebook 30): the design's numbers — parameters, CAD measurements, materials, the mass budget, the
centre of gravity and the derived geometry in SI — for the track, gear and terrain calculations (``tracks``,
``gears``, ``terramechanics``). The Chiron (MuJoCo) model of the rover comes with the terrain race (todo 5j.7).

Every number comes from the design (``pekari_rover.py`` parameters, mm), from the CAD (areas and volumes ×
material densities), from the actuator catalogue (``actuators.py``: ``pekari drive 16:1 planetary``, one per track)
or from the listed parts; inputs the notebook does not fix are named in ``ASSUMPTIONS`` with the reason for each.

``pekari()`` builds the rover for Chiron (MuJoCo, notebook 30 §8). MuJoCo has no track, so each track is a row of
rollers under the belt's outer surface (the multi-roller approximation of todo 5j.7): the raised sprocket and idler,
the four road wheels and a belt roller midway between each pair — every 60 mm along the ground run. The road wheels
and the roller between them sit on the two bogies (passive hinges at the pivots, ±12°); the middle roller on the
frame. Every roller is a velocity servo commanded to the side's belt speed (ω = v / r). A real sprocket pulls the whole
belt, so its force goes wherever the belt touches; here each roller may carry a third of the side's gearmotor force
(``ROLLER_SHARE``): enough when only two or three rollers touch (on an obstacle), an overestimate of the drive's limit
only when many rollers saturate at once (a stalled rover on flat ground) — read the side forces against the gearmotor's
(``timeseries``). The track's internal loss is the rollers' joint friction loss.

    import pekari_rover_robot as prr
    budget = prr.mass_budget()            # part -> kg
    cg = prr.cg()                         # x, z of the empty and the loaded rover
    g = prr.geometry()                    # b, L, B, r_sprocket, ... [m]
"""
from __future__ import annotations

import math
from dataclasses import replace

from vegeta.chiron import ChironLab, FootSpec, Geom, Joint, Link, PointMass, Robot, Servo

import actuators as act

__all__ = ["DESIGN", "CAD", "MATERIALS", "PARTS_KG", "ASSUMPTIONS", "DRIVE", "MOTOR", "PAYLOAD_KG", "G",
           "WHEEL_FILL", "LAB_OPTIONS", "SIDES", "design_params", "pekari", "pekari_lab", "roller_joints", "rollers", "cad_numbers", "geometry", "mass_budget", "masses", "cg", "total_mass"]

G = 9.81
PAYLOAD_KG = 5.0
DRIVE = "pekari drive 16:1 planetary"                # one per track, in the hull's rear corners
MOTOR = "BLDC 100 W, 24 V"                           # the bare motor inside the drive (§5)

# ---- the design: pekari_rover.py's parameter defaults (mm; tests check they match PekariRover.parameters) ----
DESIGN = {
    "hull_length": 560.0, "hull_width": 300.0, "hull_height": 140.0, "hull_chamfer": 15.0, "ground_clearance": 90.0,
    "sensor_box": 80.0, "basket_length": 380.0, "basket_width": 260.0, "basket_height": 120.0, "basket_wall": 1.5,
    "basket_x": -70.0, "frame_offset": 56.0, "frame_thickness": 6.0, "track_width": 80.0, "track_pitch": 31.0,
    "link_thickness": 8.0, "grouser_height": 6.0, "guide_height": 12.0, "guide_width": 16.0, "pin_diameter": 5.0,
    "sprocket_teeth": 12, "sprocket_x": -290.0, "sprocket_z": 140.0, "idler_diameter": 110.0, "idler_x": 290.0,
    "idler_z": 110.0, "road_wheel_diameter": 70.0, "road_wheels": 4, "road_wheel_spacing": 120.0,
    "return_roller_diameter": 40.0, "return_roller_z": 162.0, "axle_diameter": 10.0, "sprocket_bore": 12.0,
    "pinion_module": 0.8, "pinion_teeth": 18, "pinion_width": 10.0, "pinion_bore": 5.0,
}

#: CAD measurements of the default design (Dedalus/CadQuery, notebook 30 §1; mm², mm³, mm); ``cad_numbers(
#: recompute=True)`` rebuilds them. Centres of mass in each part's own frame (the hull's in the vehicle's).
CAD = {
    "hull_surface_area": 581355.8,
    "hull_com": (3.63, 0.0, 161.45),
    "link_volume": 25768.4,
    "sprocket_volume": 295737.3,
    "idler_volume": 655493.3,
    "road_wheel_volume": 244573.0,
    "pinion_volume": 1384.4,
    "basket_volume": 401053.5,
    "basket_com": (0.0, 0.0, 42.41),
}

#: Densities [kg/mm³].
MATERIALS = {"Al 6061-T6 hull shell": 2.70e-6, "Al 5754 basket sheet": 2.66e-6, "PA6-GF30 links, sprockets, wheels": 1.36e-6,
             "steel pins (42CrMo4)": 7.85e-6}
SHELL_T_MM = 1.5                                     # hull shell: 1.5 mm Al sheet, bent and riveted (the CAD hull is solid)
WHEEL_FILL = 0.35                                    # wheels and sprockets are ribbed mouldings: 35 % of the solid volume
PARTS_KG = {
    "battery 24 V 10 Ah LiFePO4 (240 Wh)": 2.3,
    "computer, motor controllers, radio, IMU": 0.8,
    "sensors: stereo camera, 2D lidar (sensor box)": 0.9,
    "wiring, connectors, switches": 0.5,
    "hull frame, deck, fasteners": 1.2,
    "track frames, bogies, axles, bearings 2x": 1.6,
}

ASSUMPTIONS = {
    "class": "a ~25 kg loaded tracked rover (empty ~19 kg, 5 kg payload on the deck) in the size class of the "
             "wheeled rover of notebook 11 — the user's choice; the drive and track are sized for that",
    "hull": "1.5 mm Al 6061 sheet over the CAD hull's surface (sensor block included); an inner frame, deck and "
            "fasteners listed separately",
    "links": "PA6-GF30 (glass-filled nylon, injection moulded) links of the CAD volume, one steel pin per link "
             "(diameter pin_diameter, the track width long)",
    "wheels": "sprockets, idlers and road wheels are PA6-GF30 mouldings ribbed to WHEEL_FILL = 35 % of the solid CAD "
              "volume (a solid wheel would be 3x heavier than the ribbed wheels of small tracked robots); the "
              "return rollers' mass is in the listed 'track frames, bogies, axles, bearings'",
    "drive": "one 'pekari drive 16:1 planetary' per track (actuators.py), in the hull's rear corners next to its "
             "sprocket; the sprocket on the gearbox's output shaft",
    "battery_position": "on the hull floor at the x that puts the empty rover's CG over the centre of the road wheels "
                        "(the contact patch's centre), within the hull (|x| ≤ hull_length/2 − 100 mm)",
    "basket": "the payload basket on the roof is the default configuration (Pekari and the later Catagon both carry "
              "one): 1.5 mm Al 5754 sheet of the CAD volume",
    "payload_position": "a 5 kg payload in the basket, centred on its floor, its CG 50 mm above the floor",
    "rubber": "the links carry no rubber pads: PA6 grousers on soil, hard plastic on asphalt (μ 0.6 is the "
              "terramechanics table's rubber value — a pad would add it; to be measured)",
    "internal_loss": "f_in = 0.035 + 0.002 v (v m/s): a small plastic-link track with 70 mm road wheels and plain "
                     "PA6 hinges loses more than Wong's steel tracks (0.022 + 0.0003 V km/h); the number to measure first",
}


def design_params(overrides: dict | None = None) -> dict:
    """``DESIGN`` with ``overrides`` (mm)."""
    p = dict(DESIGN)
    for k, v in (overrides or {}).items():
        if k not in DESIGN:
            raise KeyError(f"unknown PekariRover parameter {k!r}")
        p[k] = v
    return p


def cad_numbers(p: dict | None = None, *, recompute: bool = False) -> dict:
    """The CAD areas and volumes the mass budget needs (see ``CAD``): stored for the default design, else rebuilt with
    ``pekari_rover.PekariRover`` (needs CadQuery)."""
    p = design_params() if p is None else dict(p)
    if not recompute and all(p.get(k) == v for k, v in DESIGN.items()):
        return dict(CAD)
    import pekari_rover

    design = pekari_rover.PekariRover()
    over = {k: v for k, v in p.items() if k in DESIGN}
    m = {k: design.generate(part=k, **over).measure() for k in ("hull", "track_link", "sprocket", "idler", "road_wheel", "pinion")}
    has_basket = p["basket_height"] > 0 and p["basket_length"] > 0 and p["basket_width"] > 0
    bk = design.generate(part="basket", **over).measure() if has_basket else {"volume": 0.0, "center_of_mass": (0.0, 0.0, 0.0)}
    return {"hull_surface_area": m["hull"]["surface_area"], "hull_com": tuple(m["hull"]["center_of_mass"]),
            "link_volume": m["track_link"]["volume"], "sprocket_volume": m["sprocket"]["volume"],
            "idler_volume": m["idler"]["volume"], "road_wheel_volume": m["road_wheel"]["volume"],
            "pinion_volume": m["pinion"]["volume"], "basket_volume": bk["volume"],
            "basket_com": tuple(bk["center_of_mass"])}


def _design():
    import pekari_rover

    return pekari_rover.PekariRover


def geometry(p: dict | None = None) -> dict:
    """Derived geometry [m]: ``b`` track width, ``L`` contact length, ``B`` gauge (track centres), ``pitch``,
    ``r_sprocket`` (pitch radius), ``z_teeth``, ``wheel_x`` road wheel axles, ``wheel_z``, ``d_wheel``, ``n_wheels``,
    ``belt_length`` (pin line), ``links`` per track, ``take_up`` (n p − belt), sprocket/idler axles, ``x_front`` /
    ``x_rear`` the ends of the track envelope, ``overall`` (length, width, height)."""
    p = design_params() if p is None else p
    D = _design()
    mm = 1e-3
    n_links, take_up = D.link_count(p)
    ov = D.overall(p)
    belt = [(x * mm, z * mm, r * mm) for x, z, r in D.belt_circles(p)]
    outer = p["link_thickness"] / 2 + p["grouser_height"]
    return {
        "b": p["track_width"] * mm, "L": D.contact_length(p) * mm, "B": D.track_gauge(p) * mm,
        "pitch": p["track_pitch"] * mm, "r_sprocket": D.sprocket_pitch_radius(p) * mm, "z_teeth": int(p["sprocket_teeth"]),
        "wheel_x": [x * mm for x in D.road_wheel_x(p)], "wheel_z": D.road_wheel_z(p) * mm,
        "d_wheel": p["road_wheel_diameter"] * mm, "n_wheels": int(p["road_wheels"]),
        "belt_length": D.belt_length(p) * mm, "links": n_links, "take_up": take_up * mm,
        "sprocket": (p["sprocket_x"] * mm, p["sprocket_z"] * mm), "idler": (p["idler_x"] * mm, p["idler_z"] * mm),
        "r_idler": p["idler_diameter"] / 2 * mm,
        "x_front": max(x + r + outer * mm for x, _, r in belt), "x_rear": min(x - r - outer * mm for x, _, r in belt),
        "belt_circles": belt, "track_center_y": D.track_center_y(p) * mm,
        "overall": {k: v * mm for k, v in ov.items()},
        "guide_height": p["guide_height"] * mm, "link_thickness": p["link_thickness"] * mm,
        "pin_diameter": p["pin_diameter"] * mm,
    }


def mass_budget(p: dict | None = None, cad: dict | None = None, parts_kg: dict | None = None) -> dict:
    """Notebook 30 §1's parts list [kg]: part -> mass (CAD × density, the catalogue drives, the listed parts)."""
    p = design_params() if p is None else p
    cad = cad_numbers(p) if cad is None else cad
    g = geometry(p)
    n = g["links"]
    pa = MATERIALS["PA6-GF30 links, sprockets, wheels"]
    pin = math.pi / 4 * p["pin_diameter"] ** 2 * p["track_width"] * MATERIALS["steel pins (42CrMo4)"]
    drive = act.get(DRIVE)
    out = {
        f"hull shell, {SHELL_T_MM} mm Al 6061 (from CAD area)": cad["hull_surface_area"] * SHELL_T_MM * MATERIALS["Al 6061-T6 hull shell"],
        f"track links 2x{n} (PA6-GF30, from CAD)": 2 * n * cad["link_volume"] * pa,
        f"track pins 2x{n} (steel, d {p['pin_diameter']:g} mm)": 2 * n * pin,
        "sprockets 2x (PA6-GF30, ribbed, from CAD)": 2 * cad["sprocket_volume"] * pa * WHEEL_FILL,
        "idlers 2x (PA6-GF30, ribbed, from CAD)": 2 * cad["idler_volume"] * pa * WHEEL_FILL,
        f"road wheels 2x{int(p['road_wheels'])} (PA6-GF30, ribbed, from CAD)": 2 * int(p["road_wheels"]) * cad["road_wheel_volume"] * pa * WHEEL_FILL,
        f"track drives 2x ({DRIVE}; {drive.mass_g / 1000:.2f} kg each)": 2 * drive.mass_g / 1000.0,
    }
    if cad.get("basket_volume", 0.0) > 0:
        out[f"basket ({p['basket_wall']:g} mm Al 5754, from CAD)"] = cad["basket_volume"] * MATERIALS["Al 5754 basket sheet"]
    out.update(PARTS_KG if parts_kg is None else parts_kg)
    return out


def total_mass(p: dict | None = None, payload_kg: float = 0.0) -> float:
    return sum(mass_budget(p).values()) + payload_kg


def _belt_centroid(circles) -> tuple:
    """Centroid (x, z) of the belt loop (the convex hull of the pin-line circles, length-weighted)."""
    import pekari_rover

    poly = pekari_rover._hull_of_circles(circles, 360)
    sx = sz = sl = 0.0
    for i in range(len(poly)):
        (x0, z0), (x1, z1) = poly[i], poly[(i + 1) % len(poly)]
        l = math.hypot(x1 - x0, z1 - z0)
        sx += l * (x0 + x1) / 2
        sz += l * (z0 + z1) / 2
        sl += l
    return sx / sl, sz / sl


def masses(p: dict | None = None, payload_kg: float = 0.0) -> list:
    """Every mass of the budget with its position: [(name, kg, x, z)] [kg, m] in the vehicle frame (x from the
    road wheels' centre, z above the ground); the battery placed by ``ASSUMPTIONS['battery_position']``."""
    p = design_params() if p is None else p
    g = geometry(p)
    budget = mass_budget(p)
    mm = 1e-3
    cad = cad_numbers(p)
    belt_x, belt_z = _belt_centroid(g["belt_circles"])
    xs, zw = g["wheel_x"], g["wheel_z"]
    sx, sz = g["sprocket"]
    ix, iz = g["idler"]
    z_floor = p["ground_clearance"] * mm
    z_mid = (p["ground_clearance"] + p["hull_height"] / 2) * mm
    z_roof = (p["ground_clearance"] + p["hull_height"]) * mm
    x_front_hull = (p["hull_length"] / 2 - p["sensor_box"] / 2 - p["hull_chamfer"]) * mm
    pos = {}
    for name in budget:
        if name.startswith("hull shell"):
            pos[name] = (cad["hull_com"][0] * mm, cad["hull_com"][2] * mm)
        elif name.startswith("track links") or name.startswith("track pins"):
            pos[name] = (belt_x, belt_z)
        elif name.startswith("sprockets"):
            pos[name] = (sx, sz)
        elif name.startswith("idlers"):
            pos[name] = (ix, iz)
        elif name.startswith("road wheels"):
            pos[name] = (sum(xs) / len(xs), zw)
        elif name.startswith("track drives"):
            pos[name] = (sx + 0.06, sz)
        elif name.startswith("computer"):
            pos[name] = (0.10, z_mid)
        elif name.startswith("sensors"):
            pos[name] = (x_front_hull, z_roof + p["sensor_box"] * 0.25 * mm)
        elif name.startswith("wiring"):
            pos[name] = (0.0, z_mid)
        elif name.startswith("hull frame"):
            pos[name] = (0.0, z_mid)
        elif name.startswith("basket"):
            pos[name] = (p["basket_x"] * mm, z_roof + cad["basket_com"][2] * mm)
        elif name.startswith("track frames"):
            pos[name] = ((sx + ix) / 2, (zw + min(sz, iz)) / 2 + 0.02)
    batt = next(k for k in budget if k.startswith("battery"))
    others = [(k, budget[k], *pos[k]) for k in budget if k != batt]
    m_tot = sum(budget.values())
    x_target = sum(xs) / len(xs)
    x_batt = (m_tot * x_target - sum(m * x for _, m, x, _ in others)) / budget[batt]
    lim = p["hull_length"] / 2 * mm - 0.10
    x_batt = max(-lim, min(lim, x_batt))
    out = others + [(batt, budget[batt], x_batt, z_floor + 0.04)]
    if payload_kg:
        has_basket = cad.get("basket_volume", 0.0) > 0
        x_pl = p["basket_x"] * mm if has_basket else 0.0
        z_pl = z_roof + (p["basket_wall"] * mm + 0.05 if has_basket else 0.06)
        out.append(("payload", payload_kg, x_pl, z_pl))
    return out


def cg(p: dict | None = None, payload_kg: float = PAYLOAD_KG) -> dict:
    """Mass [kg] and CG (x, z) [m] of the empty and the loaded rover, and the battery's x."""
    out = {}
    for label, pl in (("empty", 0.0), ("loaded", payload_kg)):
        ms = masses(p, pl)
        m = sum(k for _, k, _, _ in ms)
        out[label] = {"mass_kg": m, "x": sum(k * x for _, k, x, _ in ms) / m, "z": sum(k * z for _, k, _, z in ms) / m}
    out["battery_x"] = next(x for n, _, x, _ in masses(p) if n.startswith("battery"))
    return out


# ----------------------------------------------------------------------------------------------- Chiron (MuJoCo)
#: ChironLab settings: 1 ms steps, a 20 mm height field over the course of notebook 30 §8.
LAB_OPTIONS = {"heightfield_cell": 0.02, "course_extent": (-1.5, 6.5, -2.0, 5.0), "log_dt": 0.02}
SIDES = {"L": 1, "R": -1}
ROLLER_SHARE = 3.0                                   # a roller can carry 1/3 of its side's drive force (below)
ROLLER_KD = 0.5                                       # N·m·s/rad: 0.1 m/s belt-speed error -> the roller's share of stall
ROLLER_ARMATURE = 5e-4                                # kg·m²: motor rotor through 16:1, shared over the rollers (assumed)
BOGIE_RANGE_DEG = 12.0
MID_ROLLER_KG = 0.02                                  # the belt rollers between the road wheels (taken from the frames)
BOGIE_KG = 0.10


def rollers(p: dict | None = None) -> list:
    """The rollers under one track [(name, x, z, r, kind, bogie)] [m]: ``r`` the belt's outer radius around the
    wheel (wheel + link + grouser; the sprocket: pitch radius + half a link + grouser), ``kind`` sprocket / idler /
    road / mid, ``bogie`` 0 (rear), 1 (front) or None (frame)."""
    p = design_params() if p is None else p
    g = geometry(p)
    mm = 1e-3
    t, gr = p["link_thickness"] * mm, p["grouser_height"] * mm
    xs, zr, rw = g["wheel_x"], g["wheel_z"], g["d_wheel"] / 2
    out = [("sprocket", g["sprocket"][0], g["sprocket"][1], g["r_sprocket"] + t / 2 + gr, "sprocket", None),
           ("idler", g["idler"][0], g["idler"][1], g["r_idler"] + t + gr, "idler", None)]
    ground = sorted(xs + [(a + b) / 2 for a, b in zip(xs[:-1], xs[1:])])
    half = len(ground) // 2
    rg = rw + t + gr
    for i, x in enumerate(ground):
        kind = "road" if any(abs(x - w) < 1e-9 for w in xs) else "mid"
        bogie = None if i == half else (0 if i < half else 1)
        out.append((f"g{i + 1}", x, zr, rg, kind, bogie))
    # the inclined belt runs up to the sprocket and the idler: a roller midway on each (the circle halfway between two
    # circles touches their common lower tangent), so an obstacle meets a ramp, as on the real belt
    for name, (x1, z1, r1) in (("ramp_rear", out[0][1:4]), ("ramp_front", out[1][1:4])):
        x2 = ground[0] if name == "ramp_rear" else ground[-1]
        out.append((name, (x1 + x2) / 2, (z1 + zr) / 2, (r1 + rg) / 2, "mid", None))
    return out


def roller_joints(side: str, p: dict | None = None) -> list:
    return [f"{side}_{name}" for name, *_ in rollers(p)]


def pekari(overrides: dict | None = None, *, payload_kg: float = PAYLOAD_KG, track_mu: float = 0.6,
           hull_mu: float = 0.5, bogies: bool = True, internal_f0: float = 0.035) -> Robot:
    """The Pekari Rover as a Chiron ``Robot`` (see the module docstring): hull (logged), per side 9 rollers (feet
    ``L_sprocket``, ``L_idler``, ``L_g1`` … ``L_g7``, ``L_ramp_rear``, ``L_ramp_front``, and ``R_…``), bogies on passive hinges (``bogies=False``:
    welded). ``track_mu``: the belt's friction on the ground (0.6: PA6 grousers on gravel, terramechanics);
    ``internal_f0``: the track's internal loss coefficient at rest (``ASSUMPTIONS['internal_loss']``)."""
    p = design_params(overrides)
    g = geometry(p)
    budget = mass_budget(p)
    ms = masses(p, payload_kg)
    m_total = sum(m for _, m, _, _ in ms)
    mm = 1e-3
    z_hull = (p["ground_clearance"] + p["hull_height"] / 2) * mm
    Lh, Wh, Hh = p["hull_length"] * mm, p["hull_width"] * mm, p["hull_height"] * mm
    b = g["b"]
    yc = g["track_center_y"]
    rl = rollers(p)
    n = len(rl)
    drive = act.get(DRIVE)
    r_s = g["r_sprocket"]
    F_stall, F_rated = drive.stall_Nm / r_s, drive.rated_Nm / r_s
    v0 = drive.no_load_rpm * 2 * math.pi / 60 * r_s
    W_side = m_total * G / 2
    n_road = int(p["road_wheels"])
    m_of = {"sprocket": next(v for k, v in budget.items() if k.startswith("sprockets")) / 2,
            "idler": next(v for k, v in budget.items() if k.startswith("idlers")) / 2,
            "road": next(v for k, v in budget.items() if k.startswith("road wheels")) / (2 * n_road),
            "mid": MID_ROLLER_KG}
    quat_y = (math.cos(math.pi / 4), -math.sin(math.pi / 4), 0.0, 0.0)          # cylinder axis z -> y
    tyre, frame_rgba, hull_rgba = (0.08, 0.08, 0.08, 1.0), (0.25, 0.27, 0.22, 1.0), (0.42, 0.40, 0.30, 1.0)
    fr = (track_mu, 0.01, 0.0001)
    n_bogie = len({bg for *_, bg in rl if bg is not None})                 # bogies per side

    def roller_link(side, name, x, z, r, kind, parent_pos):
        sy = SIDES[side]
        servo = Servo(stall_torque=F_stall / ROLLER_SHARE * r, rated_torque=F_rated / ROLLER_SHARE * r, no_load_speed=v0 / r,
                      stall_current=drive.stall_A, voltage=drive.voltage_V, kp=0.0, kd=ROLLER_KD,
                      armature=ROLLER_ARMATURE, source=f"{DRIVE}, a third of the side's force per roller")
        jn = f"{side}_{name}"
        return Link(f"{jn}_link", pos=(x - parent_pos[0], sy * yc - parent_pos[1], z - parent_pos[2]),
                    joints=[Joint(jn, axis=(0, 1, 0), range=None, tag="wheel", servo=servo, leg=jn,
                                  frictionloss=internal_f0 * W_side / n * r)],
                    geoms=[Geom(jn, "cylinder", (r, b / 2), quat=quat_y, mass=m_of[kind], role="foot", friction=fr,
                                rgba=tyre)])

    children, feet = [], []
    hull_origin = (0.0, 0.0, z_hull)
    for side, sy in SIDES.items():
        on_frame = [rr for rr in rl if rr[5] is None]
        for name, x, z, r, kind, _ in on_frame:
            children.append(roller_link(side, name, x, z, r, kind, hull_origin))
        for k in range(2):
            members = [rr for rr in rl if rr[5] == k]
            if not members:
                continue
            xp = sum(rr[1] for rr in members) / len(members)
            zp = members[0][2] + 0.022
            piv = (xp, sy * yc, zp)
            sub = [roller_link(side, name, x, z, r, kind, piv) for name, x, z, r, kind, _ in members]
            span = max(rr[1] for rr in members) - min(rr[1] for rr in members)
            joints = ([Joint(f"{side}_bogie{k}", axis=(0, 1, 0), range=(-math.radians(BOGIE_RANGE_DEG), math.radians(BOGIE_RANGE_DEG)),
                             damping=2.0, tag="bogie")] if bogies else [])
            children.append(Link(f"{side}_bogie{k}", pos=(piv[0] - hull_origin[0], piv[1], piv[2] - hull_origin[2]), joints=joints,
                                 geoms=[Geom(f"{side}_bogie{k}_plate", "box", (span / 2 + 0.01, 0.003, 0.01),
                                             pos=(0.0, -sy * (b / 2 + 0.006), -0.011), mass=BOGIE_KG, role="visual",
                                             rgba=frame_rgba)],
                                 children=sub))
        for name, *_ in rl:
            jn = f"{side}_{name}"
            feet.append(FootSpec(jn, jn, [jn], "hull"))
    # the hull: every listed mass as a point mass (the wheels' masses ride on their rollers; the belt rollers and
    # bogie plates are taken from the track frames)
    skip = ("sprockets", "idlers", "road wheels")
    masses_h = []
    for name, m, x, z in ms:
        if name.startswith(skip):
            continue
        if name.startswith("track frames"):
            m -= 2 * (MID_ROLLER_KG * sum(1 for rr in rl if rr[4] == "mid") + BOGIE_KG * n_bogie)
        masses_h.append(PointMass(name, m, (x, 0.0, z - z_hull)))
    roof = Hh / 2
    geoms = [Geom("hull", "box", (Lh / 2, Wh / 2, Hh / 2), mass=None, role="body", friction=(hull_mu, 0.005, 0.0001),
                  rgba=hull_rgba)]
    if p["sensor_box"] > 0:
        s_ = p["sensor_box"] * mm
        geoms.append(Geom("sensor_box", "box", (s_ / 2, 0.75 * s_, s_ / 4),
                          pos=(Lh / 2 - s_ / 2 - p["hull_chamfer"] * mm, 0.0, roof + s_ / 4), role="visual", rgba=frame_rgba))
    if p["basket_height"] > 0:
        bl, bw, bh = p["basket_length"] * mm, p["basket_width"] * mm, p["basket_height"] * mm
        geoms.append(Geom("basket", "box", (bl / 2, bw / 2, bh / 2), pos=(p["basket_x"] * mm, 0.0, roof + bh / 2),
                          role="visual", rgba=(0.6, 0.6, 0.62, 0.6)))
    for side, sy in SIDES.items():                                      # the track frames and the belt, visual
        geoms.append(Geom(f"{side}_frame", "box", (0.29, 0.003, 0.035), pos=(0.0, sy * (Wh / 2 + 0.003), 0.11 - z_hull),
                          role="visual", rgba=frame_rgba))
    hull = Link("hull", log=True, geoms=geoms, masses=masses_h, children=children)
    nominal = {f"{side}_{name}": 0.0 for side in SIDES for name, *_ in rl}
    if bogies:
        nominal.update({f"{side}_bogie{k}": 0.0 for side in SIDES for k in range(2)})
    robot = Robot("Pekari Rover", hull, feet=feet, nominal_qpos=nominal,
                  notes=f"Pekari Rover: hull + 2 tracks as 11 rollers each; {DRIVE} per track shared over the rollers; "
                        f"payload {payload_kg:g} kg in the basket",
                  sources={"geometry": "designs/pekari_rover.py", "masses": "pekari_rover_robot.masses",
                           "drive": "designs/actuators.py", "assumptions": "pekari_rover_robot.ASSUMPTIONS"})
    robot.validate()
    robot.params, robot.geometry = p, g
    robot.rollers = rl
    robot.payload_kg = payload_kg
    return robot


def pekari_lab(terrain=None, robot: Robot | None = None, **kwargs) -> ChironLab:
    """``ChironLab(pekari(), terrain, **LAB_OPTIONS)`` (``kwargs`` override the options)."""
    opts = dict(LAB_OPTIONS)
    opts.update(kwargs)
    return ChironLab(robot or pekari(), terrain, **opts)
