"""Serval: the fast hit-and-run quad. Kerchak's quad (``gorilla_quad``, ``gorilla_quad_autonomy``) with nobody aboard,
no seat, no handlebar: it drives itself, fast, and carries two mounted tools, one above the other:

* **the aerogolf cannon** — Armageddon's golf-ball launcher (the Pekari's turret, ``gorilla_bazooka.SPEC``: two
  counter-rotating flywheels, rims 20-70 m/s, a ball every 0.4 s) on a powered turret in the seat's place, with a big
  hopper behind it (``GOLF['hopper']`` balls) and a gyro-stabilised pan-tilt, so it fires on the move;
* **the flak on top** — the recoilless 8 mm cannon (``gorilla_bazooka.FLAK``: 280 m/s, 1.0 m tube, no recoil, a ball
  every 0.4 s) on its own pan-tilt at the top of a fixed mast that rises through the golf turret's hollow ring; a
  larger air bottle and magazine (``FLAK_MOUNT``) for many shots; the gunner's sensors on its cradle (a 70° search
  camera, a 20° zoom sight with a laser rangefinder: Armageddon's tracker, ``track.py``). The two turrets turn
  independently (two slip rings, one round the other).

**Why the flak on top**: a recoilless gun throws its back-blast (``FLAK['backblast_m']``, 2.1 m) out of the rear
venturi. With the flak the highest thing on the quad, the blast goes back over everything when it fires level or
upwards; firing steeply up sends the blast down behind the pivot, onto the golf turret under it or a rear tyre, so
``blast_clear(pan, tilt)`` says where it may fire (``BLAST_KEEP_OUT``: up to 40 degrees over the bow, 61-67 to the sides
and astern, ~32 over the rear quarters, where the blast would reach a rear tyre) and the game enforces it. The deck,
the fenders and the trunk carry a 2 mm steel blast plate and take the rest.

**Fast**: no gorilla (384 kg) and no seat, bar or scabbards: the quad is ~170 kg lighter than Kerchak's loaded quad;
the same four PMSM wheel motors run to 6000 rpm on a taller reduction (``DRIVE['ratio']``) for ``V_MAX``; the
drag area drops without the rider. ``sizing()`` has the acceleration (traction-, torque- and power-limited), the top
speed, the braking, the stability with both turrets (the static stability factor and the lateral g to roll over),
and the battery.

**Hit and run** (``hit_and_run``): dash in at speed, fire both on the move inside their ranges (the golf cannon at
ground targets out to ``golf_range``, the flak at drones out to 150 m), turn and dash out; the exposure time and the
shots it gets off are the tactics' numbers.

    import serval
    print(serval.summary())
    serval.ServalDesign().run("out")          # CAD: part="serval" (the whole), "golf_turret", "flak_mount", ...
    serval.views("exports/serval/serval_views.png")
"""
from __future__ import annotations

import json
import math

import cadquery as cq
import numpy as np

import gorilla_bazooka as gb
import gorilla_quad as gq
from vegeta.dedalus import Design, Parameter

__all__ = ["V_MAX", "DRIVE", "GOLF", "FLAK_MOUNT", "SENSORS", "REMOVED", "sizing", "accelerate", "golf_range", "blast_clear",
           "BLAST_KEEP_OUT", "hit_and_run", "summary", "game_spec", "ServalDesign", "views"]

G = 9.81
V_MAX = 80 / 3.6              #: top speed [m/s]
MU = 0.8                      #: tyre-ground friction (ATV tyres on tarmac, dry)
CDA = 0.75                    #: drag area, nobody aboard, the two turrets [m²]
CRR = gq.CRR

#: the drive: the quad's four PMSM wheel motors (3 kW continuous, 6 kW peak) run up to 6000 rpm, on a taller reduction
DRIVE = {"motors": 4, "peak_kW": 6.0, "cont_kW": 3.0, "peak_torque_Nm": 48.0, "max_rpm": 6000.0, "efficiency": 0.9,
         "ratio": None}       # set below from the top speed
DRIVE["ratio"] = round(DRIVE["max_rpm"] * 2 * math.pi / 60 / (V_MAX / gq.GEOM["wheel_r"]), 2)

#: what Kerchak's quad carries that Serval does not: (mass [kg], where (x, z) [m])
REMOVED = {"handlebar, column, steering rack (steer-by-wire stays)": (9.0, (0.70, 0.90)),
           "seat's moving part": (gq.SEAT_MOVING_KG, (gq.GEOM["pelvis"][0], gq.GEOM["seat_top"] - 0.05)),
           "bazooka, flak, scabbards": (17.6, (-0.05, 0.85))}

#: the aerogolf cannon's turret, in the seat's place (the quad's frame, m): the ring on the seat base
GOLF = {"ring": (-0.12, 0.0, 0.86), "ring_d": 0.46, "tube_m": gb.SPEC["tube_m"], "bore_mm": gb.SPEC["bore_mm"],
        "rim_speed_range": gb.SPEC["rim_speed_range"], "grip": gb.SPEC["grip"], "feed_s": gb.SPEC["feed_s"],
        "scatter_rad": gb.SPEC["scatter_rad"], "scatter_speed": gb.SPEC["scatter_speed"],
        "hopper": 240, "pan_deg": 360.0, "tilt_deg": (-10.0, 55.0), "slew_deg_s": 120.0, "stabilised": True,
        "trunnion_up": 0.16,        # the tube's tilt axis above the ring
        "mass_kg": {"turret ring, slip ring, pan drive": 6.5, "tilt yoke + drive, gyro stabiliser": 3.0,
                    "flywheels + 2 BLDC motors, housing": 2.7, "tube (CF, 44 mm bore, 0.70 m)": 0.9,
                    "hopper (240 balls), feed auger": 3.5}}
GOLF["mass_kg"]["240 golf balls"] = round(GOLF["hopper"] * gb.BALL["mass"], 1)

#: the flak's mount on top: a fixed mast through the golf ring, its own pan-tilt at the top
FLAK_MOUNT = {"pivot": (-0.12, 0.0, 1.70), "pan_deg": 360.0, "tilt_deg": (-10.0, 85.0), "slew_deg_s": 150.0,
              "tube_m": gb.FLAK["tube_m"], "breech_back_m": 0.30, "venturi_back_m": 0.62, "backblast_m": gb.FLAK["backblast_m"],
              "muzzle_speed": gb.FLAK["muzzle_speed"], "fire_every_s": gb.FLAK["fire_every_s"],
              "bottle_l": 4.0, "bottle_bar": 300.0, "magazine": 480,
              "mass_kg": {"mast (Al tube 90 mm) and its base": 4.0, "pan-tilt head, drives, slip ring": 4.5,
                          "tube (steel-lined CF, 1.0 m)": gb.FLAK["mass_kg"]["tube (steel-lined CF, 1.0 m)"],
                          "breech, valve, venturi": gb.FLAK["mass_kg"]["breech, valve, venturi"],
                          "air bottle 4 L, 300 bar (CF)": 3.1, "magazine 480 balls": 0.4 + 480 * gb.FLAK["ball_mass"]}}
#: shots per bottle: the shoulder flak's 0.8 L gives 93, the air per shot the same
FLAK_MOUNT["shots_per_bottle"] = int(gb.FLAK["shots"] * FLAK_MOUNT["bottle_l"] / 0.8)

#: the gunner's sensors on the flak's cradle (Armageddon's tracker): they pan and tilt with the flak
SENSORS = {"search_camera": {"hfov_deg": 70, "px": (1280, 720)}, "zoom_sight": {"hfov_deg": 20, "px": (1280, 720)},
           "laser_rangefinder": {"sigma_m": 0.1, "beam_deg": 0.4},
           #: four fixed 90-degree search cameras round the mast under the flak's head: nothing comes unseen (the game
           #: showed a kamikaze diving from behind while the flak's own camera looked elsewhere)
           "ring_cameras": {"count": 4, "hfov_deg": 90, "px": (1920, 1080), "below_pivot_m": 0.2},
           "mass_kg": 2.4}

#: what the flak's back-blast must not wash (m, boxes (lo, hi)): the golf turret, its hopper and the mast under the
#: flak, and the four tyres. The deck, the fenders and the trunk get a 2 mm steel blast plate (``BLAST_PLATE``) and may
#: take it - nobody rides.
G_ = gq.GEOM
SENSITIVE = {"golf turret, hopper, mast": ((-0.70, -0.26, 0.80), (0.12, 0.26, 1.24))}
for _c in gq.CORNERS:
    _x, _y = gq._corner_xy(_c)
    SENSITIVE[f"tyre {_c}"] = ((_x - 0.36, _y - 0.16, 0.0), (_x + 0.36, _y + 0.16, 0.70))
BLAST_PLATE = {"what": "2 mm steel over the rear deck, the trunk and the fenders' tops", "mass_kg": 6.0, "pos": (-0.70, 0.0, 0.82)}


def _sum(d):
    return sum(v for v in d.values())


def sizing():
    """The numbers: the masses and the centre of gravity with both turrets, the stability, the drive, the tools."""
    base = gq.sizing(rider_mass=0.0)
    m_v = base["mass_vehicle_kg"]
    x_v, z_v = base["cg_vehicle_m"]
    removed = sum(m for m, _ in REMOVED.values())
    rx = sum(m * p[0] for m, p in REMOVED.values()) / removed
    rz = sum(m * p[1] for m, p in REMOVED.values()) / removed
    m_quad = m_v - removed
    x_q, z_q = (m_v * x_v - removed * rx) / m_quad, (m_v * z_v - removed * rz) / m_quad
    golf_m = _sum(GOLF["mass_kg"]) + BLAST_PLATE["mass_kg"]
    flak_m = _sum(FLAK_MOUNT["mass_kg"]) + SENSORS["mass_kg"]
    gx, gz = GOLF["ring"][0], GOLF["ring"][2] + 0.12
    fx, fz = FLAK_MOUNT["pivot"][0], FLAK_MOUNT["pivot"][2] - 0.25          # the mast low, the head high
    m = m_quad + golf_m + flak_m
    x = (m_quad * x_q + golf_m * gx + flak_m * fx) / m
    h = (m_quad * z_q + golf_m * gz + flak_m * fz) / m
    L, T = gq.GEOM["wheelbase"], gq.GEOM["track"]
    front = m * (L / 2 + x) / L
    ssf = T / (2 * h)
    run = accelerate(m)
    P_top = (CRR * m * G * V_MAX + 0.5 * 1.2 * CDA * V_MAX ** 3) / DRIVE["efficiency"]
    P_50 = (CRR * m * G * (50 / 3.6) + 0.5 * 1.2 * CDA * (50 / 3.6) ** 3) / 0.85 + 150.0
    battery_Wh = base["battery_Wh"]
    return {"mass_kg": round(m, 1), "mass_quad_bare_kg": round(m_quad, 1), "mass_golf_turret_kg": round(golf_m, 1),
            "mass_flak_mount_kg": round(flak_m, 1), "vs_kerchaks_loaded_quad_kg": round(m - gq.sizing()["mass_total_kg"], 1),
            "cg_m": (round(x, 3), round(h, 3)), "axle_load_kg": {"front": round(front, 1), "rear": round(m - front, 1)},
            "ssf": round(ssf, 3), "rollover_lat_g": round(ssf, 2), "cornering_limit_g": round(min(MU, ssf), 2),
            "drive": {**DRIVE, "wheel_torque_Nm": round(DRIVE["peak_torque_Nm"] * DRIVE["ratio"] * 0.95, 1)},
            "top_speed_kmh": round(run["top_kmh"], 1), "power_at_top_speed_kW": round(P_top / 1000, 1),
            "t_0_50_s": run["t_50"], "t_0_70_s": run["t_70"], "d_0_70_m": run["d_70"],
            "braking_g": MU, "stop_from_70_m": round((70 / 3.6) ** 2 / (2 * MU * G), 1),
            "battery_Wh": battery_Wh, "range_50kmh_km": round(0.9 * battery_Wh / P_50 * 50, 0),
            "golf": {"hopper": GOLF["hopper"], "rate_per_s": round(1 / GOLF["feed_s"], 2), "range_m_at_70": round(golf_range(70.0), 1),
                     "range_m_at_45": round(golf_range(45.0), 1), "minutes_of_fire": round(GOLF["hopper"] * GOLF["feed_s"] / 60, 1)},
            "flak": {"shots": min(FLAK_MOUNT["magazine"], FLAK_MOUNT["shots_per_bottle"]), "magazine": FLAK_MOUNT["magazine"],
                     "shots_per_bottle": FLAK_MOUNT["shots_per_bottle"], "rate_per_s": round(1 / FLAK_MOUNT["fire_every_s"], 2),
                     "pivot_height_m": FLAK_MOUNT["pivot"][2]},
            "masses_kg": {"quad without seat, bar, scabbards": round(m_quad, 1), **GOLF["mass_kg"], **FLAK_MOUNT["mass_kg"],
                          "sensors (cameras, laser)": SENSORS["mass_kg"], "blast plate": BLAST_PLATE["mass_kg"]}}


def accelerate(m, dt=0.01):
    """From standstill to top speed on the flat: the tractive force is the least of the tyres' grip (four-wheel drive,
    ``MU`` g), the motors' torque through the reduction, and their peak power; less rolling resistance and drag."""
    v = d = t = 0.0
    out = {"t_50": None, "t_70": None, "d_70": None}
    r = gq.GEOM["wheel_r"]
    while t < 60.0:
        f_grip = MU * m * G
        f_torque = DRIVE["motors"] * DRIVE["peak_torque_Nm"] * DRIVE["ratio"] * 0.95 / r
        f_power = DRIVE["motors"] * DRIVE["peak_kW"] * 1000 * DRIVE["efficiency"] / max(v, 0.5)
        wheel_rpm_limit = v / r * DRIVE["ratio"] * 60 / (2 * math.pi) >= DRIVE["max_rpm"]
        f = 0.0 if wheel_rpm_limit else min(f_grip, f_torque, f_power)
        a = (f - CRR * m * G - 0.5 * 1.2 * CDA * v * v) / m
        if a <= 1e-3:
            break
        v += a * dt
        d += v * dt
        t += dt
        if out["t_50"] is None and v >= 50 / 3.6:
            out["t_50"] = round(t, 1)
        if out["t_70"] is None and v >= 70 / 3.6:
            out["t_70"], out["d_70"] = round(t, 1), round(d, 0)
    out["top_kmh"] = v * 3.6
    return out


def golf_range(rim_speed, elevation_deg=None):
    """How far a golf ball goes on level ground from the golf tube's height (the game's flight: drag on a sphere)."""
    best = 0.0
    z0 = GOLF["ring"][2] + GOLF["trunnion_up"]
    for el in ([elevation_deg] if elevation_deg is not None else range(5, 46, 5)):
        a = math.radians(el)
        sp = rim_speed * GOLF["grip"]
        p, v = np.array([0.0, 0.0, z0]), np.array([sp * math.cos(a), 0.0, sp * math.sin(a)])
        while p[2] > 0.0:
            v = v + (gb.drag_force(v) / gb.BALL["mass"] + np.array([0.0, 0.0, -G])) * 2e-3
            p = p + v * 2e-3
        best = max(best, float(p[0]))
    return best


def blast_clear(pan_deg, tilt_deg, steps=24):
    """Whether the flak may fire at this pan (from the bow, left positive) and tilt: the back-blast's axis, from the
    venturi back ``backblast_m``, must stay out of ``SENSITIVE`` (the golf turret, its hopper and the mast under the flak,
    the tyres); the deck, the fenders and the trunk take it on the blast plate."""
    pan, tilt = math.radians(pan_deg), math.radians(tilt_deg)
    d = np.array([math.cos(tilt) * math.cos(pan), math.cos(tilt) * math.sin(pan), math.sin(tilt)])
    pivot = np.array(FLAK_MOUNT["pivot"])
    venturi = pivot - FLAK_MOUNT["venturi_back_m"] * d
    for lo, hi in SENSITIVE.values():
        lo, hi = np.array(lo), np.array(hi)
        for k in range(steps + 1):
            p = venturi - d * FLAK_MOUNT["backblast_m"] * k / steps
            if np.all(p >= lo) and np.all(p <= hi):
                return False
    return True


def _keep_out():
    """The highest tilt the flak may fire at, by pan (every 15 degrees from the bow)."""
    out = {}
    for pan in range(0, 360, 15):
        ok = [t for t in range(int(FLAK_MOUNT["tilt_deg"][0]), int(FLAK_MOUNT["tilt_deg"][1]) + 1) if blast_clear(pan, t)]
        out[pan] = max(ok) if ok else None
    return out


BLAST_KEEP_OUT = _keep_out()


def hit_and_run(stand_off=200.0, engage=150.0, speed_kmh=70.0):
    """A run at a target: from ``stand_off`` m out it dashes in at ``speed_kmh``, fires both on the move while within
    ``engage`` m (the flak at drones, the golf cannon at what is on the ground within its range), passes, turns away
    at the cornering limit and dashes out. The exposure (seconds within ``engage``) and the shots it gets off."""
    s = sizing()
    v = speed_kmh / 3.6
    g_lat = s["cornering_limit_g"] * 0.8
    turn_r = v * v / (g_lat * G)
    inside = 2 * engage / v + math.pi * turn_r / v
    golf_r = min(s["golf"]["range_m_at_70"], engage)
    golf_window = 2 * golf_r / v + math.pi * turn_r / v
    return {"speed_kmh": speed_kmh, "turn_radius_m": round(turn_r, 1), "exposure_s": round(inside, 1),
            "flak_shots": int(inside / FLAK_MOUNT["fire_every_s"]), "golf_shots": int(golf_window / GOLF["feed_s"]),
            "runs_per_magazine": round(min(FLAK_MOUNT["magazine"], FLAK_MOUNT["shots_per_bottle"]) / max(1, int(inside / FLAK_MOUNT["fire_every_s"])), 1),
            "runs_per_hopper": round(GOLF["hopper"] / max(1, int(golf_window / GOLF["feed_s"])), 1),
            "time_in_from_stand_off_s": round((stand_off - engage) / v, 1)}


def summary():
    return json.dumps({"sizing": sizing(), "hit_and_run": hit_and_run(), "blast_keep_out_max_tilt_by_pan": BLAST_KEEP_OUT}, indent=1)


def game_spec():
    """What Armageddon needs to put Serval in the game: the vehicle, the two mounts, the sensors, the blast rule."""
    s = sizing()
    return {"name": "serval", "source": "vegeta labs/legged/designs/serval.py", "kind": "rover",
            "vehicle": {"mass_kg": s["mass_kg"], "top_speed_m_s": round(V_MAX, 2), "accel_m_s2": round(MU * G * 0.85, 2),
                        "braking_m_s2": round(MU * G, 2), "turn_rate_rad_s_at_speed": round(s["cornering_limit_g"] * G / V_MAX, 3),
                        "wheelbase_m": gq.GEOM["wheelbase"], "track_m": gq.GEOM["track"], "rollover_lat_g": s["rollover_lat_g"],
                        "size_m": (2.6, 2.2, round(FLAK_MOUNT["pivot"][2] + 0.15, 2))},
            "golf": {"pivot": (GOLF["ring"][0], 0.0, GOLF["ring"][2] + GOLF["trunnion_up"]), "tube_m": GOLF["tube_m"],
                     "speeds": GOLF["rim_speed_range"], "grip": GOLF["grip"], "feed_s": GOLF["feed_s"], "hopper": GOLF["hopper"],
                     "pan_deg": GOLF["pan_deg"], "tilt_deg": GOLF["tilt_deg"], "slew_deg_s": GOLF["slew_deg_s"], "stabilised": True,
                     "scatter_rad": GOLF["scatter_rad"]},
            "flak": {"pivot": FLAK_MOUNT["pivot"], "tube_m": FLAK_MOUNT["tube_m"], "muzzle_speed": FLAK_MOUNT["muzzle_speed"],
                     "fire_every_s": FLAK_MOUNT["fire_every_s"], "shots": s["flak"]["shots"], "pan_deg": FLAK_MOUNT["pan_deg"],
                     "tilt_deg": FLAK_MOUNT["tilt_deg"], "slew_deg_s": FLAK_MOUNT["slew_deg_s"],
                     "blast_keep_out_max_tilt_by_pan_deg": BLAST_KEEP_OUT, "backblast_m": FLAK_MOUNT["backblast_m"]},
            "sensors": SENSORS, "hit_and_run": hit_and_run()}


# ---------------------------------------------------------------- CAD (mm, the quad's frame)
def _cyl(axis, r, x0, x1, at=(0.0, 0.0, 0.0)):
    """A cylinder along ``axis`` ('x' or 'z') from x0 to x1 (mm), its axis through ``at``."""
    if axis == "x":
        return cq.Workplane("YZ").circle(r).extrude(x1 - x0).translate((x0 + at[0], at[1], at[2]))
    return cq.Workplane("XY").circle(r).extrude(x1 - x0).translate((at[0], at[1], x0 + at[2]))


class ServalDesign(Design):
    """Serval (the quad's frame, mm, at ride height): ``serval`` (all), ``quad`` (the quad without seat, bar or
    scabbards, with the autonomy kit), ``golf_turret`` (ring, yoke, flywheel housing, tube, the hopper), ``flak_mount``
    (mast, pan-tilt head, tube, breech, venturi, bottle, magazine, the sensors)."""

    parameters = [Parameter("part", "serval", choices=("serval", "quad", "golf_turret", "flak_mount"), description="what to build"),
                  Parameter("golf_pan", 0.0, "deg", description="the golf turret's pan (from the bow, left +)"),
                  Parameter("golf_tilt", 10.0, "deg", description="the golf tube's elevation"),
                  Parameter("flak_pan", 30.0, "deg", description="the flak's pan"),
                  Parameter("flak_tilt", 25.0, "deg", description="the flak's elevation")]

    def quad(self, p):
        q = gq.GorillaQuad()
        qp = q.resolve(detail=0)
        g, mm = gq.GEOM, 1000.0
        import gorilla_quad_autonomy as ga
        shapes = [q.frame(qp), q.trunk(qp).val(), q.tools(qp), ga.AutonomyKit().build({"part": "kit"})]
        wheel, up, arm, coil = q.wheel(qp), q.upright(qp), q.wishbone(qp), q.coilover(qp)
        for c in gq.CORNERS:
            x, y = gq._corner_xy(c)
            sy = 1 if y > 0 else -1
            mirror = (lambda s: s.mirror("XZ")) if sy < 0 else (lambda s: s)
            shapes.append(wheel.translate(cq.Vector(x * mm, y * mm, g["wheel_r"] * mm)))
            shapes.append(mirror(up).translate(cq.Vector(x * mm, y * mm, g["wheel_r"] * mm)))
            for dz in (-90, 110):
                shapes.append(mirror(arm).translate(cq.Vector(x * mm, sy * g["frame_y"] * mm, g["wheel_r"] * mm + dz)))
            shapes.append(coil.translate(cq.Vector(x * mm + (-80 if c[0] == "F" else 80), sy * (g["frame_y"] * mm + 180), g["wheel_r"] * mm - 80)))
        return gq._compound(shapes)

    def golf_turret(self, p):
        x, y, z = (v * 1000 for v in GOLF["ring"])
        R = GOLF["ring_d"] / 2 * 1000
        ring = cq.Workplane("XY").circle(R).circle(70).extrude(70).translate((x, y, z))                 # hollow: the mast goes through
        plate = cq.Workplane("XY").circle(R - 10).circle(70).extrude(12).translate((x, y, z + 70))
        cheeks = [cq.Workplane("XY").box(160, 22, 170).translate((x, y + s * 120, z + 82 + 85)) for s in (-1, 1)]
        tz = z + GOLF["trunnion_up"] * 1000
        housing = cq.Workplane("XY").box(260, 200, 150).translate((x - 40, y, tz)).edges("|X").fillet(30)
        wheels = [cq.Workplane("XY").circle(50).extrude(40, both=True).translate((x - 40, y + s * 55, tz)) for s in (-1, 1)]
        tube = _cyl("x", 27, 80, 80 + GOLF["tube_m"] * 1000, (x, y, tz))
        bore = _cyl("x", 22, 60, 120 + GOLF["tube_m"] * 1000, (x, y, tz))
        tube = tube.cut(bore)
        sight = cq.Workplane("XY").box(90, 30, 30).translate((x + 200, y - 60, tz + 60))
        gun = housing
        for s in [tube, sight] + wheels:
            gun = gun.union(s)
        gun = gun.rotate((x, y, tz), (x, y + 1, tz), -p["golf_tilt"])
        hopper = (cq.Workplane("XY").box(420, 380, 300).translate((x - 330, y, z + 260)).edges("|Z").fillet(60)
                  .faces(">Z").shell(-4))
        chute = cq.Workplane("XY").box(140, 60, 60).translate((x - 140, y, z + 160))
        out = ring.union(plate).union(gun).union(hopper).union(chute)
        for c in cheeks:
            out = out.union(c)
        return out.rotate((x, y, 0), (x, y, 1), p["golf_pan"])

    def flak_mount(self, p):
        px, py, pz = (v * 1000 for v in FLAK_MOUNT["pivot"])
        z_ring = GOLF["ring"][2] * 1000
        mast = cq.Workplane("XY").circle(45).circle(38).extrude(pz - 120 - (z_ring - 60)).translate((px, py, z_ring - 60))
        base = cq.Workplane("XY").box(240, 240, 20).translate((px, py, z_ring - 70))
        head = cq.Workplane("XY").circle(95).extrude(90).translate((px, py, pz - 150))
        yoke = (cq.Workplane("XY").box(140, 200, 150).translate((px, py, pz - 10))
                .cut(cq.Workplane("XY").box(150, 150, 120).translate((px, py, pz + 10))))
        L = FLAK_MOUNT["tube_m"] * 1000
        tube = _cyl("x", 16, 0, L, (px, py, pz))
        breech = _cyl("x", 42, -FLAK_MOUNT["breech_back_m"] * 1000, 40, (px, py, pz))
        venturi = (cq.Workplane("YZ").circle(42).workplane(offset=-(FLAK_MOUNT["venturi_back_m"] - FLAK_MOUNT["breech_back_m"]) * 1000)
                   .circle(70).loft().translate((px - FLAK_MOUNT["breech_back_m"] * 1000, py, pz)))
        bottle = _cyl("x", 62, -260, 300, (px, py - 150, pz - 40))
        magazine = cq.Workplane("XY").box(120, 70, 110).translate((px + 60, py + 120, pz - 30))
        cams = [cq.Workplane("XY").box(110, 70, 60).translate((px + 150, py - 75, pz + 80)),                # zoom sight + laser
                cq.Workplane("XY").box(70, 60, 50).translate((px + 120, py + 75, pz + 75))]                 # search camera
        gun = tube.union(breech).union(venturi).union(bottle).union(magazine)
        for c in cams:
            gun = gun.union(c)
        gun = gun.rotate((px, py, pz), (px, py + 1, pz), -p["flak_tilt"])
        top = head.union(yoke).union(gun).rotate((px, py, 0), (px, py, 1), p["flak_pan"])
        return mast.union(base).union(top)

    def build(self, p):
        part = p["part"]
        if part != "serval":
            return getattr(self, part)(p)
        return gq._compound([self.quad(p), self.golf_turret(p).val(), self.flak_mount(p).val()])


COLORS = {"quad": (0.85, 0.62, 0.20, 1.0), "golf_turret": (0.20, 0.32, 0.22, 1.0), "flak_mount": (0.30, 0.31, 0.34, 1.0)}


def views(path, title="Serval - the fast hit-and-run quad: aerogolf turret, the flak on top"):
    """Four shaded views (3/4 front, 3/4 rear, side, top) of the CAD, matplotlib (no OpenGL needed)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    d = ServalDesign()
    p = d.resolve()
    q = gq.GorillaQuad()
    qp = q.resolve(detail=0)
    import gorilla_quad_autonomy as ga
    g = gq.GEOM
    meshes = [(gq._tess(q.frame(qp), tol=4.0), gq.COLORS["frame"]), (gq._tess(q.trunk(qp)), gq.COLORS["trunk"]),
              (gq._tess(q.tools(qp)), gq.COLORS["tools"]), (gq._tess(ga.AutonomyKit().build({"part": "kit"}), tol=3.0), (0.15, 0.15, 0.16, 1.0)),
              (gq._tess(d.golf_turret(p), tol=2.0), (0.35, 0.55, 0.30, 1.0)), (gq._tess(d.flak_mount(p), tol=2.0), (0.22, 0.23, 0.26, 1.0))]
    wheel = gq._tess(q.wheel(qp), tol=4.0)
    rr = np.linalg.norm(wheel[:, :, [0, 2]].mean(axis=1), axis=1)
    for c in gq.CORNERS:
        x, y = gq._corner_xy(c)
        sy = 1.0 if y > 0 else -1.0
        w = wheel * np.array([1.0, sy, 1.0]) + np.array([x, y, g["wheel_r"]])
        meshes.append((w[rr > g["wheel_r"] * 0.6], gq.COLORS["tyre"]))
        meshes.append((w[rr <= g["wheel_r"] * 0.6], gq.COLORS["rim"]))
    light = np.array([0.4, -0.5, 0.8]) / np.linalg.norm([0.4, -0.5, 0.8])
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(title, fontsize=15)
    for k, (name, elev, azim) in enumerate((("3/4 front: the golf turret and its hopper, the flak on its mast", 22, -40),
                                            ("3/4 rear: the trunk under the blast plate", 25, 140), ("side", 0, -90), ("top", 89, -90))):
        ax = fig.add_subplot(2, 2, k + 1, projection="3d")
        for tris, col in meshes:
            n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
            n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
            shade = 0.35 + 0.65 * np.abs(n @ light)
            fc = np.clip(np.array(col[:3])[None, :] * shade[:, None], 0, 1)
            ax.add_collection3d(Poly3DCollection(tris, facecolors=fc, edgecolors="none"))
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        ax.set_zlim(0.0, 2.3)
        ax.set_box_aspect((2.6, 2.6, 2.3), zoom=1.35)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(name)
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


if __name__ == "__main__":
    print(summary())
