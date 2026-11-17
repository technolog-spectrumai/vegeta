"""Kerchak's quad: a single-seat electric four-wheeler sized round the gorilla (``space_unit_robots.bare``, 384 kg).

What it is built for, and where each number comes from:

* **the rider** — Kerchak seated astride (``seated_pose``: the pelvis upright on the seat, the soles flat on
  footboards, the fists round the grips; IK on the frame's own kinematics). The seat is as wide as his pelvis
  (0.56 m), the footboards as long as his feet (0.34 m), the grips where his long arms (1.04 m + hand) reach with the
  elbows bent, the bar as wide as his shoulders (0.82 m) plus a fist each side;
* **stability** — four wheels, the track set from the static stability factor ``SSF = T / (2 h_cg)`` with the
  gorilla on board (``SSF_TARGET``); the battery and the four motors low in the frame to pull ``h_cg`` down;
* **drive** — one motor per wheel (inboard, on the sprung frame, half-shafts to the hubs, so the unsprung mass stays
  light), sized for a 35 % grade with 1 m/s² to spare and 45 km/h on the flat; torque vectoring is free;
* **suspension** — independent double wishbones and coilovers at every corner, the springs for a ``RIDE_HZ`` ride
  (car-comfort, below the 1.5-2 Hz of sport ATVs) and ``ZETA`` damping, so the gorilla does not cry;
* **steering** — a motorbike handlebar on a raked column, direct (1:1) to the front knuckles with Ackermann;
* **trunk** — an open tray over the rear axle for tools (no lid: grab and go).

Units: SI in the sizing and the MuJoCo model, mm in the CAD. Frame: x forward, y left, z up; the origin on the
ground under the middle of the wheelbase at static ride height.

    import gorilla_quad as gq
    s = gq.sizing()                 # the numbers
    gq.GorillaQuad().run("out")     # CAD: STEP + STL (part="quad", "frame", "wheel", ...)
    prop = gq.quad_prop()           # the MuJoCo model (a free Chiron prop)
"""
from __future__ import annotations

import math

import cadquery as cq
import numpy as np

from vegeta.chiron import Geom, Joint, Link, PointMass, Prop
from vegeta.dedalus import Design, Parameter

__all__ = ["RIDER", "GEOM", "MASS", "sizing", "seated_pose", "GorillaQuad", "quad_prop", "quad_mesh", "CORNERS", "COLORS"]

G = 9.81
SSF_TARGET = 1.0          #: static stability factor with the rider (passenger cars 1.2-1.5, ATVs 0.8-1.0)
RIDE_HZ = 1.1             #: sprung-mass natural frequency [Hz] (picked by the bump sweep: 1.3 -> 1.1 Hz cut the gorilla's chest acceleration ~20 %)
ARB_RATIO = 1.5           #: each axle's anti-roll bar: its rate at the wheel [N/m] for opposite travel, x the corner spring rate
SEAT_HZ = 1.3             #: the suspended seat with the rider on it [Hz] (truck seats 1.2-2 Hz)
SEAT_ZETA = 0.4
SEAT_TRAVEL = 0.06        #: the seat's travel each way [m]
SEAT_MOVING_KG = 4.0      #: the seat's moving part (pan, cushion) [kg]
TYRE_SOLREF = (0.03, 0.5)  #: the tyres' contact (MuJoCo time constant, damping ratio): ~150 kN/m low-pressure ATV tyres
ZETA = 0.3                #: damping ratio of the corner
GRADE = 0.35              #: the steepest slope to climb [-]
ACC_SPARE = 1.0           #: acceleration still available on it [m/s²]
V_MAX = 45 / 3.6          #: top speed [m/s]
CRR = 0.035               #: rolling resistance, ATV tyres off-road
CDA = 1.3                 #: drag area with a gorilla on board [m²]

#: Kerchak's measures (``space_unit_robots.bare`` frame, m): checked by ``tests/test_gorilla_quad.py``.
RIDER = {"mass": 384.5, "pelvis_width": 0.56, "pelvis_half_height": 0.085, "shoulder_width": 0.82, "foot_length": 0.34,
         "foot_width": 0.18, "thigh": 0.36, "shin": 0.34, "upper_arm": 0.54, "forearm": 0.50}

#: The vehicle's geometry [m] (frame origin on the ground, mid wheelbase).
GEOM = {
    "wheel_r": 0.33, "wheel_w": 0.25, "wheelbase": 1.70, "track": 1.80,
    "pelvis": (-0.15, 0.0, 0.93),           # Kerchak's pelvis origin when seated
    "seat_top": 0.93 - 0.085 - 0.01,        # under the pelvis shell, the cushion compressed 10 mm
    "footboard_z": 0.31, "footboard_x": 0.27, "footboard_y": 0.24, "footboard_len": 0.46, "footboard_w": 0.26,
    "grip_x": 0.56, "grip_y": 0.50, "grip_z": 1.12, "grip_r": 0.025,
    "column_base": (0.76, 0.0, 0.62), "column_rake_deg": 25.0,
    "trunk": {"x": -0.78, "len": 0.56, "w": 0.86, "h": 0.28, "floor_z": 0.74, "wall": 0.004},
    "battery": {"x": -0.10, "len": 0.62, "w": 0.42, "h": 0.24, "z": 0.30},
    "frame_y": 0.30, "rail_z": 0.30, "top_rail_z": 0.70,
    "droop": 0.10, "bump": 0.14,            # wheel travel below and above static ride height
    "steer_max": 0.6,                       # handlebar and wheels [rad]
    "ackermann": 0.12,                      # quadratic Ackermann term [1/rad]
}

#: Masses [kg]: sprung (frame side) and per-corner unsprung.
MASS = {
    "frame (4130 chromoly tube, welded)": 46.0,
    "battery 72 V 100 Ah LiFePO4 (7.2 kWh)": 56.0,
    "motors 4x PMSM 3 kW cont / 6 kW peak + 6:1 reduction": 4 * 12.0,
    "inverters 4x, VCU, DC-DC, wiring": 18.0,
    "footboards, bodywork, fenders, seat base and scissor": 20.0,
    "handlebar, column, steering rack": 9.0,
    "trunk tray (Al 4 mm) and tools": 14.0,
    "wishbones, coilovers (sprung half)": 4 * 3.5,
}
UNSPRUNG = {"wheel + tyre 26x10-12": 14.0, "upright, hub, brake disc + caliper": 6.0, "wishbones, coilover, half-shaft (unsprung half)": 4.5}
CORNERS = ("FL", "FR", "RL", "RR")


def _corner_xy(c):
    g = GEOM
    return (g["wheelbase"] / 2 if c[0] == "F" else -g["wheelbase"] / 2, g["track"] / 2 if c[1] == "L" else -g["track"] / 2)


def sizing(rider_cg=(-0.05, 0.0, 0.99)) -> dict:
    """The numbers: masses and the centre of gravity, the axle loads, the stability factor, springs and dampers, the
    motors and the battery. ``rider_cg``: Kerchak's centre of mass seated (``seated_pose``'s, in the vehicle frame)."""
    g = GEOM
    m_r = RIDER["mass"]
    m_s = sum(MASS.values()) + SEAT_MOVING_KG
    m_u = sum(UNSPRUNG.values())
    m_v = m_s + 4 * m_u
    m = m_r + m_v
    # the vehicle's own centre of gravity: battery and motors low, the frame mid, the trunk high at the back
    cg_parts = [(MASS["frame (4130 chromoly tube, welded)"], (0.0, 0.48)), (MASS["battery 72 V 100 Ah LiFePO4 (7.2 kWh)"], (g["battery"]["x"], g["battery"]["z"] + g["battery"]["h"] / 2)),
                (MASS["motors 4x PMSM 3 kW cont / 6 kW peak + 6:1 reduction"], (0.0, g["wheel_r"])), (MASS["inverters 4x, VCU, DC-DC, wiring"], (0.0, 0.45)),
                (MASS["footboards, bodywork, fenders, seat base and scissor"], (0.0, 0.62)), (SEAT_MOVING_KG, (GEOM["pelvis"][0], GEOM["seat_top"] - 0.05)), (MASS["handlebar, column, steering rack"], (0.70, 0.90)),
                (MASS["trunk tray (Al 4 mm) and tools"], (g["trunk"]["x"], g["trunk"]["floor_z"] + 0.08)), (MASS["wishbones, coilovers (sprung half)"], (0.0, 0.45)),
                (4 * m_u, (0.0, g["wheel_r"]))]
    xv = sum(mm * p[0] for mm, p in cg_parts) / m_v
    zv = sum(mm * p[1] for mm, p in cg_parts) / m_v
    x = (m_r * rider_cg[0] + m_v * xv) / m
    h = (m_r * rider_cg[2] + m_v * zv) / m
    L = g["wheelbase"]
    front = m * (L / 2 + x) / L                       # the axle loads [kg]
    rear = m - front
    ssf = g["track"] / (2 * h)
    # springs at the wheel, per corner, for the ride frequency on the corner's sprung mass
    corner = {}
    for c in CORNERS:
        ms = (front if c[0] == "F" else rear) / 2 - m_u
        k = ms * (2 * math.pi * RIDE_HZ) ** 2
        corner[c] = {"sprung_kg": round(ms, 1), "k_N_m": round(k), "c_Ns_m": round(2 * ZETA * math.sqrt(k * ms)),
                     "sag_m": round(ms * G / k, 3)}
    # drive: per wheel, the grade with acceleration to spare; top speed power
    theta = math.atan(GRADE)
    F_wheel = m * (G * (math.sin(theta) + CRR * math.cos(theta)) + ACC_SPARE) / 4
    T_wheel = F_wheel * g["wheel_r"]
    w_max = V_MAX / g["wheel_r"]
    P_top = CRR * m * G * V_MAX + 0.5 * 1.2 * CDA * V_MAX ** 3
    P_30 = (CRR * m * G * (30 / 3.6) + 0.5 * 1.2 * CDA * (30 / 3.6) ** 3) / 0.85 + 100.0
    ratio = 6.0
    k_seat = m_r * (2 * math.pi * SEAT_HZ) ** 2
    seat = {"k_N_m": round(k_seat), "c_Ns_m": round(2 * SEAT_ZETA * math.sqrt(k_seat * m_r)), "sag_m": round(m_r * G / k_seat, 3),
            "hz": SEAT_HZ, "zeta": SEAT_ZETA, "travel_m": SEAT_TRAVEL}
    battery_Wh = 7200.0
    return {
        "mass_vehicle_kg": round(m_v, 1), "mass_rider_kg": m_r, "mass_total_kg": round(m, 1), "unsprung_corner_kg": m_u,
        "cg_vehicle_m": (round(xv, 3), round(zv, 3)), "cg_total_m": (round(x, 3), round(h, 3)),
        "axle_load_kg": {"front": round(front, 1), "rear": round(rear, 1)},
        "ssf": round(ssf, 3), "ssf_target": SSF_TARGET, "track_for_target_m": round(2 * h * SSF_TARGET, 3),
        "rollover_lat_g": round(ssf, 2),
        "anti_roll_bar_N_m": {"front": round(ARB_RATIO * corner["FL"]["k_N_m"]), "rear": round(ARB_RATIO * corner["RL"]["k_N_m"])},
        "corners": corner, "seat": seat, "ride_hz": RIDE_HZ, "zeta": ZETA, "travel_m": (g["droop"], g["bump"]),
        "wheel_force_N": round(F_wheel), "wheel_torque_Nm": round(T_wheel), "wheel_speed_max_rad_s": round(w_max, 1),
        "motor": {"ratio": ratio, "peak_torque_Nm": round(T_wheel / ratio / 0.95, 1), "max_speed_rpm": round(w_max * ratio * 60 / (2 * math.pi)),
                  "peak_power_kW": round(F_wheel * 6.0 / 1000, 1), "cont_power_kW": round(P_top / 4 / 1000 / 0.9, 2)},
        "power_top_speed_kW": round(P_top / 1000, 2), "battery_Wh": battery_Wh,
        "range_30kmh_km": round(0.9 * battery_Wh / P_30 * 30, 0),
    }


# ---------------------------------------------------------------- the rider seated
def seated_pose(kin, steer=0.0, curl=None, q_init=None, axis_sign=None):
    """Kerchak's joint angles astride the quad, the pelvis upright at ``GEOM['pelvis']``: the soles flat on the
    footboards, the fists round the grips of a handlebar turned ``steer`` [rad]. Returns (q, worst IK error [m]).
    The targets go in the pelvis's frame (the IK's base sits at the origin)."""
    import gorilla_scenarios as gs
    import humanoid_motions as hum
    import space_unit_robots as sr

    curl = gs.CURL_CLOSED if curl is None else curl
    pel = np.asarray(GEOM["pelvis"])
    q = kin.q0.copy()
    for j, v in {"spine_lumbar_pitch": 0.12, "spine_thoracic_pitch": 0.18, "L_hip_pitch": -1.25, "R_hip_pitch": -1.25,
                 "L_knee": 1.5, "R_knee": 1.5, "L_shoulder_pitch": -0.9, "R_shoulder_pitch": -0.9, "L_elbow": -0.8, "R_elbow": -0.8}.items():
        q[kin.index[j]] = v
    for side in ("L", "R"):
        sr.set_curl(q, kin, side, curl)
    q_ref = q.copy()
    if q_init is not None:
        q = np.array(q_init, dtype=float)
    kin.set(q)
    targets = []
    g = GEOM
    for side, sy in (("L", 1.0), ("R", -1.0)):
        body, c = kin.geom_point(side)
        for o in hum.SOLE_POINTS:
            sole = c + np.asarray(o)
            sole[2] = c[2] - 0.024                                         # the sole box's bottom face
            goal = np.array([g["footboard_x"] + o[0], sy * g["footboard_y"] + o[1], g["footboard_z"]]) - pel
            targets.append((body, sole, goal))
        grip = handlebar_grip(side, steer) - pel
        axis = handlebar_axis(steer)
        hold = gs.bar(grip, axis=axis * (axis_sign or {'L': 1.0, 'R': 1.0})[side], n=(-1.0, 0.0, 0.0), wrist_dir=(-0.75, 0.0, -0.66))
        targets += gs.hand_targets(kin, side, hold)
    limbs = [j for j in kin.actuated if "finger" not in j and "thumb" not in j and "spine" not in j and j != "waist_yaw"]
    return kin.ik(targets, q_init=q, q_ref=q_ref, joints=limbs, iters=800, reg=3e-4)


def _column_axis():
    a = math.radians(GEOM["column_rake_deg"])
    return np.array([-math.sin(a), 0.0, math.cos(a)])                     # raked back: the top towards the rider


def _rot(axis, ang):
    axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K


def handlebar_grip(side, steer=0.0):
    """A grip's centre in the vehicle frame with the bar turned ``steer``."""
    g = GEOM
    base = np.asarray(g["column_base"])
    p = np.array([g["grip_x"], (1 if side == "L" else -1) * g["grip_y"], g["grip_z"]]) - base
    return base + _rot(_column_axis(), steer) @ p


def handlebar_axis(steer=0.0):
    return _rot(_column_axis(), steer) @ np.array([0.0, 1.0, 0.0])


# ---------------------------------------------------------------- CAD (mm)
def _tube(a, b, r):
    a, b = cq.Vector(*a), cq.Vector(*b)
    return cq.Solid.makeCylinder(r, (b - a).Length, a, (b - a).normalized())


def _box(cx, cy, cz, lx, ly, lz):
    return cq.Solid.makeBox(lx, ly, lz, cq.Vector(cx - lx / 2, cy - ly / 2, cz - lz / 2))


def _compound(shapes):
    return cq.Compound.makeCompound([s.val() if isinstance(s, cq.Workplane) else s for s in shapes])


class GorillaQuad(Design):
    """Kerchak's quad (``gorilla_quad`` module docstring). ``part``: the whole vehicle at ride height, or one part in
    its own frame: ``frame`` (the welded tube frame with the footboards, seat, battery box, motors, fenders, trunk -
    everything on the sprung body, in the vehicle frame), ``handlebar`` (about the column's pivot), ``upright`` and
    ``wheel`` (about the wheel's centre, the axle along y), ``wishbone``, ``coilover``, ``trunk``, ``tools``."""

    parameters = [Parameter("part", "quad", choices=("quad", "frame", "handlebar", "upright", "wheel", "wishbone", "coilover", "trunk", "tools", "seat"),
                            description="what to build"),
                  Parameter("tube_r", 16.0, "mm", min=8, description="frame tube radius (32 mm OD 4130)"),
                  Parameter("detail", 1, "", min=0, max=1, description="0 drops the tread lugs (faster)")]

    # -- parts in their own frames
    def wheel(self, p):
        g = GEOM
        R, W = g["wheel_r"] * 1000, g["wheel_w"] * 1000
        tyre = cq.Workplane("XZ").circle(R).circle(R * 0.62).extrude(W / 2, both=True).edges().fillet(38)
        rim = cq.Workplane("XZ").circle(R * 0.63).circle(R * 0.55).extrude(W / 2 - 10, both=True)
        disc = cq.Workplane("XZ").circle(R * 0.56).extrude(6, both=True).translate((0, -W / 2 + 30, 0))
        for k in range(5):                                                 # five windows in the wheel's face, to see it turn
            a = 2 * math.pi * k / 5
            disc = disc.cut(cq.Workplane("XZ").center(R * 0.36 * math.cos(a), R * 0.36 * math.sin(a)).circle(R * 0.12).extrude(20, both=True)
                            .translate((0, -W / 2 + 30, 0)))
        hub = cq.Workplane("XZ").circle(55).extrude(W / 2 - 20, both=True)
        parts = [tyre, rim, disc, hub]
        if p["detail"]:
            for k in range(24):                                            # tread lugs, alternate chevrons
                a = 2 * math.pi * k / 24
                lug = cq.Workplane("XY").box(40, W * 0.42, 14).translate((0, (W * 0.22) * (1 if k % 2 else -1), R + 2))
                parts.append(lug.rotate((0, 0, 0), (0, 1, 0), math.degrees(a)))
        return _compound(parts)

    def upright(self, p):
        g = GEOM
        W = g["wheel_w"] * 1000
        y = -W / 2 - 40                                                    # inboard of the wheel (built for the left side)
        body = cq.Workplane("XY").box(70, 50, 260).translate((0, y, 0)).edges("|Y").fillet(15)
        spindle = cq.Workplane("XZ").circle(28).extrude(80).translate((0, y + 20, 0))
        caliper = cq.Workplane("XY").box(60, 40, 90).translate((-150, y + 60, 60))
        return _compound([body, spindle, caliper])

    def wishbone(self, p):
        """A lower A-arm, pivots on the frame at y = 0, the ball joint at y = +L (built along +y)."""
        L = (GEOM["track"] / 2 - GEOM["frame_y"] - GEOM["wheel_w"] / 2 - 0.06) * 1000
        return _compound([_tube((-180, 0, 0), (0, L, 0), 15), _tube((180, 0, 0), (0, L, 0), 15),
                          cq.Workplane("XZ").circle(25).extrude(30, both=True).translate((0, L, 0))])

    def coilover(self, p):
        """A coilover 420 mm eye to eye along z: the body, the shaft and the spring (a helix swept with a wire)."""
        body = cq.Workplane("XY").circle(24).extrude(230)
        shaft = cq.Workplane("XY").circle(9).extrude(190).translate((0, 0, 230))
        helix = cq.Wire.makeHelix(pitch=38, height=300, radius=40)
        spring = cq.Workplane("XY").center(40, 0).circle(7).sweep(cq.Workplane(obj=helix), isFrenet=True).translate((0, 0, 70))
        return _compound([body, shaft, spring])

    def seat(self, p):
        g = GEOM
        top = g["seat_top"] * 1000
        x0 = g["pelvis"][0] * 1000
        cush = cq.Workplane("XY").box(560, 620, 100).translate((x0 + 40, 0, top - 50)).edges().fillet(30)
        bolster = cq.Workplane("XY").box(110, 520, 170).translate((x0 - 270, 0, top + 30)).edges().fillet(35)
        tank = cq.Workplane("XY").box(360, 380, 210).translate((x0 + 470, 0, top - 30)).edges().fillet(60)   # the electronics' hump between the knees
        return _compound([cush, bolster, tank])

    def trunk(self, p):
        t = GEOM["trunk"]
        L, Wd, H, w = t["len"] * 1000, t["w"] * 1000, t["h"] * 1000, t["wall"] * 1000
        outer = cq.Workplane("XY").box(L, Wd, H).edges("|Z").fillet(20)
        inner = cq.Workplane("XY").box(L - 2 * w, Wd - 2 * w, H).translate((0, 0, w))
        tray = outer.cut(inner)                                            # open on top: no lid
        rail = cq.Workplane("XY").box(L + 30, 24, 24).translate((0, 0, H / 2))
        tray = tray.union(rail.translate((0, Wd / 2, 0))).union(rail.translate((0, -Wd / 2, 0)))
        return tray.translate((t["x"] * 1000, 0, t["floor_z"] * 1000 + H / 2))

    def tools(self, p):
        t = GEOM["trunk"]
        x, z = t["x"] * 1000, t["floor_z"] * 1000 + 6
        wrench = _compound([cq.Workplane("XY").box(320, 30, 10).translate((x - 40, 120, z + 5)),
                            cq.Workplane("XY").circle(30).circle(16).extrude(10).translate((x + 120, 120, z))])
        hammer = _compound([cq.Workplane("YZ").circle(14).extrude(340).translate((x - 170, -150, z + 14)),
                            cq.Workplane("XY").box(50, 130, 45).translate((x + 175, -150, z + 22))])
        box = cq.Workplane("XY").box(200, 140, 110).translate((x - 60, -10, z + 55)).edges("|Z").fillet(10)
        return _compound([wrench, hammer, box])

    def handlebar(self, p):
        """About the column's pivot (``GEOM['column_base']``): the stem along the raked column, the risers, the bar
        across with its grips (gorilla-fist size, 50 mm), the lights' pod."""
        g = GEOM
        base = np.asarray(g["column_base"])
        top = (np.array([g["grip_x"], 0.0, g["grip_z"] - 0.05]) - base)
        stem = _tube((0, 0, 0), tuple(top * 1000), 22)
        gl, gr = (handlebar_grip("L") - base) * 1000, (handlebar_grip("R") - base) * 1000
        inner_l, inner_r = gl - np.array([0, 140, 0]), gr + np.array([0, 140, 0])
        bar = [_tube(tuple(top * 1000), tuple(inner_l), 14), _tube(tuple(top * 1000), tuple(inner_r), 14),
               _tube(tuple(inner_l), tuple(gl + [0, 120, 0]), 14), _tube(tuple(inner_r), tuple(gr - [0, 120, 0]), 14),
               _tube(tuple(gl - [0, 110, 0]), tuple(gl + [0, 110, 0]), g["grip_r"] * 1000),
               _tube(tuple(gr - [0, 110, 0]), tuple(gr + [0, 110, 0]), g["grip_r"] * 1000)]
        pod = cq.Workplane("XY").box(110, 260, 90).translate(tuple(top * 1000 + [70, 0, -40])).edges().fillet(20)
        lights = [cq.Workplane("YZ").circle(34).extrude(20).translate(tuple(top * 1000 + [125, s * 70, -40])) for s in (-1, 1)]
        return _compound([stem, *bar, pod, *lights])

    def frame(self, p):
        """Everything on the sprung body, in the vehicle frame."""
        g, r = GEOM, p["tube_r"]
        mm = 1000.0
        fy, zr, zt = g["frame_y"] * mm, g["rail_z"] * mm, g["top_rail_z"] * mm
        xf, xr = 0.95 * mm, -1.05 * mm
        tubes = []
        for s in (1, -1):
            y = s * fy
            pts = [(xr, y, zr), (-0.55 * mm, y, zr - 40), (0.45 * mm, y, zr - 40), (xf, y, zr + 60)]
            tubes += [_tube(a, b, r) for a, b in zip(pts, pts[1:])]                       # lower rail
            top = [(xr, y * 0.95, zt + 30), (-0.40 * mm, y * 0.95, zt), (0.40 * mm, y * 0.8, zt - 20), (0.70 * mm, y * 0.55, 0.62 * mm)]
            tubes += [_tube(a, b, r) for a, b in zip(top, top[1:])]                     # upper rail, up to the steering head
            for x in (-0.95, -0.55, -0.15, 0.35, 0.85):                                 # uprights between the rails
                lo = min(pts, key=lambda q: abs(q[0] - x * mm))
                tubes.append(_tube((x * mm, y, lo[2]), (x * mm, y * (0.95 if x < 0 else 0.8), zt - (0 if x < 0.3 else 30)), r * 0.8))
            tubes.append(_tube((xr, y, zr), (xr, y * 0.95, zt + 30), r))
            # footboards: grating plates on outriggers, a heel guard
            fx = g["footboard_x"] * mm
            fyb = s * g["footboard_y"] * mm
            tubes.append(_tube((fx - 250, y, zr - 40), (fx - 250, fyb + s * 160, g["footboard_z"] * mm - 20), r * 0.8))
            tubes.append(_tube((fx + 250, y, zr - 40), (fx + 250, fyb + s * 160, g["footboard_z"] * mm - 20), r * 0.8))
        for x in (xr, -0.55 * mm, 0.0, 0.45 * mm, xf):                                    # cross members
            tubes.append(_tube((x, -fy, zr - (40 if -0.6 * mm < x < 0.5 * mm else 0) + (60 if x == xf else 0)),
                               (x, fy, zr - (40 if -0.6 * mm < x < 0.5 * mm else 0) + (60 if x == xf else 0)), r))
        # the steering head
        base = np.asarray(g["column_base"]) * mm
        head = base - 260 * _column_axis()
        tubes.append(_tube(tuple(head), tuple(base - 20 * _column_axis()), 30))
        # bumpers
        tubes += [_tube((xf + 120, -0.55 * mm, 0.42 * mm), (xf + 120, 0.55 * mm, 0.42 * mm), 22),
                  _tube((xf, -0.3 * mm, 0.36 * mm), (xf + 120, -0.45 * mm, 0.42 * mm), 18),
                  _tube((xf, 0.3 * mm, 0.36 * mm), (xf + 120, 0.45 * mm, 0.42 * mm), 18)]
        shapes = [cq.Workplane(obj=t) for t in tubes]
        for s in (1, -1):
            fb = cq.Workplane("XY").box(g["footboard_len"] * mm, g["footboard_w"] * mm, 8).translate(
                (g["footboard_x"] * mm, s * g["footboard_y"] * mm, g["footboard_z"] * mm - 4))
            for k in range(6):
                fb = fb.cut(cq.Workplane("XY").box(30, g["footboard_w"] * mm - 60, 20).translate(
                    (g["footboard_x"] * mm - 180 + 72 * k, s * g["footboard_y"] * mm, g["footboard_z"] * mm - 4)))
            shapes.append(fb)
            shapes.append(cq.Workplane("XY").box(30, g["footboard_w"] * mm, 90).translate(
                (g["footboard_x"] * mm - g["footboard_len"] * mm / 2, s * g["footboard_y"] * mm, g["footboard_z"] * mm + 45)))
        b = g["battery"]
        shapes.append(cq.Workplane("XY").box(b["len"] * mm, b["w"] * mm, b["h"] * mm).translate((b["x"] * mm, 0, (b["z"] + b["h"] / 2) * mm - 40))
                      .edges("|X").fillet(20))
        # four motors inboard, each with its inverter
        for c in CORNERS:
            x, y = _corner_xy(c)
            sy = 1 if y > 0 else -1
            mx = x * mm
            shapes.append(cq.Workplane("XZ").circle(80).extrude(60, both=True).translate((mx, sy * (fy - 70), g["wheel_r"] * mm)))
            shapes.append(cq.Workplane("XY").box(140, 90, 60).translate((mx - (180 if c[0] == "F" else -180), sy * (fy - 90), g["wheel_r"] * mm + 60)))
        # fenders over the wheels, clear of the full bump travel
        for c in CORNERS:
            x, y = _corner_xy(c)
            R = (g["wheel_r"] + g["bump"] + 0.04) * mm
            fen = (cq.Workplane("XZ").circle(R + 5).circle(R).extrude(g["wheel_w"] * mm / 2 + 30, both=True)
                   .cut(cq.Workplane("XY").box(3 * R, 3 * R, 3 * R).translate((0, 0, -1.5 * R + 0.15 * R))))
            shapes.append(fen.translate((x * mm, y * mm, g["wheel_r"] * mm)))
            sy = 1 if y > 0 else -1
            shapes.append(cq.Workplane(obj=_tube((x * mm, sy * fy, 0.62 * mm), (x * mm, sy * (abs(y) * mm - 100), R + g["wheel_r"] * mm + 60), 12)))   # the fender's stay
        # the nose: a front rack with the electronics' cover
        shapes.append(cq.Workplane("XY").box(380, 640, 120).translate((0.80 * mm, 0, 0.62 * mm)).edges("|Z").fillet(40))
        return _compound(shapes)

    def build(self, p):
        part = p["part"]
        if part != "quad":
            return getattr(self, part)(p)
        g = GEOM
        mm = 1000.0
        shapes = [self.frame(p), self.seat(p), self.trunk(p).val(), self.tools(p)]
        hb = self.handlebar(p).translate(cq.Vector(*(np.asarray(g["column_base"]) * mm)))
        shapes.append(hb)
        wheel, up, arm, coil = self.wheel(p), self.upright(p), self.wishbone(p), self.coilover(p)
        for c in CORNERS:
            x, y = _corner_xy(c)
            sy = 1 if y > 0 else -1
            mirror = (lambda s: s.mirror("XZ")) if sy < 0 else (lambda s: s)
            shapes.append(wheel.translate(cq.Vector(x * mm, y * mm, g["wheel_r"] * mm)))
            shapes.append(mirror(up).translate(cq.Vector(x * mm, y * mm, g["wheel_r"] * mm)))
            for dz in (-90, 110):
                shapes.append(mirror(arm).translate(cq.Vector(x * mm, sy * g["frame_y"] * mm, g["wheel_r"] * mm + dz)))
            shapes.append(coil.translate(cq.Vector(x * mm + (-80 if c[0] == "F" else 80), sy * (g["frame_y"] * mm + 180), g["wheel_r"] * mm - 80)))
        return _compound(shapes)


# ---------------------------------------------------------------- the MuJoCo model
COLORS = {"frame": (0.93, 0.45, 0.10, 1.0), "seat": (0.12, 0.12, 0.13, 1.0), "tyre": (0.09, 0.09, 0.10, 1.0), "rim": (0.72, 0.74, 0.77, 1.0),
          "bar": (0.20, 0.21, 0.23, 1.0), "trunk": (0.32, 0.34, 0.37, 1.0), "battery": (0.13, 0.30, 0.55, 1.0), "upright": (0.55, 0.57, 0.60, 1.0),
          "tools": (0.85, 0.15, 0.12, 1.0)}


def quad_prop(x0=0.0, y0=0.0, *, rigid=False, sz=None) -> Prop:
    """The quad as a free Chiron prop: the sprung body (``quad_chassis``), the handlebar on its raked column
    (``quad_bar``, hinge ``steer_column``), and per corner a body sliding along z on the spring and damper
    (``corner_XX``, slide ``susp_XX``; ``rigid`` removes the slides), the front ones carrying a steering knuckle
    (``knuckle_XX``, hinge ``steer_XX``), and the wheel on its axle (``wheel_XX``, hinge ``spin_XX``, motor-driven in the
    sim). The collision shapes are simple; the drawings use the CAD (``quad_mesh``)."""
    s = sz or sizing()
    g = GEOM
    R, W = g["wheel_r"], g["wheel_w"]
    corners = []
    m_u = s["unsprung_corner_kg"]
    for c in CORNERS:
        x, y = _corner_xy(c)
        k, damp, sag = s["corners"][c]["k_N_m"], s["corners"][c]["c_Ns_m"], s["corners"][c]["sag_m"]
        wheel = Link(f"wheel_{c}", joints=[Joint(f"spin_{c}", axis=(0, 1, 0), damping=0.05, armature=0.02 * 36)],     # the rotor's inertia through 6:1
                     geoms=[Geom(f"tyre_{c}", "sphere", (R,), mass=UNSPRUNG["wheel + tyre 26x10-12"],          # a sphere rolls cleanly over a heightfield (the CAD draws the tyre)
                                 friction=(0.9, 0.01, 0.001), rgba=COLORS["tyre"])])
        upright = [Geom(f"upright_{c}", "box", (0.035, 0.025, 0.13), pos=(0.0, -np.sign(y) * (W / 2 + 0.04), 0.0), role="visual",
                        mass=UNSPRUNG["upright, hub, brake disc + caliper"] + UNSPRUNG["wishbones, coilover, half-shaft (unsprung half)"], rgba=COLORS["upright"])]
        if c[0] == "F":
            inner = Link(f"knuckle_{c}", joints=[Joint(f"steer_{c}", axis=(0, 0, 1), range=(-0.85, 0.85), damping=2.0)],
                         geoms=upright, children=[wheel])
            corner_geoms = [Geom(f"corner_{c}", "sphere", (0.03,), mass=0.5, role="visual", rgba=COLORS["upright"])]
            children = [inner]
        else:
            corner_geoms, children = upright, [wheel]
        joints = [] if rigid else [Joint(f"susp_{c}", kind="slide", axis=(0, 0, 1), range=(-g["droop"], g["bump"]),
                                         stiffness=k, damping=damp, springref=-sag)]
        corners.append(Link(f"corner_{c}", pos=(x, y, R), joints=joints, geoms=corner_geoms, children=children))
    m_s = sum(MASS.values())
    b = g["battery"]
    chassis_geoms = [
        Geom("q_battery", "box", (b["len"] / 2, b["w"] / 2, b["h"] / 2), pos=(b["x"], 0.0, b["z"] + b["h"] / 2 - 0.04),
             mass=MASS["battery 72 V 100 Ah LiFePO4 (7.2 kWh)"], role="visual", rgba=COLORS["battery"]),
        Geom("q_frame", "box", (1.0, g["frame_y"], 0.2), pos=(-0.05, 0.0, 0.48), mass=MASS["frame (4130 chromoly tube, welded)"] + MASS["inverters 4x, VCU, DC-DC, wiring"]
             + MASS["wishbones, coilovers (sprung half)"] + MASS["footboards, bodywork, fenders, seat base and scissor"], role="visual", rgba=COLORS["frame"]),
        Geom("q_trunk", "box", (g["trunk"]["len"] / 2, g["trunk"]["w"] / 2, 0.05), pos=(g["trunk"]["x"], 0.0, g["trunk"]["floor_z"] + 0.05),
             mass=MASS["trunk tray (Al 4 mm) and tools"], role="visual", rgba=COLORS["trunk"]),
    ]
    masses = [PointMass(f"motor_{c}", 12.0, (_corner_xy(c)[0], np.sign(_corner_xy(c)[1]) * (g["frame_y"] - 0.07), R)) for c in CORNERS]
    assert abs(sum(gm.mass for gm in chassis_geoms) + sum(pm.mass for pm in masses) - m_s + MASS["handlebar, column, steering rack"]) < 1e-6
    base = g["column_base"]
    bar = Link("quad_bar", pos=tuple(base), joints=[Joint("steer_column", axis=tuple(_column_axis()), range=(-g["steer_max"], g["steer_max"]), damping=3.0)],
               geoms=[Geom("q_bar", "capsule", (0.02,), fromto=tuple((handlebar_grip("R") - base)) + tuple((handlebar_grip("L") - base)),
                           mass=MASS["handlebar, column, steering rack"], role="visual", rgba=COLORS["bar"])])
    st = s["seat"]
    seat = Link("quad_seat", pos=(g["pelvis"][0], 0.0, g["seat_top"] - 0.05),
                joints=[] if rigid else [Joint("seat_slide", kind="slide", axis=(0, 0, 1), range=(-SEAT_TRAVEL, SEAT_TRAVEL), stiffness=st["k_N_m"],
                                                damping=st["c_Ns_m"], springref=st["sag_m"])],    # preloaded upward: it carries the rider
                geoms=[Geom("q_seat", "box", (0.28, 0.31, 0.05), mass=SEAT_MOVING_KG, role="visual", rgba=COLORS["seat"])])
    root = Link("quad_chassis", pos=(x0, y0, 0.0), geoms=chassis_geoms, masses=masses, children=[bar, seat, *corners])
    return Prop(root, free=True, priority=3, condim=4, solref=TYRE_SOLREF)


# ---------------------------------------------------------------- the drawings' mesh
def _tess(shape, tol=2.5, ang=0.35):
    shape = shape.val() if isinstance(shape, cq.Workplane) else shape
    v, t = shape.tessellate(tol, ang)
    v = np.array([[p.x, p.y, p.z] for p in v]) * 1e-3
    return v[np.array(t)] if len(t) else np.zeros((0, 3, 3))


def quad_mesh(detail=1):
    """(pieces, struts) for ``kerchak_render``: the CAD tessellated into each quad body's frame; the wishbones, the
    coilovers and the half-shafts as struts between the sprung body and the corners (they stretch with the travel)."""
    d = GorillaQuad()
    p = d.resolve(detail=detail)
    g = GEOM
    pieces = []
    pieces.append(("quad_chassis", _tess(d.frame(p)), COLORS["frame"]))
    seat_pos = np.array([g["pelvis"][0], 0.0, g["seat_top"] - 0.05])
    pieces.append(("quad_seat", _tess(d.seat(p)) - seat_pos, COLORS["seat"]))
    pieces.append(("quad_chassis", _tess(d.trunk(p)), COLORS["trunk"]))
    pieces.append(("quad_chassis", _tess(d.tools(p)), COLORS["tools"]))
    pieces.append(("quad_bar", _tess(d.handlebar(p)), COLORS["bar"]))
    wheel = _tess(d.wheel(p), tol=3.0)
    # colour the wheel: the tyre black, the rim/disc silver (by radius from the axle)
    rr = np.linalg.norm(wheel[:, :, [0, 2]].mean(axis=1), axis=1)
    tyre, rim = wheel[rr > g["wheel_r"] * 0.6], wheel[rr <= g["wheel_r"] * 0.6]
    up = _tess(d.upright(p))
    struts = []
    for c in CORNERS:
        x, y = _corner_xy(c)
        sy = 1.0 if y > 0 else -1.0
        mirror = np.array([1.0, sy, 1.0])
        carrier = f"knuckle_{c}" if c[0] == "F" else f"corner_{c}"
        pieces.append((f"wheel_{c}", tyre * mirror, COLORS["tyre"]))
        pieces.append((f"wheel_{c}", rim * mirror, COLORS["rim"]))
        pieces.append((carrier, up * mirror, COLORS["upright"]))
        cx, cz = x, g["wheel_r"]
        ub = sy * (g["wheel_w"] / 2 + 0.06)
        for dz, rb in ((-0.09, 0.016), (0.11, 0.014)):                       # the wishbones: two legs each
            for dx in (-0.18, 0.18):
                struts.append(("quad_chassis", np.array([cx + dx, sy * g["frame_y"], cz + dz]), f"corner_{c}",
                               np.array([0.0, -ub, dz * 0.8]), rb, rb))
        struts.append(("quad_chassis", np.array([cx + (-0.08 if c[0] == "F" else 0.08), sy * (g["frame_y"] + 0.08), cz + 0.36]),
                       f"corner_{c}", np.array([(-0.08 if c[0] == "F" else 0.08), -ub * 1.2, -0.08]), 0.045, 0.012))   # the coilover: body, shaft
        struts.append(("quad_chassis", np.array([cx, sy * (g["frame_y"] - 0.0), cz]), f"wheel_{c}", np.array([0.0, -sy * (g["wheel_w"] / 2), 0.0]),
                       0.022, 0.018))                                                                            # the half-shaft
    return pieces, struts
