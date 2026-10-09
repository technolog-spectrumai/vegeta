"""NISUS — a 1.4 m twin-boom pusher for observation (Nisus-OBS) and onboard computer vision (Nisus-Zero), notebook 31.

The aircraft: a continuous straight-tapered wing (1400 mm span, 230/170 mm chords, 0.28 m², AR 7) on a rounded central
pod (450 x 100 x 110 mm) that carries the battery, the electronics and the camera in the nose; one electric pusher
propeller (9 inch) on the pod's tail between two carbon tail booms (10 mm tubes, 320 mm apart, 580 mm long including
their sockets); a horizontal tail (400 x 140 mm) joining the booms with a full-span elevator, two fins with rudders on
the boom ends, a belly skid. No folding wings. Both variants share this geometry; they differ in what the pod carries
(``nisus_systems``).

Nothing is designed twice: the wing (NACA 4-digit loft with dihedral and a constant-chord centre section) is notebook
09's ``FixedWing._wing``; the sections are its ``naca4``. New here: the pod (a loft of ellipses whose axis rises into
the motor), the booms with their printed root fittings, the H-tail on the booms, the motor mount, the skid, the spar,
the trays, and the layout arithmetic (stations, areas, tail volumes, the propeller's clearances, the component bays).

Frame (as ``FixedWing``): wing root leading edge at the origin, chord along +X (the free stream along +X, the nose at
-X), span along Y (+Y the right wing), up +Z; units mm. The pod axis is at ``z = -pod_height/2`` ahead of the wing and
rises to ``motor_z`` at the tail (the propeller sits high so the belly skid, not the blades, meets the ground).
Angles: ``wing_incidence_deg`` turns the wing nose-up about its root quarter chord; ``motor_downthrust_deg`` tilts the
thrust line nose-down (its line then passes nearer the centre of gravity, which sits below the motor axis).

``part``: ``aircraft`` (everything but the propeller: CFD, pictures, masses), ``wing``, ``spar`` (the carbon spar tube
in span-wise segments, so the FEA can load each segment with its share of the lift), ``rear_spar`` (the rear
carry-through tube the boom fittings also clamp), ``pod`` (the pod shell with its
nose, hollow), ``nose`` (the replaceable nose cone, hollow), ``boom`` (one tube), ``boom_fitting`` (one printed root
fitting), ``tail`` (stabiliser with both fins), ``tail_fitting`` (one printed boom-end sleeve), ``motor_mount``
(the printed cup on the pod's tail), ``tray`` (the electronics tray), ``skid`` (the belly keel), ``battery_tray``.

Beside the CAD: ``layout`` (stations, areas, volumes, arms, clearances), ``bays`` (where the components go inside the
pod), ``planform`` (the lifting surfaces as flat quads for the vortex lattice), ``wetted_areas`` and ``drag_buildup``
(the parasite drag estimate), ``outline`` (three-view polygons for the drawings and the movie), ``exploded_parts``,
``boom_check`` (the boom tube by hand: stress, deflection, frequency, the propeller clearance after deflection).
"""
from __future__ import annotations

import math

import cadquery as cq
import numpy as np
from vegeta.dedalus import Design, Parameter

from fixed_wing import FixedWing, naca4

VARIANTS = ("OBS", "Zero")
VARIANT_NAME = {"OBS": "Nisus-OBS (remotely piloted observation, live video)",
                "Zero": "Nisus-Zero (onboard computer vision on a Jetson, live video)"}


def _fw(name, default=None, **kw):
    """FixedWing's parameter ``name`` with a new default where NISUS differs."""
    p = next(q for q in FixedWing.parameters if q.name == name)
    f = dict(name=p.name, default=p.default, units=p.units, min=p.min, max=p.max, description=p.description, choices=p.choices)
    if default is not None:
        f["default"] = default
    f.update(kw)
    return Parameter(**f)


PARTS = ("aircraft", "wing", "spar", "rear_spar", "pod", "nose", "boom", "boom_fitting", "tail", "tail_fitting", "motor_mount",
         "tray", "skid", "battery_tray")
FEA_PARTS = ("boom_fitting_fea", "boom_fea", "motor_mount_fea", "spar_fea", "rear_spar_fea")
# boom_fitting_fea: the fitting with the socket bore in two halves (separate faces for the bearing loads); boom_fea: the
# boom tube in three pieces (in the socket, the free span, under the tail sleeve); motor_mount_fea: the mount untilted
# (the FEA rotates the loads into its frame); spar_fea / rear_spar_fea: the right half of each tube, straight along its
# outboard line from y = 0 (no dihedral kink: the kink's sliver faces spoil the mesh; the load is symmetric and the
# saddle clamps the centre, so the half model is exact for the clamped case)


class Nisus(FixedWing):
    """NISUS: a twin-boom single-pusher fixed wing with ``FixedWing``'s wing."""

    parameters = [
        Parameter("part", "aircraft", choices=PARTS + FEA_PARTS, description="what to build"),
        Parameter("show_prop", False, description="aircraft: add the propeller disc (a thin cylinder) for pictures"),
        _fw("span", 1400.0),
        _fw("root_chord", 230.0),
        Parameter("tip_chord", 170.0, "mm", min=30),
        _fw("dihedral_deg", 3.0),
        Parameter("wing_incidence_deg", 1.5, "deg", min=-5, max=10, description="wing root chord nose-up to the pod axis (the pod flies level near 15 m/s)"),
        _fw("camber", 0.03, description="NACA max camber (3412: 0.03)"),
        _fw("camber_pos", 0.4),
        _fw("thickness", 0.12),
        Parameter("spar_x_frac", 0.30, "", min=0.15, max=0.5, description="main spar at this fraction of the local chord"),
        Parameter("spar_od", 10.0, "mm", min=3), Parameter("spar_id", 8.0, "mm", min=0),
        Parameter("spar_half_length", 500.0, "mm", min=100, description="the carbon tube reaches this far from the centre"),
        Parameter("rear_spar_x_frac", 0.65, "", min=0.4, max=0.8, description="rear carry-through tube at this fraction of the local chord"),
        Parameter("rear_spar_od", 10.0, "mm", min=3), Parameter("rear_spar_id", 8.0, "mm", min=0),
        Parameter("rear_spar_half_length", 175.0, "mm", min=60, description="rear tube: through the pod to just outboard of the boom fittings"),
        Parameter("aileron_chord_frac", 0.25, "", min=0.1, max=0.5),
        Parameter("aileron_span_frac", 0.45, "", min=0.1, max=0.9, description="outer share of the half span with an aileron"),
        Parameter("pod_length", 550.0, "mm", min=200, description="the 450 mm starting pod could not balance (its CG sat near 60 % MAC with the battery as far forward as it goes): the nose grew 100 mm"),
        Parameter("pod_width", 100.0, "mm", min=40),
        Parameter("pod_height", 110.0, "mm", min=40),
        Parameter("nose_length", 300.0, "mm", min=50, description="pod nose tip ahead of the wing leading edge (the battery and, on Zero, the computer sit in it)"),
        Parameter("nose_cone_length", 70.0, "mm", min=20, description="the replaceable nose (camera) cone"),
        Parameter("pod_wall", 1.2, "mm", min=0.6, description="printed pod shell (LW-PLA)"),
        Parameter("boom_y", 160.0, "mm", min=50, description="boom centre from the symmetry plane (320 mm spacing)"),
        Parameter("boom_od", 10.0, "mm", min=4), Parameter("boom_id", 8.0, "mm", min=0),
        Parameter("boom_length", 550.0, "mm", min=200, description="tube length including the socket overlap (580 in the starting layout; 30 mm shorter to hold the overall length near 900 mm with the longer nose)"),
        Parameter("boom_x0", 135.0, "mm", description="tube front end (inside the root fitting's socket)"),
        Parameter("boom_z", -20.0, "mm", description="boom axis height (under the wing)"),
        Parameter("fitting_width", 14.0, "mm", min=10, description="boom root fitting: width of its two clamp rings and web along y"),
        Parameter("fitting_ring_wall", 2.5, "mm", min=1.2, description="boom root fitting: wall of the rings around the spar tubes"),
        Parameter("fitting_web", 10.0, "mm", min=3, description="boom root fitting: thickness of the web joining the rings and the socket"),
        Parameter("socket_od", 18.0, "mm", min=12, description="boom root fitting: socket tube outside diameter (wall = (socket_od - boom_od)/2)"),
        Parameter("tail_span", 400.0, "mm", min=100),
        Parameter("tail_chord", 140.0, "mm", min=40),
        Parameter("tail_thickness", 0.05, "", min=0.03, max=0.15, description="tail section NACA 00xx thickness ratio (0.05: a 7 mm foam plate at 140 mm chord)"),
        Parameter("tail_incidence_deg", 1.5, "deg", min=-6, max=6, description="stabiliser chord to the pod axis (same as the wing: the wing's downwash gives the tail about -2° in cruise)"),
        Parameter("elevator_frac", 0.35, "", min=0.1, max=0.6),
        Parameter("fin_height", 140.0, "mm", min=40, description="fin above the boom axis"),
        Parameter("fin_ventral", 60.0, "mm", min=0, description="ventral fin below the boom axis: its lower edge is the tail bumper the aircraft rests on"),
        Parameter("fin_chord", 140.0, "mm", min=40),
        Parameter("rudder_frac", 0.35, "", min=0.1, max=0.6),
        Parameter("motor_diameter", 28.0, "mm", min=10, description="motor bell (X2216 class: 27.9 mm)"),
        Parameter("motor_length", 34.0, "mm", min=10),
        Parameter("motor_z", 0.0, "mm", description="thrust line height at the pod's tail (at the pod's top line: the propeller clears the ground with the aircraft resting on its skid and tail bumpers)"),
        Parameter("motor_downthrust_deg", 6.0, "deg", min=-5, max=12, description="the thrust line tilted nose-down so it passes near the CG (which sits ~37 mm below and 0.23 m ahead of the propeller)"),
        Parameter("mount_thickness", 4.0, "mm", min=2),
        Parameter("prop_diameter", 228.6, "mm", min=100, description="9 inch"),
        Parameter("prop_gap", 8.0, "mm", min=2, description="motor bell end to the propeller plane"),
        Parameter("skid_depth", 45.0, "mm", min=0, description="belly keel below the pod's lowest line"),
        Parameter("skid_x0", -40.0, "mm", description="keel front"), Parameter("skid_x1", 110.0, "mm", description="keel rear (its rear corner is the main ground contact, just behind the CG)"),
        _fw("angle_of_attack_deg"),
    ]

    # ------------------------------------------------------------------------------------------- numbers
    @staticmethod
    def _fwp(p) -> dict:
        """``p`` with the names FixedWing's ``_wing`` reads."""
        return dict(p, taper=p["tip_chord"] / p["root_chord"], fuselage_diameter=p["pod_width"])

    @staticmethod
    def layout(p) -> dict:
        """Stations [mm], areas [m²], arms and volumes, clearances — the numbers everything else uses."""
        c0, c1, b = p["root_chord"], p["tip_chord"], p["span"]
        b2, lam = b / 2, c1 / c0
        yc = p["pod_width"] / 2 + 5.0                                   # FixedWing's centre-section half width
        S = b * (c0 + c1) / 2 * 1e-6
        mac = (2 / 3) * c0 * (1 + lam + lam ** 2) / (1 + lam)
        y_mac = b2 * (1 + 2 * lam) / (3 * (1 + lam))
        le_sweep = (c0 - c1) / 4                                         # tip LE offset: a straight quarter-chord line
        x_mac_le = le_sweep * y_mac / b2
        x_ac = x_mac_le + 0.25 * mac
        x_nose = -p["nose_length"]
        x_pod_end = p["pod_length"] - p["nose_length"]
        x_m0 = x_pod_end + p["mount_thickness"]
        x_m1 = x_m0 + p["motor_length"]
        prop_x = x_m1 + p["prop_gap"]
        R = p["prop_diameter"] / 2
        boom_x1 = p["boom_x0"] + p["boom_length"]
        tail_le, tail_te = boom_x1 - p["tail_chord"], boom_x1
        x_ac_t = tail_le + 0.25 * p["tail_chord"]
        fin_le = boom_x1 - p["fin_chord"]
        x_ac_v = fin_le + 0.25 * p["fin_chord"]
        S_h = p["tail_span"] * p["tail_chord"] * 1e-6
        S_v1 = (p["fin_height"] + p["fin_ventral"]) * p["fin_chord"] * 1e-6
        l_t, l_v = x_ac_t - x_ac, x_ac_v - x_ac
        # control surfaces
        y_a0 = yc + (1 - p["aileron_span_frac"]) * (b2 - yc)
        c_a0, c_a1 = c0 + (c1 - c0) * (y_a0 / b2), c1
        S_ail1 = p["aileron_chord_frac"] * (c_a0 + c_a1) / 2 * (b2 - y_a0) * 1e-6
        # propeller clearances (nominal: no deflection, no tolerance)
        d_boom = math.hypot(p["boom_y"], p["boom_z"] - p["motor_z"])
        clear_boom = d_boom - p["boom_od"] / 2 - R
        belly = -p["pod_height"]
        prop_low = p["motor_z"] - R
        skid_low = belly - p["skid_depth"]
        # at rest on the keel's rear corner and the ventral fins' lower rear corners: the ground line under the propeller
        z_stub = p["boom_z"] - p["fin_ventral"]
        x_stub = p["boom_x0"] + p["boom_length"]
        z_line = skid_low + (prop_x - p["skid_x1"]) * (z_stub - skid_low) / (x_stub - p["skid_x1"])
        rest_pitch = math.degrees(math.atan2(z_stub - skid_low, x_stub - p["skid_x1"]))
        return {"S_ref": S, "AR": b ** 2 * 1e-6 / S, "mac": mac, "y_mac": y_mac, "x_mac_le": x_mac_le, "x_ac": x_ac,
                "yc": yc, "b2": b2, "taper": lam, "le_sweep_tip": le_sweep,
                "x_nose": x_nose, "x_pod_end": x_pod_end, "pod_axis_z": -p["pod_height"] / 2, "x_nose_joint": x_nose + p["nose_cone_length"],
                "x_motor0": x_m0, "x_motor1": x_m1, "prop_x": prop_x, "prop_R": R, "motor_z": p["motor_z"],
                "boom_x1": boom_x1, "tail_le": tail_le, "tail_te": tail_te, "fin_le": fin_le, "boom_z": p["boom_z"],
                "x_ac_tail": x_ac_t, "x_ac_fin": x_ac_v, "l_t": l_t, "l_v": l_v,
                "S_h": S_h, "S_v_each": S_v1, "S_v": 2 * S_v1, "V_h": S_h * l_t / (S * mac), "V_v": 2 * S_v1 * l_v / (S * b),
                "S_elevator": p["elevator_frac"] * S_h, "S_rudder_each": p["rudder_frac"] * S_v1,
                "S_aileron_each": S_ail1, "y_aileron0": y_a0,
                "overall_length": tail_te - x_nose, "height_over_skid": p["fin_height"] + p["boom_z"] - skid_low,
                "prop_clearance_boom": clear_boom, "prop_clearance_wing_te": prop_x - c0,
                "prop_lowest_z": prop_low, "skid_lowest_z": skid_low, "prop_ground_margin": prop_low - skid_low,
                "prop_ground_margin_resting": prop_low - z_line, "resting_pitch_deg": rest_pitch, "tail_bumper_z": z_stub,
                "prop_tip_z_top": p["motor_z"] + R, "pod_top_z": 0.0, "spar_x_root": p["spar_x_frac"] * c0}

    @staticmethod
    def pod_profile(p, n=28) -> np.ndarray:
        """The pod's sections along x: columns (x, a, b, zc) — half width, half height and the centre height of the
        ellipse at each station (the loft is ruled between them: a smooth loft overshoots the small end sections). An elliptic nose over the first 30 % of the length, a constant body, then a tail
        over the last 35 % where the section shrinks (a cubic) to the motor-mount circle and the axis rises to
        ``motor_z``."""
        L = Nisus.layout(p)
        a0, b0, z0 = p["pod_width"] / 2, p["pod_height"] / 2, L["pod_axis_z"]
        Ln, Lt = 0.30 * p["pod_length"], 0.35 * p["pod_length"]
        x0, x1 = L["x_nose"], L["x_pod_end"]
        r_end = p["motor_diameter"] / 2 + 2.0 + p["pod_wall"]
        rows = []
        for t in np.linspace(0, 1, n):                                  # the nose: ellipse quarter
            s = math.sin(t * math.pi / 2)
            rows.append((x0 + Ln * (1 - math.cos(t * math.pi / 2)), max(a0 * s, 2.0), max(b0 * s, 2.2), z0))
        rows.append((x1 - Lt, a0, b0, z0))
        for t in np.linspace(0, 1, n)[1:]:                              # the tail: cubic taper, axis rising
            f = 3 * t ** 2 - 2 * t ** 3
            rows.append((x1 - Lt + Lt * t, a0 + (r_end - a0) * f, b0 + (r_end - b0) * f, z0 + (p["motor_z"] - z0) * f))
        out = [rows[0]] + [r for i, r in enumerate(rows[1:], 1) if r[0] - rows[i - 1][0] > 1e-6]
        return np.array(out)

    @staticmethod
    def bays(p) -> dict:
        """Where the components go inside the pod (x ranges [mm], the usable inner width/height there): the camera in
        the nose cone, the battery bay behind it (the pack slides along it to set the CG), the computer bay where
        Nisus-Zero's Jetson stands on its side beside the pack (OBS slides its flight-controller tray there), the
        flight-controller tray under the wing's front, the video transmitter, the ESC and wiring at the tail with the
        cooling vents."""
        L = Nisus.layout(p)
        w = p["pod_width"] - 2 * p["pod_wall"]
        h = p["pod_height"] - 2 * p["pod_wall"]
        x0 = L["x_nose"]
        return {"camera (nose cone)": (x0 + 10, x0 + p["nose_cone_length"] - 10.0, w * 0.6, h * 0.6),
                "battery bay": (x0 + p["nose_cone_length"] - 25.0, -15.0, w, h),
                "computer bay (Zero: the Jetson on its side beside the battery)": (-120.0, -15.0, w, h),
                "flight controller tray (Zero; OBS slides it into the computer bay)": (-10.0, 80.0, w, h),
                "video, ESC, wiring, vents": (80.0, L["x_pod_end"] - 15.0, w * 0.8, h * 0.8)}

    # ------------------------------------------------------------------------------------------- shapes
    @staticmethod
    def _section(wp, c, dx, dz, up, lo):
        pts_u = [(dx + x * c, dz + y * c) for x, y in up]
        pts_l = [(dx + x * c, dz + y * c) for x, y in lo]
        return (wp.moveTo(*pts_u[0]).spline(pts_u[1:], includeCurrent=True)
                .spline(pts_l[::-1][1:], includeCurrent=True).close())

    def _wing_solid(self, p):
        """FixedWing's wing (centre section + two tapered panels with dihedral), turned nose-up by the incidence
        about the root quarter chord."""
        w = FixedWing._wing(self, self._fwp(p))
        if p["wing_incidence_deg"]:
            x = 0.25 * p["root_chord"]
            w = w.rotate((x, 0, 0), (x, 1, 0), p["wing_incidence_deg"])
        return w

    def _pod_solid(self, p, inset=0.0, x_from=None, x_to=None):
        prof = self.pod_profile(p)
        if x_from is not None or x_to is not None:
            lo = prof[:, 0].min() if x_from is None else x_from
            hi = prof[:, 0].max() if x_to is None else x_to
            keep = prof[(prof[:, 0] >= lo - 1e-6) & (prof[:, 0] <= hi + 1e-6)]
            if x_from is not None and keep[0, 0] > lo + 1e-6:
                keep = np.vstack([[lo] + [float(np.interp(lo, prof[:, 0], prof[:, k])) for k in (1, 2, 3)], keep])
            if x_to is not None and keep[-1, 0] < hi - 1e-6:
                keep = np.vstack([keep, [hi] + [float(np.interp(hi, prof[:, 0], prof[:, k])) for k in (1, 2, 3)]])
            prof = keep
        wp = cq.Workplane("YZ")
        first = True
        for x, a, b, zc in prof:
            a_, b_ = max(a - inset, 0.8), max(b - inset, 0.8)
            wp = (cq.Workplane("YZ").workplane(offset=x) if first else wp.workplane(offset=x - prev_x)).center(0, zc).ellipse(a_, b_)
            wp = wp.center(0, -zc)                      # back to the axis for the next center() call
            first, prev_x = False, x
        return wp.loft(ruled=True, combine=True)

    def _pod_shell(self, p, x_from=None, x_to=None):
        outer = self._pod_solid(p, 0.0, x_from, x_to)
        inner = self._pod_solid(p, p["pod_wall"], x_from, x_to)
        return outer.cut(inner)

    def _boom_pieces(self, p, side=1):
        """The boom tube for the FEA in three pieces fused without cleaning: inside the socket (``boom_x0`` to the
        wing's trailing edge), the free span, under the tail sleeve (the last ``tail_chord``)."""
        L = self.layout(p)
        s = self.fitting_stations(p)
        y = side * p["boom_y"]
        cuts = [p["boom_x0"], s["x_te"], L["boom_x1"] - p["tail_chord"], L["boom_x1"]]
        parts = []
        for a, b in zip(cuts[:-1], cuts[1:]):
            t = cq.Solid.makeCylinder(p["boom_od"] / 2, b - a, cq.Vector(a, y, p["boom_z"]), cq.Vector(1, 0, 0))
            if p["boom_id"] > 0:
                t = t.cut(cq.Solid.makeCylinder(p["boom_id"] / 2, b - a + 0.2, cq.Vector(a - 0.1, y, p["boom_z"]), cq.Vector(1, 0, 0)))
            parts.append(t)
        return cq.Workplane("XY").add(parts[0].fuse(*parts[1:]))

    def _boom(self, p, side=1):
        L = self.layout(p)
        r = p["boom_od"] / 2
        tube = cq.Workplane("YZ").workplane(offset=p["boom_x0"]).center(side * p["boom_y"], p["boom_z"]).circle(r).extrude(p["boom_length"])
        if p["boom_id"] > 0:
            bore = cq.Workplane("YZ").workplane(offset=p["boom_x0"] - 1).center(side * p["boom_y"], p["boom_z"]).circle(p["boom_id"] / 2).extrude(p["boom_length"] + 2)
            tube = tube.cut(bore)
        return tube

    def _lower_surface_z(self, p, x_mm, y_mm):
        """The wing's lower-surface height at (x, y) [mm] in the aircraft frame (incidence and dihedral included)."""
        L = self.layout(p)
        c = p["root_chord"] + (p["tip_chord"] - p["root_chord"]) * max(0.0, (abs(y_mm) - L["yc"]) / (L["b2"] - L["yc"]))
        xle = L["le_sweep_tip"] * max(0.0, (abs(y_mm) - L["yc"]) / (L["b2"] - L["yc"]))
        dz = max(0.0, abs(y_mm) - L["yc"]) * math.tan(math.radians(p["dihedral_deg"]))
        xc = min(max((x_mm - xle) / c, 0.0), 0.97)
        t, m, pc = p["thickness"], p["camber"], p["camber_pos"]
        yt = 5 * t * (0.2969 * math.sqrt(xc) - 0.1260 * xc - 0.3516 * xc ** 2 + 0.2843 * xc ** 3 - 0.1015 * xc ** 4)
        yc = (m / pc ** 2 * (2 * pc * xc - xc ** 2)) if xc < pc else (m / (1 - pc) ** 2 * (1 - 2 * pc + 2 * pc * xc - xc ** 2))
        z = (yc - yt) * c + dz
        a = math.radians(p["wing_incidence_deg"])
        xq = 0.25 * p["root_chord"]
        return float(-(x_mm - xq) * math.sin(a) + z * math.cos(a))

    def fitting_stations(self, p) -> dict:
        """The boom fitting's numbers at the boom station [mm]: the two tubes' centres, the socket from ``boom_x0`` to
        the wing's trailing edge."""
        L = self.layout(p)
        y = p["boom_y"]
        f = (y - L["yc"]) / (L["b2"] - L["yc"])
        c_loc = p["root_chord"] + (p["tip_chord"] - p["root_chord"]) * f
        x_te = L["le_sweep_tip"] * f + c_loc
        xs, zs = self.tube_point(p, y, p["spar_x_frac"])
        xr, zr = self.tube_point(p, y, p["rear_spar_x_frac"])
        return {"x_front": xs, "z_front": zs, "x_rear": xr, "z_rear": zr, "x_te": x_te, "c_loc": c_loc, "socket_length": x_te - p["boom_x0"]}

    def _boom_fitting(self, p, side=1, split_bore=False, bore=True):
        """The printed boom root fitting: two clamp rings around the main spar and the rear tube (inside the wing; the
        foam is slotted for them), a 6 mm web through the wing's lower skin, and the socket tube (14 mm OD) from
        ``boom_x0`` to the wing's trailing edge into which the boom is bonded. The boom's shear and moment reach the
        two tubes as a couple. ``split_bore``: the socket bore in two halves (separate faces for the FEA's bearing
        loads). Clamp screws (M3 through each ring's split) are not drawn."""
        y = side * p["boom_y"]
        s = self.fitting_stations(p)
        w, rw = p["fitting_width"], p["fitting_ring_wall"]
        r_f, r_r = p["spar_od"] / 2 + 0.1, p["rear_spar_od"] / 2 + 0.1
        y0 = y - w / 2

        def ring(x, z, r):
            return cq.Workplane("XZ").workplane(offset=-y0 - w).center(x, z).circle(r + rw).extrude(w)

        zb = p["boom_z"]
        tw = min(p["fitting_web"], w)
        web = (cq.Workplane("XZ").workplane(offset=-y0 - w / 2 - tw / 2)
               .polyline([(s["x_front"], s["z_front"]), (s["x_rear"], s["z_rear"]), (s["x_te"], zb), (s["x_front"], zb)]).close().extrude(tw))
        socket = cq.Workplane("YZ").workplane(offset=p["boom_x0"]).center(y, zb).circle(p["socket_od"] / 2).extrude(s["x_te"] - p["boom_x0"])
        fit = ring(s["x_front"], s["z_front"], r_f).union(ring(s["x_rear"], s["z_rear"], r_r)).union(web).union(socket)
        for x, z, r in ((s["x_front"], s["z_front"], r_f), (s["x_rear"], s["z_rear"], r_r)):
            fit = fit.cut(cq.Workplane("XZ").workplane(offset=-y0 - w - 1).center(x, z).circle(r).extrude(w + 2))
        r_b = p["boom_od"] / 2 + 0.1
        if not bore:                       # the assembled aircraft: the socket solid, so the boom fuses with it
            return fit
        if split_bore:                     # two cuts, not cleaned: the front and rear halves of the bore are separate faces
            xm = 0.5 * (p["boom_x0"] + s["x_te"])
            front = cq.Workplane("YZ").workplane(offset=p["boom_x0"] - 1).center(y, zb).circle(r_b).extrude(xm - p["boom_x0"] + 1)
            rear = cq.Workplane("YZ").workplane(offset=xm).center(y, zb).circle(r_b).extrude(s["x_te"] - xm + 1)
            return fit.cut(front, clean=False).cut(rear, clean=False)
        bore = cq.Workplane("YZ").workplane(offset=p["boom_x0"] - 1).center(y, zb).circle(r_b).extrude(s["x_te"] - p["boom_x0"] + 2)
        return fit.cut(bore)

    def _tail_fitting(self, p, side=1):
        """The printed sleeve at the boom's end that carries the stabiliser and the fin: a 1.5 mm wall sleeve over the
        tail chord (13 mm OD) with a saddle for the stabiliser's spar at 30 % of its chord and a post for the fin."""
        L = self.layout(p)
        y = side * p["boom_y"]
        x0, x1 = L["tail_le"], L["boom_x1"]
        sleeve = cq.Workplane("YZ").workplane(offset=x0).center(y, p["boom_z"]).circle(6.5).extrude(x1 - x0)
        saddle = cq.Workplane("XY").box(30.0, 14.0, 10.0, centered=(True, True, False)).translate((x0 + 0.3 * p["tail_chord"], y, p["boom_z"]))
        post = cq.Workplane("XY").box(40.0, 4.0, 24.0, centered=(True, True, False)).translate((x0 + 0.3 * p["fin_chord"], y, p["boom_z"] - 12.0))
        fit = sleeve.union(saddle).union(post)
        bore = cq.Workplane("YZ").workplane(offset=x0 - 1).center(y, p["boom_z"]).circle(p["boom_od"] / 2 + 0.1).extrude(x1 - x0 + 2)
        return fit.cut(bore)

    def _stab(self, p):
        L = self.layout(p)
        up, lo = naca4(0.0, 0.4, p["tail_thickness"])
        s = p["tail_span"] / 2
        stab = self._section(cq.Workplane("XZ").workplane(offset=-s), p["tail_chord"], L["tail_le"], p["boom_z"], up, lo).extrude(2 * s)
        if p["tail_incidence_deg"]:
            x = L["tail_le"] + 0.25 * p["tail_chord"]
            stab = stab.rotate((x, 0, p["boom_z"]), (x, 1, p["boom_z"]), p["tail_incidence_deg"])
        return stab

    def _fin(self, p, side=1):
        L = self.layout(p)
        up, lo = naca4(0.0, 0.4, p["tail_thickness"])
        z0 = p["boom_z"] - p["fin_ventral"]
        h = p["fin_height"] + p["fin_ventral"]
        return self._section(cq.Workplane("XY").workplane(offset=z0), p["fin_chord"], L["fin_le"], side * p["boom_y"], up, lo).extrude(h)

    def _motor_mount(self, p, tilt=True):
        """The printed cup on the pod's tail: a plate ``mount_thickness`` thick with the motor's 16/19 mm cross bolt
        pattern and an 8 mm centre hole, a 12 mm skirt bonded inside the pod's end ring; tilted by the down-thrust."""
        L = self.layout(p)
        r_out = p["motor_diameter"] / 2 + 2.0
        plate = cq.Workplane("YZ").workplane(offset=L["x_pod_end"]).center(0, p["motor_z"]).circle(r_out - 0.5).extrude(p["mount_thickness"])
        skirt = (cq.Workplane("YZ").workplane(offset=L["x_pod_end"] - 12.0).center(0, p["motor_z"]).circle(r_out).circle(r_out - 2.0).extrude(12.0))
        mount = plate.union(skirt)
        holes = cq.Workplane("YZ").workplane(offset=L["x_pod_end"] - 1).center(0, p["motor_z"]).pushPoints([(8, 0), (-8, 0), (0, 9.5), (0, -9.5)]).circle(1.6).extrude(p["mount_thickness"] + 2)
        centre = cq.Workplane("YZ").workplane(offset=L["x_pod_end"] - 13).center(0, p["motor_z"]).circle(4.0).extrude(20.0)
        mount = mount.cut(holes).cut(centre)
        return self._tilt(mount, p) if tilt else mount

    def _tilt(self, shape, p):
        L = self.layout(p)
        if p["motor_downthrust_deg"]:
            shape = shape.rotate((L["x_pod_end"], 0, p["motor_z"]), (L["x_pod_end"], 1, p["motor_z"]), -p["motor_downthrust_deg"])
        return shape

    def _motor(self, p):
        L = self.layout(p)
        can = cq.Workplane("YZ").workplane(offset=L["x_motor0"]).center(0, p["motor_z"]).circle(p["motor_diameter"] / 2).extrude(p["motor_length"])
        shaft = cq.Workplane("YZ").workplane(offset=L["x_motor1"]).center(0, p["motor_z"]).circle(2.5).extrude(p["prop_gap"] + 4)
        return self._tilt(can.union(shaft), p)

    def _prop_disc(self, p):
        L = self.layout(p)
        disc = cq.Workplane("YZ").workplane(offset=L["prop_x"] - 0.5).center(0, p["motor_z"]).circle(L["prop_R"]).extrude(1.0)
        return self._tilt(disc, p)

    def _skid(self, p):
        """The belly keel: a 6 mm wide TPU plate under the pod from ``skid_x0`` to ``skid_x1``, its top 3 mm into the
        pod's lower contour, its bottom flat at ``skid_depth`` below the pod's lowest line, the front end ramped."""
        if p["skid_depth"] <= 0:
            return None
        prof = self.pod_profile(p)
        xs = np.linspace(p["skid_x0"], p["skid_x1"], 12)
        z_low = np.interp(xs, prof[:, 0], prof[:, 3] - prof[:, 2])
        bottom = -p["pod_height"] - p["skid_depth"]
        top = [(float(x), float(z) + 3.0) for x, z in zip(xs, z_low)]
        ramp = 0.3 * (p["skid_x1"] - p["skid_x0"])
        poly = top + [(p["skid_x1"], bottom), (p["skid_x0"] + ramp, bottom)]
        return cq.Workplane("XZ").polyline(poly).close().extrude(3.0, both=True)

    def _tray(self, p, bay="flight controller tray (Zero; OBS slides it into the computer bay)"):
        """The removable electronics tray: a 2 mm PETG plate sliding on two rails, with 20 x 20 mm M2 holes for the
        flight controller and slots for the receiver and telemetry radio."""
        b = self.bays(p)[bay]
        x0, x1, w, h = b
        tray = cq.Workplane("XY").box(x1 - x0 - 6.0, w - 8.0, 2.0, centered=(False, True, False)).translate((x0 + 3.0, 0, -p["pod_height"] / 2 - 10.0))
        holes = (cq.Workplane("XY").workplane(offset=-p["pod_height"] / 2 - 11.0).center(0.5 * (x0 + x1), 0)
                 .pushPoints([(10, 10), (-10, 10), (10, -10), (-10, -10)]).circle(1.1).extrude(4.0))
        slots = cq.Workplane("XY").workplane(offset=-p["pod_height"] / 2 - 11.0).center(0.5 * (x0 + x1), 0).pushPoints([(0, 30), (0, -30)]).slot2D(30.0, 4.0, 0).extrude(4.0)
        return tray.cut(holes).cut(slots)

    def _battery_tray(self, p):
        """The battery cradle: a 2 mm floor with two 8 mm lips and a strap slot, in the battery bay."""
        x0, x1, w, h = self.bays(p)["battery bay"]
        z = -p["pod_height"] + p["pod_wall"] + 12.0
        floor = cq.Workplane("XY").box(x1 - x0 - 10.0, 46.0, 2.0, centered=(False, True, False)).translate((x0 + 5.0, 0, z))
        lips = [cq.Workplane("XY").box(x1 - x0 - 10.0, 2.0, 10.0, centered=(False, True, False)).translate((x0 + 5.0, s * 22.0, z)) for s in (1, -1)]
        tray = floor.union(lips[0]).union(lips[1])
        slot = cq.Workplane("XY").box(16.0, 60.0, 4.0, centered=(True, True, False)).translate((0.5 * (x0 + x1), 0, z - 1.0))
        return tray.cut(slot)

    def tube_point(self, p, y_mm, x_frac) -> tuple:
        """The centre (x, z) [mm] of a spanwise tube at ``x_frac`` of the local chord at station ``y_mm``, on the
        wing's chord plane (dihedral and incidence included)."""
        L = self.layout(p)
        a = math.radians(p["wing_incidence_deg"])
        xq = 0.25 * p["root_chord"]
        f = max(0.0, (abs(y_mm) - L["yc"]) / (L["b2"] - L["yc"]))
        c = p["root_chord"] + (p["tip_chord"] - p["root_chord"]) * f
        x = L["le_sweep_tip"] * f + x_frac * c
        zc = max(0.0, abs(y_mm) - L["yc"]) * math.tan(math.radians(p["dihedral_deg"]))
        # the camber line's height at that chord fraction keeps the tube centred in the section
        m_, pc = p["camber"], p["camber_pos"]
        ycam = (m_ / pc ** 2 * (2 * pc * x_frac - x_frac ** 2)) if x_frac < pc else (m_ / (1 - pc) ** 2 * (1 - 2 * pc + 2 * pc * x_frac - x_frac ** 2))
        z = zc + ycam * c
        return xq + (x - xq) * math.cos(a) + z * math.sin(a), -(x - xq) * math.sin(a) + z * math.cos(a)

    def _tube_line(self, p, x_frac, od, id_, half_length, n_seg):
        """A carbon tube along the ``x_frac`` chord line from the centre to ``half_length`` each side, in segments
        (the centre piece to the pod's side, then ``n_seg - 1`` outboard): not cleaned, so every segment keeps its
        faces (the FEA loads them one by one)."""
        L = self.layout(p)
        parts = []
        stations = [0.0, L["yc"]] + list(np.linspace(L["yc"], half_length, n_seg)[1:])
        for side in (1, -1):
            for y0, y1 in zip(stations[:-1], stations[1:]):
                pts = []
                for y in (side * y0, side * y1):
                    xx, zz = self.tube_point(p, y, x_frac)
                    pts.append(cq.Vector(xx, y, zz))
                d = pts[1] - pts[0]
                seg = cq.Solid.makeCylinder(od / 2, d.Length, pts[0], d)
                if id_ > 0:
                    seg = seg.cut(cq.Solid.makeCylinder(id_ / 2, d.Length + 0.2, pts[0] - d.normalized() * 0.1, d))
                parts.append(seg)
        return cq.Workplane("XY").add(parts[0].fuse(*parts[1:]))

    def _half_tube(self, p, x_frac, od, id_, half_length, n_seg):
        """The right half of a spanwise tube for the FEA: straight along its outboard line (the dihedral) from y = 0
        to ``half_length``, in the same segments as ``_tube_line`` (the centre piece to the pod's side, then
        ``n_seg - 1`` outboard), fused without cleaning."""
        L = self.layout(p)
        stations = [0.0, L["yc"]] + list(np.linspace(L["yc"], half_length, n_seg)[1:])
        xa, za = self.tube_point(p, L["yc"], x_frac)
        xb, zb = self.tube_point(p, half_length, x_frac)
        a, b = np.array([xa, L["yc"], za]), np.array([xb, half_length, zb])
        u = (b - a) / np.linalg.norm(b - a)
        parts = []
        for y0, y1 in zip(stations[:-1], stations[1:]):
            p0 = a + u * (y0 - L["yc"]) / u[1]
            p1 = a + u * (y1 - L["yc"]) / u[1]
            d = cq.Vector(*(p1 - p0))
            seg = cq.Solid.makeCylinder(od / 2, d.Length, cq.Vector(*p0), d)
            if id_ > 0:
                seg = seg.cut(cq.Solid.makeCylinder(id_ / 2, d.Length + 0.2, cq.Vector(*(p0 - u * 0.1)), d))
            parts.append(seg)
        return cq.Workplane("XY").add(parts[0].fuse(*parts[1:]))

    def _spar(self, p, n_seg=8):
        """The main carbon spar along the ``spar_x_frac`` chord line (straight for a straight taper) with the dihedral,
        in ``n_seg`` segments per side outside the centre piece (separate faces for the FEA)."""
        return self._tube_line(p, p["spar_x_frac"], p["spar_od"], p["spar_id"], p["spar_half_length"], n_seg)

    def _rear_spar(self, p):
        """The rear carry-through tube at ``rear_spar_x_frac``: through the pod's saddle to just outboard of the boom
        fittings — with the main spar it takes the booms' moment as a couple (the boom fittings clamp both)."""
        return self._tube_line(p, p["rear_spar_x_frac"], p["rear_spar_od"], p["rear_spar_id"], p["rear_spar_half_length"], 3)

    def build(self, p):
        if p["tip_chord"] >= p["root_chord"]:
            raise ValueError("tip_chord must be smaller than root_chord")
        L = self.layout(p)
        if L["prop_clearance_boom"] < 10.0:
            raise ValueError(f"the propeller comes within {L['prop_clearance_boom']:.1f} mm of a boom: widen boom_y or shrink the propeller")
        if L["prop_ground_margin_resting"] < 10.0:
            raise ValueError(f"resting on its keel and tail bumpers the propeller is {L['prop_ground_margin_resting']:.1f} mm from the ground (< 10)")
        if L["prop_clearance_wing_te"] < 20.0:
            raise ValueError("the propeller plane must be behind the wing trailing edge")
        if p["rear_spar_half_length"] < p["boom_y"] + p["fitting_width"] / 2:
            raise ValueError("the rear tube must reach through the boom fittings")
        part = p["part"]
        if part == "wing":
            return self._wing_solid(p)
        if part == "spar":
            return self._spar(p)
        if part == "rear_spar":
            return self._rear_spar(p)
        if part == "pod":
            return self._pod_shell(p)
        if part == "nose":
            return self._pod_shell(p, x_to=L["x_nose_joint"])
        if part == "boom":
            return self._boom(p)
        if part == "boom_fitting":
            return self._boom_fitting(p)
        if part == "boom_fitting_fea":
            return self._boom_fitting(p, split_bore=True)
        if part == "boom_fea":
            return self._boom_pieces(p)
        if part == "motor_mount_fea":
            return self._motor_mount(p, tilt=False)
        if part == "spar_fea":
            return self._half_tube(p, p["spar_x_frac"], p["spar_od"], p["spar_id"], p["spar_half_length"], 8)
        if part == "rear_spar_fea":
            return self._half_tube(p, p["rear_spar_x_frac"], p["rear_spar_od"], p["rear_spar_id"], p["rear_spar_half_length"], 3)
        if part == "tail":
            return self._stab(p).union(self._fin(p, 1)).union(self._fin(p, -1))
        if part == "tail_fitting":
            return self._tail_fitting(p)
        if part == "motor_mount":
            return self._motor_mount(p)
        if part == "tray":
            return self._tray(p)
        if part == "battery_tray":
            return self._battery_tray(p)
        if part == "skid":
            return self._skid(p)
        aircraft = self._pod_solid(p).union(self._wing_solid(p))
        for side in (1, -1):
            aircraft = aircraft.union(self._boom_fitting(p, side, bore=False)).union(self._boom(p, side)).union(self._fin(p, side))
        aircraft = aircraft.union(self._stab(p)).union(self._motor_mount(p)).union(self._motor(p))
        skid = self._skid(p)
        if skid is not None:
            aircraft = aircraft.union(skid)
        if p["show_prop"]:
            aircraft = aircraft.union(self._prop_disc(p))
        if p["angle_of_attack_deg"]:
            aircraft = aircraft.rotate((0, 0, 0), (0, 1, 0), p["angle_of_attack_deg"])    # right-hand rule about +Y: nose up
        return aircraft


# --------------------------------------------------------------------------------------------------- helpers
def overrides(p=None) -> dict:
    """A parameter set without the per-call keys (``part``, ``show_prop``, ``angle_of_attack_deg``), to pass to
    ``Nisus().generate(**overrides(p), part=...)``."""
    return {k: v for k, v in dict(p or {}).items() if k not in ("part", "show_prop", "angle_of_attack_deg")}


def resolve(p=None, **kw) -> dict:
    d = dict(p or {})
    d.update(kw)
    return Nisus().resolve(**d)


def exploded_parts(p=None, spread=1.0) -> dict:
    """The aircraft's parts as separate shapes moved apart for an exploded view: {name: cq shape}. ``spread`` scales
    the offsets (0 = assembled)."""
    p = resolve(p)
    d = Nisus()
    L = Nisus.layout(p)
    s = spread
    parts = {
        "wing": d._wing_solid(p).translate((0, 0, 120 * s)),
        "spar": d._spar(p).translate((0, 0, 190 * s)),
        "rear spar": d._rear_spar(p).translate((0, 0, 170 * s)),
        "pod": d._pod_shell(p, x_from=L["x_nose_joint"]),
        "nose cone": d._pod_shell(p, x_to=L["x_nose_joint"]).translate((-90 * s, 0, 0)),
        "boom L": d._boom(p, -1).translate((0, -90 * s, -40 * s)),
        "boom R": d._boom(p, 1).translate((0, 90 * s, -40 * s)),
        "boom fitting L": d._boom_fitting(p, -1).translate((0, -40 * s, 40 * s)),
        "boom fitting R": d._boom_fitting(p, 1).translate((0, 40 * s, 40 * s)),
        "stabiliser": d._stab(p).translate((120 * s, 0, 0)),
        "fin L": d._fin(p, -1).translate((120 * s, -70 * s, 60 * s)),
        "fin R": d._fin(p, 1).translate((120 * s, 70 * s, 60 * s)),
        "tail fitting L": d._tail_fitting(p, -1).translate((60 * s, -90 * s, -40 * s)),
        "tail fitting R": d._tail_fitting(p, 1).translate((60 * s, 90 * s, -40 * s)),
        "motor mount": d._motor_mount(p).translate((60 * s, 0, -70 * s)),
        "motor": d._motor(p).translate((100 * s, 0, -70 * s)),
        "propeller disc": d._prop_disc(p).translate((150 * s, 0, -70 * s)),
        "electronics tray": d._tray(p).translate((0, 0, -120 * s)),
        "battery tray": d._battery_tray(p).translate((0, 0, -150 * s)),
    }
    skid = d._skid(p)
    if skid is not None:
        parts["belly skid"] = skid.translate((0, 0, -80 * s))
    return parts


def planform(p=None) -> dict:
    """The lifting surfaces as flat quads [m] in the aircraft frame (x aft, y right, z up), each (LE inboard, LE
    outboard, TE outboard, TE inboard): ``wing`` (centre halves, two panels with the dihedral), ``tail`` (two halves
    at the boom height). The fins are given as ``fins`` (vertical quads) for the lateral estimates. ``S_ref``,
    ``c_ref`` (MAC), ``b_ref``, the incidences."""
    p = resolve(p)
    L = Nisus.layout(p)
    c0, c1, yc, b2 = p["root_chord"], p["tip_chord"], L["yc"], L["b2"]
    dx, dz = L["le_sweep_tip"], (b2 - yc) * math.tan(math.radians(p["dihedral_deg"]))
    centre = np.array([[0, 0, 0], [0, -yc, 0], [c0, -yc, 0], [c0, 0, 0]], float)
    panel = np.array([[0, -yc, 0], [dx, -b2, dz], [dx + c1, -b2, dz], [c0, -yc, 0]], float)
    mirror = np.array([1, -1, 1])
    wing = [centre, panel, centre * mirror, panel * mirror]
    ts, tc, zt = p["tail_span"] / 2, p["tail_chord"], p["boom_z"]
    xt = L["tail_le"]
    tail_l = np.array([[xt, 0, zt], [xt, -ts, zt], [xt + tc, -ts, zt], [xt + tc, 0, zt]], float)
    fin_l = np.array([[L["fin_le"], -p["boom_y"], zt - p["fin_ventral"]], [L["fin_le"], -p["boom_y"], zt + p["fin_height"]],
                      [L["boom_x1"], -p["boom_y"], zt + p["fin_height"]], [L["boom_x1"], -p["boom_y"], zt - p["fin_ventral"]]], float)
    return {"wing": [q / 1000 for q in wing], "tail": [tail_l / 1000, tail_l * mirror / 1000],
            "fins": [fin_l / 1000, fin_l * mirror / 1000], "S_ref": L["S_ref"], "c_ref": L["mac"] / 1000, "b_ref": p["span"] / 1000,
            "wing_incidence_deg": p["wing_incidence_deg"], "tail_incidence_deg": p["tail_incidence_deg"],
            "x_ac_wing": L["x_ac"] / 1000, "l_t": L["l_t"] / 1000, "l_v": L["l_v"] / 1000, "S_h": L["S_h"], "S_v": L["S_v"],
            "V_h": L["V_h"], "V_v": L["V_v"], "z_tail": zt / 1000}


def _frusta_ellipse(prof):
    """Lateral area of a loft of ellipses: Ramanujan's perimeter at each station x the slant length between them."""
    x, a, b = prof[:, 0], prof[:, 1], prof[:, 2]
    h = ((a - b) / (a + b)) ** 2
    per = math.pi * (a + b) * (1 + 3 * h / (10 + np.sqrt(4 - 3 * h)))
    return float(np.sum(0.5 * (per[1:] + per[:-1]) * np.hypot(np.diff(x), np.diff(0.5 * (a + b)))))


def wetted_areas(p=None) -> dict:
    """Wetted areas [m²] and reference lengths [m] per component from the same numbers the CAD builds: the exposed
    wing (both skins, 1.02 x planform for the curvature, outside the pod), the pod, the two booms outside the wing
    and the stabiliser, the stabiliser (both faces) and the two fins."""
    p = resolve(p)
    L = Nisus.layout(p)
    S = L["S_ref"]
    S_exposed = S - p["pod_width"] * p["root_chord"] * 1e-6
    prof = Nisus.pod_profile(p)
    pod = _frusta_ellipse(prof) * 1e-6
    boom_free = (p["boom_length"] - (p["root_chord"] - p["boom_x0"]) - p["tail_chord"]) * 1e-3
    booms = 2 * math.pi * p["boom_od"] * 1e-3 * boom_free
    tail = 2 * L["S_h"] * 1.02
    fins = 2 * 2 * L["S_v_each"] * 1.02
    return {"wing": 2 * 1.02 * S_exposed, "pod": pod, "booms": booms, "tail": tail, "fins": fins,
            "wing_mac": L["mac"] / 1000, "pod_length": p["pod_length"] / 1000, "pod_diameter": math.sqrt(p["pod_width"] * p["pod_height"]) / 1000,
            "boom_length": boom_free, "boom_diameter": p["boom_od"] / 1000, "tail_chord": p["tail_chord"] / 1000,
            "fin_chord": p["fin_chord"] / 1000, "planform": S, "aspect_ratio": L["AR"], "span": p["span"] / 1000,
            "thickness": p["thickness"], "tail_t_c": p["tail_thickness"]}


EXTRA_CD_AREA_M2 = 0.0012      # m²: camera lens, antennas, hatch lines, control horns, pushrods, the stopped motor can's
                               # face — a flat allowance (Hoerner-level guess; an assumption to replace by CFD/flight test)


def drag_buildup(p=None, speed: float = 16.0, nu: float = 1.5e-5) -> dict:
    """Parasite drag area ``Cd0 S`` [m²] by component (Raymer's build-up, as ``merlin.drag_buildup``): turbulent
    flat-plate friction on each component's length, its form factor (wing and tail ``1 + 2 t/c + 60 (t/c)^4``, pod
    and booms ``1 + 60/f³ + f/400`` with the fineness f = L/d), its wetted area, plus 10 % interference (wing–pod,
    boom fittings, tail–boom junctions) and ``EXTRA_CD_AREA_M2``. ``cd0`` is referred to the planform ``S_ref``.
    Low Reynolds numbers (the booms at Re ~ 1e4, the wing ~ 2e5) make the flat-plate law optimistic by tens of
    percent on the small parts; the CFD and the flight test are the check."""
    p = resolve(p)
    a = wetted_areas(p)

    def cf(length):
        re = max(speed * length / nu, 1e4)
        return 0.455 / math.log10(re) ** 2.58

    t, tt = a["thickness"], a["tail_t_c"]
    f_pod = a["pod_length"] / a["pod_diameter"]
    f_boom = a["boom_length"] / a["boom_diameter"]
    parts = {"wing": cf(a["wing_mac"]) * (1 + 2 * t + 60 * t ** 4) * a["wing"],
             "pod": cf(a["pod_length"]) * (1 + 60 / f_pod ** 3 + f_pod / 400) * a["pod"],
             "booms": cf(a["boom_length"]) * (1 + 60 / f_boom ** 3 + f_boom / 400) * a["booms"],
             "tail": cf(a["tail_chord"]) * (1 + 2 * tt + 60 * tt ** 4) * a["tail"],
             "fins": cf(a["fin_chord"]) * (1 + 2 * tt + 60 * tt ** 4) * a["fins"]}
    parts["interference (10 %)"] = 0.10 * sum(parts.values())
    parts["extras (lens, antennas, horns)"] = EXTRA_CD_AREA_M2
    total = sum(parts.values())
    return {"parts_m2": parts, "cd_area_m2": total, "cd0": total / a["planform"], "planform": a["planform"],
            "aspect_ratio": a["aspect_ratio"], "speed": speed, "re_wing": speed * a["wing_mac"] / nu, "wetted": a}


def outline(p=None) -> dict:
    """Polygons [m] for the three-view drawing and the movie: ``top`` (x, y), ``side`` (x, z), ``front`` (y, z);
    ``hinges`` (control-surface hinge lines per view), ``prop`` (plane and disc), ``bays`` (the component bays as
    boxes in top and side view), and the key stations."""
    p = resolve(p)
    L = Nisus.layout(p)
    pl = planform(p)
    prof = Nisus.pod_profile(p)
    x, a, b, zc = prof.T
    pod_top = np.vstack([np.c_[x, a], np.c_[x[::-1], -a[::-1]]]) / 1000
    pod_side = np.vstack([np.c_[x, zc + b], np.c_[x[::-1], (zc - b)[::-1]]]) / 1000
    c0 = p["root_chord"]
    up, lo = naca4(p["camber"], p["camber_pos"], p["thickness"])
    ai = math.radians(p["wing_incidence_deg"])
    xq = 0.25 * c0
    sec = []
    for xx, yy in up + lo[::-1]:
        X, Z = xx * c0, yy * c0
        sec.append((xq + (X - xq) * math.cos(ai) + Z * math.sin(ai), -(X - xq) * math.sin(ai) + Z * math.cos(ai)))
    wing_sec = np.array(sec) / 1000
    boom_top = [np.array([[p["boom_x0"], s * (p["boom_y"] - p["boom_od"] / 2)], [L["boom_x1"], s * (p["boom_y"] - p["boom_od"] / 2)],
                          [L["boom_x1"], s * (p["boom_y"] + p["boom_od"] / 2)], [p["boom_x0"], s * (p["boom_y"] + p["boom_od"] / 2)]]) / 1000 for s in (1, -1)]
    boom_side = np.array([[p["boom_x0"], p["boom_z"] - p["boom_od"] / 2], [L["boom_x1"], p["boom_z"] - p["boom_od"] / 2],
                          [L["boom_x1"], p["boom_z"] + p["boom_od"] / 2], [p["boom_x0"], p["boom_z"] + p["boom_od"] / 2]]) / 1000
    fin_side = np.array([[L["fin_le"], p["boom_z"] - p["fin_ventral"]], [L["boom_x1"], p["boom_z"] - p["fin_ventral"]],
                         [L["boom_x1"], p["boom_z"] + p["fin_height"]], [L["fin_le"], p["boom_z"] + p["fin_height"]]]) / 1000
    tail_side = np.array([[L["tail_le"], p["boom_z"] - 0.5 * p["tail_thickness"] * p["tail_chord"]], [L["tail_te"], p["boom_z"]],
                          [L["tail_le"], p["boom_z"] + 0.5 * p["tail_thickness"] * p["tail_chord"]]]) / 1000
    motor_side = np.array([[L["x_motor0"], p["motor_z"] - p["motor_diameter"] / 2], [L["x_motor1"], p["motor_z"] - p["motor_diameter"] / 2],
                           [L["x_motor1"], p["motor_z"] + p["motor_diameter"] / 2], [L["x_motor0"], p["motor_z"] + p["motor_diameter"] / 2]]) / 1000
    skid_side = np.array([[p["skid_x0"], -p["pod_height"]], [p["skid_x1"], -p["pod_height"]], [p["skid_x1"], L["skid_lowest_z"]],
                          [p["skid_x0"], L["skid_lowest_z"]]]) / 1000
    # front view (y, z): the pod's largest ellipse, the wing (its thickness band with dihedral), booms, fins, stab, prop
    th = np.linspace(0, 2 * math.pi, 64)
    i_max = int(np.argmax(a))
    pod_front = np.c_[a[i_max] * np.cos(th), zc[i_max] + b[i_max] * np.sin(th)] / 1000
    yc, b2 = L["yc"], L["b2"]
    dz = (b2 - yc) * math.tan(math.radians(p["dihedral_deg"]))
    tmax = p["thickness"] * c0
    wing_front = np.array([[-b2, dz], [-yc, 0], [yc, 0], [b2, dz], [b2, dz + tmax * p["tip_chord"] / c0], [yc, tmax], [-yc, tmax], [-b2, dz + tmax * p["tip_chord"] / c0]]) / 1000
    booms_front = [np.c_[s * p["boom_y"] + p["boom_od"] / 2 * np.cos(th), p["boom_z"] + p["boom_od"] / 2 * np.sin(th)] / 1000 for s in (1, -1)]
    fins_front = [np.array([[s * p["boom_y"] - 5, p["boom_z"] - p["fin_ventral"]], [s * p["boom_y"] + 5, p["boom_z"] - p["fin_ventral"]],
                            [s * p["boom_y"] + 5, p["boom_z"] + p["fin_height"]], [s * p["boom_y"] - 5, p["boom_z"] + p["fin_height"]]]) / 1000 for s in (1, -1)]
    stab_front = np.array([[-p["tail_span"] / 2, p["boom_z"] - 4], [p["tail_span"] / 2, p["boom_z"] - 4], [p["tail_span"] / 2, p["boom_z"] + 4], [-p["tail_span"] / 2, p["boom_z"] + 4]]) / 1000
    prop_front = np.c_[L["prop_R"] * np.cos(th), p["motor_z"] + L["prop_R"] * np.sin(th)] / 1000
    skid_front = np.array([[-3, -p["pod_height"]], [3, -p["pod_height"]], [3, L["skid_lowest_z"]], [-3, L["skid_lowest_z"]]]) / 1000
    # hinge lines
    ya0 = L["y_aileron0"]
    hinges_top = []
    for s in (1, -1):
        for y in (ya0, b2):
            pass
        f0 = (ya0 - yc) / (b2 - yc)
        c_a0 = c0 + (p["tip_chord"] - c0) * f0
        x_h0 = L["le_sweep_tip"] * f0 + (1 - p["aileron_chord_frac"]) * c_a0
        x_h1 = L["le_sweep_tip"] + (1 - p["aileron_chord_frac"]) * p["tip_chord"]
        hinges_top.append(np.array([[x_h0, s * ya0], [x_h1, s * b2]]) / 1000)
    hinges_top.append(np.array([[L["tail_le"] + (1 - p["elevator_frac"]) * p["tail_chord"], -p["tail_span"] / 2],
                                [L["tail_le"] + (1 - p["elevator_frac"]) * p["tail_chord"], p["tail_span"] / 2]]) / 1000)
    hinge_side = np.array([[L["fin_le"] + (1 - p["rudder_frac"]) * p["fin_chord"], p["boom_z"] - p["fin_ventral"]],
                           [L["fin_le"] + (1 - p["rudder_frac"]) * p["fin_chord"], p["boom_z"] + p["fin_height"]]]) / 1000
    bays = {k: (v[0] / 1000, v[1] / 1000, v[2] / 1000, v[3] / 1000) for k, v in Nisus.bays(p).items()}
    return {"top": {"pod": pod_top, "wing": [q[:, :2] for q in pl["wing"]], "tail": [q[:, :2] for q in pl["tail"]],
                    "booms": boom_top, "fins": [np.array([[L["fin_le"], s * p["boom_y"] - 4], [L["boom_x1"], s * p["boom_y"] - 4],
                                                          [L["boom_x1"], s * p["boom_y"] + 4], [L["fin_le"], s * p["boom_y"] + 4]]) / 1000 for s in (1, -1)],
                    "motor": np.array([[L["x_motor0"], -p["motor_diameter"] / 2], [L["x_motor1"], -p["motor_diameter"] / 2],
                                       [L["x_motor1"], p["motor_diameter"] / 2], [L["x_motor0"], p["motor_diameter"] / 2]]) / 1000,
                    "prop": np.array([[L["prop_x"], -L["prop_R"]], [L["prop_x"], L["prop_R"]]]) / 1000, "hinges": hinges_top},
            "side": {"pod": pod_side, "wing_section": wing_sec, "boom": boom_side, "fin": fin_side, "tail": tail_side,
                     "motor": motor_side, "skid": skid_side, "prop": np.array([[L["prop_x"], p["motor_z"] - L["prop_R"]], [L["prop_x"], p["motor_z"] + L["prop_R"]]]) / 1000,
                     "hinges": [hinge_side]},
            "front": {"pod": pod_front, "wing": wing_front, "booms": booms_front, "fins": fins_front, "stab": stab_front, "prop": prop_front, "skid": skid_front},
            "bays": bays, "layout": L, "pod_axis_z": L["pod_axis_z"] / 1000, "prop_x": L["prop_x"] / 1000}


# --------------------------------------------------------------------------------------------------- the boom by hand
CARBON_TUBE = {"E_GPa": 120.0, "G_GPa": 5.0, "flexural_strength_MPa": 500.0, "density_g_cm3": 1.55,
               "source": "pultruded unidirectional carbon tube, typical catalogue values (E 120-150 GPa along the fibres, "
                         "G ~5 GPa, flexural strength 500-800 MPa); the shop's figures replace these (assumption)"}


def boom_check(p=None, *, tail_load_N: float = 12.0, fin_side_load_N: float = 5.0, tail_mass_kg: float = 0.03,
               tolerance_mm: float = 1.5, tube=CARBON_TUBE) -> dict:
    """The boom tube by hand: section properties, bending stress and tip deflection under the tail load (one
    boom's share of the stabiliser's maximum lift) as a cantilever from the socket's rear, torsion and bending from
    the fin's side load, the first bending frequency with the tail mass at the tip, and the propeller clearance
    after the boom's lateral deflection at the propeller plane and the build tolerance."""
    p = resolve(p)
    L = Nisus.layout(p)
    do, di = p["boom_od"], p["boom_id"]
    I = math.pi / 64 * (do ** 4 - di ** 4)                      # mm^4
    J = 2 * I
    A = math.pi / 4 * (do ** 2 - di ** 2)
    E = tube["E_GPa"] * 1e3                                       # MPa
    G = tube["G_GPa"] * 1e3
    x_root = p["root_chord"]                                       # the socket's rear: the tube is clamped to here
    Lc = L["x_ac_tail"] - x_root                                   # cantilever to the tail's aerodynamic centre [mm]
    M = tail_load_N * Lc
    sigma = M * (do / 2) / I
    defl_tip = tail_load_N * Lc ** 3 / (3 * E * I)
    # the fin's side load: a lateral tip load (bending in y) and a torque about the boom (arm: half the fin height)
    arm = 0.5 * (p["fin_height"] - p["fin_ventral"])
    T = fin_side_load_N * arm
    tau = T * (do / 2) / J
    twist = T * Lc / (G * J)
    defl_lat_tip = fin_side_load_N * Lc ** 3 / (3 * E * I)
    xp = L["prop_x"] - x_root
    defl_lat_prop = fin_side_load_N * xp ** 2 * (3 * Lc - xp) / (6 * E * I) if xp > 0 else 0.0
    m_boom = tube["density_g_cm3"] * 1e-3 * A * Lc * 1e-3          # kg over the cantilever (A mm², L mm → cm³ x 1e-3... )
    m_boom = tube["density_g_cm3"] * A * Lc * 1e-3 / 1000           # g/cm³ x mm² x mm = 1e-3 g → kg
    k = 3 * E * I / Lc ** 3 * 1e3                                   # N/m  (MPa·mm^4/mm^3 = N/mm → x1e3)
    f1 = math.sqrt(k / (tail_mass_kg / 2 + 0.24 * m_boom)) / (2 * math.pi)
    clearance = L["prop_clearance_boom"] - defl_lat_prop - tolerance_mm
    return {"I_mm4": I, "J_mm4": J, "A_mm2": A, "cantilever_mm": Lc, "tail_load_N": tail_load_N, "bending_moment_Nmm": M,
            "bending_stress_MPa": sigma, "safety_factor_bending": tube["flexural_strength_MPa"] / sigma,
            "tip_deflection_mm": defl_tip, "fin_side_load_N": fin_side_load_N, "torque_Nmm": T, "shear_stress_MPa": tau,
            "twist_deg": math.degrees(twist), "lateral_tip_deflection_mm": defl_lat_tip,
            "lateral_deflection_at_prop_mm": defl_lat_prop, "boom_mass_cantilever_kg": m_boom,
            "first_bending_hz": f1, "prop_clearance_nominal_mm": L["prop_clearance_boom"],
            "prop_clearance_after_deflection_and_tolerance_mm": clearance, "tube": dict(tube)}


__all__ = ["Nisus", "VARIANTS", "VARIANT_NAME", "PARTS", "FEA_PARTS", "overrides", "resolve", "exploded_parts", "planform", "wetted_areas",
           "drag_buildup", "outline", "boom_check", "CARBON_TUBE", "EXTRA_CD_AREA_M2"]
