"""ARIGO-1 — a compact, agile, autonomous fixed wing after the northern goshawk and the A6M Zero (notebook 39).

The goshawk's lesson: a short, broad, rounded wing and a long tail win a chase through cover. The Zero's: a light
wing loading buys low stall speeds, tight turns and endurance. ARIGO-1 is a small fixed wing (2-3 kg, 0.8-1.0 m span)
built on both: a broad, tapered, moderately swept wing with rounded tips and washout, a large conventional tail on a
long fuselage, and electric propulsion in one of three layouts. It does not hover; it is hand- or catapult-launched
and lands on its belly. It is designed here independently of GAVILAN (its own wing, tail, fuselage and propulsion
geometry); the nose and section helpers are the lab's (``emerillon.ogive``, ``fixed_wing.naca4``).

The variants of the wing (``VARIANTS``):

- **goshawk** — short, broad, swept 20 degrees: agility first;
- **zero** — longer, nearly unswept, tapered: low-speed handling and endurance first;
- **hybrid** — between them.

The propulsion layouts (``propulsion``):

- ``pusher`` — one motor at the fuselage's end, the (folding) propeller behind the tail; the nose camera sees clear;
- ``tractor`` — one motor in the nose; the camera moves to a chin turret below the propeller's disc, and still looks
  through it (``camera_occlusion``);
- ``twin`` — two tractor nacelles on the wing; the nose is the camera's, the slipstream blows the inner wing.

Frame: x aft from the nose tip, y right, z up; mm. ``part``: ``aircraft`` (everything; the propellers as discs with
``show_props``), ``wing``, ``tail``, ``fuselage``.
"""
from __future__ import annotations

import math

import cadquery as cq
import numpy as np

from vegeta.dedalus import Design, Parameter

from emerillon import ogive
from fixed_wing import naca4

PROPULSION = ("pusher", "tractor", "twin", "tractor_lowcam")     # tractor_lowcam: the camera under the nose, looking through the disc


class Arigo1(Design):
    """ARIGO-1: fuselage, swept tapered wing with rounded tips and washout, conventional tail, one of three propulsions."""

    parameters = [
        Parameter("part", "aircraft", choices=("aircraft", "wing", "tail", "fuselage"), description="what to build"),
        Parameter("propulsion", "pusher", choices=PROPULSION, description="pusher at the tail, tractor in the nose, or twin wing tractors"),
        Parameter("show_props", False, description="the propellers as thin discs (pictures)"),
        # ---- fuselage
        Parameter("fuselage_length", 720.0, "mm", min=300.0, description="nose tip to the fuselage's end"),
        Parameter("fuselage_diameter", 78.0, "mm", min=40.0, description="the battery and avionics bay"),
        Parameter("nose_length", 150.0, "mm", min=40.0, description="ogive nose"),
        Parameter("tail_cone", 330.0, "mm", min=80.0, description="the fuselage's taper to the tail"),
        Parameter("tail_end_diameter", 30.0, "mm", min=10.0, description="the fuselage's end (the pusher's motor mount)"),
        # ---- wing
        Parameter("span", 950.0, "mm", min=500.0),
        Parameter("root_chord", 300.0, "mm", min=80.0),
        Parameter("taper", 0.50, "", min=0.25, max=1.0, description="tip chord (before the rounding) / root chord"),
        Parameter("le_sweep_deg", 14.0, "deg", min=-5.0, max=35.0),
        Parameter("dihedral_deg", 4.0, "deg", min=-5.0, max=12.0),
        Parameter("washout_deg", 2.5, "deg", min=0.0, max=8.0, description="tip incidence below the root's (linear)"),
        Parameter("incidence_deg", 2.0, "deg", min=-4.0, max=8.0, description="root incidence to the fuselage axis"),
        Parameter("tip_round", 0.14, "", min=0.0, max=0.4, description="the outer share of each half-span rounded off (elliptic chord fall)"),
        Parameter("camber", 0.02, "", min=0.0, max=0.06), Parameter("camber_pos", 0.4, "", min=0.2, max=0.6),
        Parameter("thickness", 0.12, "", min=0.06, max=0.18),
        Parameter("wing_x", 210.0, "mm", min=50.0, description="root leading edge from the nose tip"),
        Parameter("wing_z", -12.0, "mm", description="the wing's root chord line below the fuselage axis (low-mid wing)"),
        Parameter("aileron_span", 0.48, "", min=0.2, max=0.8, description="the ailerons' share of each half-span, outboard"),
        Parameter("aileron_chord", 0.25, "", min=0.1, max=0.4, description="aileron chord / local chord"),
        # ---- tail (conventional)
        Parameter("htail_span", 513.0, "mm", min=100.0),
        Parameter("htail_root_chord", 230.0, "mm", min=40.0),
        Parameter("htail_taper", 0.65, "", min=0.3, max=1.0),
        Parameter("htail_sweep_deg", 12.0, "deg", min=0.0, max=35.0),
        Parameter("fin_height", 236.0, "mm", min=40.0),
        Parameter("fin_root_chord", 256.0, "mm", min=40.0),
        Parameter("fin_taper", 0.55, "", min=0.3, max=1.0),
        Parameter("fin_sweep_deg", 30.0, "deg", min=0.0, max=55.0),
        Parameter("tail_thickness", 0.09, "", min=0.04, max=0.14, description="tail sections' thickness / chord (NACA 00xx)"),
        Parameter("elevator_chord", 0.35, "", min=0.15, max=0.6), Parameter("rudder_chord", 0.40, "", min=0.15, max=0.6),
        Parameter("htail_z", 2.0, "mm", min=-20.0, max=120.0, description="the stabiliser's plane above the fuselage axis (raised: out of the wing's wake, a cruciform tail)"),
        Parameter("static_margin", 0.12, "", min=0.02, max=0.4, description="the design CG ahead of the lattice's neutral point, in mean chords (when cg_x is 0)"),
        Parameter("cg_x", 0.0, "mm", min=0.0, description="the design CG from the nose tip; 0: placed by static_margin"),
        Parameter("tail_le_x", 520.0, "mm", min=200.0, description="the stabiliser's root leading edge from the nose tip"),
        # ---- propulsion
        Parameter("prop_diameter", 254.0, "mm", min=100.0, description="pusher / tractor propeller (twin: each)"),
        Parameter("twin_prop_diameter", 203.2, "mm", min=100.0, description="twin: each propeller"),
        Parameter("motor_diameter", 35.0, "mm", min=15.0), Parameter("motor_length", 30.0, "mm", min=10.0),
        Parameter("nacelle_y", 235.0, "mm", min=80.0, description="twin: the nacelles from the centre plane"),
        Parameter("nacelle_length", 120.0, "mm", min=40.0), Parameter("nacelle_diameter", 38.0, "mm", min=15.0),
        # ---- camera and perching
        Parameter("camera_diameter", 40.0, "mm", min=15.0, description="the observation camera's ball (nose, or chin turret on the tractor)"),
        Parameter("chin_drop", 150.0, "mm", min=20.0, description="tractor: the chin turret's centre below the fuselage axis"),
        Parameter("lowcam_drop", 55.0, "mm", min=20.0, description="tractor_lowcam: the camera's centre below the fuselage axis (in a fairing under the nose; no strut)"),
        Parameter("perch_hook", False, description="a retractable dorsal hook for mechanical perching on a bar (optional)"),
        Parameter("skid", True, description="a belly skid for the belly landing"),
    ]

    # ------------------------------------------------------------------------------------------------ geometry
    @staticmethod
    def layout(p) -> dict:
        """Areas, chords, the mean chord and its station, the aerodynamic centres, the tail volumes, the propeller and
        camera stations, the clearances [mm, mm2]."""
        b2 = p["span"] / 2
        cr, ct = p["root_chord"], p["root_chord"] * p["taper"]
        stations = Arigo1.wing_stations(p)
        y = np.array([s[0] for s in stations]); c = np.array([s[2] for s in stations]); xl = np.array([s[1] for s in stations])
        S = 2 * float(np.trapezoid(c, y))
        mac = 2 * float(np.trapezoid(c * c, y)) / S
        y_mac = 2 * float(np.trapezoid(c * y, y)) / S
        x_mac = float(np.interp(y_mac, y, xl))
        x_ac = x_mac + 0.25 * mac
        # tail
        hb2 = p["htail_span"] / 2
        hct = p["htail_root_chord"] * p["htail_taper"]
        S_h = (p["htail_root_chord"] + hct) * hb2
        mac_h = 2 / 3 * (p["htail_root_chord"] ** 2 + p["htail_root_chord"] * hct + hct ** 2) / (p["htail_root_chord"] + hct)
        y_mac_h = hb2 / 3 * (p["htail_root_chord"] + 2 * hct) / (p["htail_root_chord"] + hct)
        x_ac_h = p["tail_le_x"] + y_mac_h * math.tan(math.radians(p["htail_sweep_deg"])) + 0.25 * mac_h
        fct = p["fin_root_chord"] * p["fin_taper"]
        S_v = 0.5 * (p["fin_root_chord"] + fct) * p["fin_height"]
        mac_v = 2 / 3 * (p["fin_root_chord"] ** 2 + p["fin_root_chord"] * fct + fct ** 2) / (p["fin_root_chord"] + fct)
        z_mac_v = p["fin_height"] / 3 * (p["fin_root_chord"] + 2 * fct) / (p["fin_root_chord"] + fct)
        x_fin = p["fuselage_length"] - p["fin_root_chord"] - 15.0
        x_ac_v = x_fin + z_mac_v * math.tan(math.radians(p["fin_sweep_deg"])) + 0.25 * mac_v
        R = p["fuselage_diameter"] / 2
        out = {"S_mm2": S, "AR": p["span"] ** 2 / S, "mac": mac, "y_mac": y_mac, "x_mac_le": x_mac, "x_ac": x_ac, "c_tip": ct,
               "S_h_mm2": S_h, "x_ac_h": x_ac_h, "S_v_mm2": S_v, "x_ac_v": x_ac_v, "z_ac_v": R * 0.6 + z_mac_v, "x_fin": x_fin,
               "l_h": x_ac_h - x_ac, "l_v": x_ac_v - x_ac, "R": R,
               "V_h": S_h * (x_ac_h - x_ac) / (S * mac), "V_v": S_v * (x_ac_v - x_ac) / (S * p["span"]),
               "stations": stations}
        if p["propulsion"] == "pusher":
            out.update({"x_prop": p["fuselage_length"] + 18.0, "props": [(0.0, 0.0)], "prop_d": p["prop_diameter"],
                        "camera": (p["camera_diameter"] / 2, 0.0, 0.0),
                        "prop_below_belly": p["prop_diameter"] / 2 - p["tail_end_diameter"] / 2})
        elif p["propulsion"] in ("tractor", "tractor_lowcam"):
            drop = p["chin_drop"] if p["propulsion"] == "tractor" else p["lowcam_drop"]
            out.update({"x_prop": -18.0, "props": [(0.0, 0.0)], "prop_d": p["prop_diameter"],
                        "camera": (70.0, 0.0, -drop), "prop_below_belly": p["prop_diameter"] / 2 - R})
        else:
            y_n = p["nacelle_y"]
            x_le_n = float(np.interp(y_n, y, xl))
            out.update({"x_prop": x_le_n - p["nacelle_length"] * 0.55, "props": [(y_n, p["wing_z"]), (-y_n, p["wing_z"])],
                        "prop_d": p["twin_prop_diameter"], "camera": (p["camera_diameter"] / 2, 0.0, 0.0),
                        "prop_below_belly": p["twin_prop_diameter"] / 2 - (R + p["wing_z"]),
                        "tip_to_fuselage": y_n - p["twin_prop_diameter"] / 2 - R})
        return out

    @staticmethod
    def wing_stations(p, n_round: int = 6) -> list:
        """The right half-wing's stations (y, x of the leading edge, chord, z, twist): the root, the start of the tip's
        rounding, and ``n_round`` stations on it, the chord falling as a quarter ellipse to 8 % of the root's at the tip;
        the leading edge on the swept line, the twist linear to ``-washout`` at the tip."""
        b2 = p["span"] / 2
        cr, ct = p["root_chord"], p["root_chord"] * p["taper"]
        tan_le = math.tan(math.radians(p["le_sweep_deg"]))
        tan_d = math.tan(math.radians(p["dihedral_deg"]))
        y_r = b2 * (1 - p["tip_round"])

        def straight(y):
            return cr + (ct - cr) * y / b2

        ys = [0.0, y_r] + [y_r + (b2 - y_r) * math.sin(0.5 * math.pi * k / n_round) for k in range(1, n_round + 1)]
        out = []
        for y in ys:
            if y <= y_r + 1e-9:
                c = straight(y)
                x_le = p["wing_x"] + y * tan_le
            else:                                                # the rounding: the chord falls about its local quarter chord
                c_r = straight(y_r)
                u = (y - y_r) / (b2 - y_r)
                c = max(c_r * math.sqrt(max(1 - u * u, 0.0)), 0.08 * cr)
                x_qc = p["wing_x"] + y * tan_le + 0.25 * straight(y)
                x_le = x_qc - 0.25 * c
            out.append((y, x_le, c, p["wing_z"] + y * tan_d, -p["washout_deg"] * y / b2))
        return out

    # ---------------------------------------------------------------------------------------------------- solids
    @staticmethod
    def _section(wp, chord, x0, z0, twist_deg, camber, camber_pos, t):
        up, lo = naca4(camber, camber_pos, t, n=28)
        a = math.radians(twist_deg)
        pts = []
        for x, z in up + lo[::-1][1:]:
            xx, zz = (x - 0.25) * chord, z * chord                         # about the quarter chord
            pts.append((x0 + 0.25 * chord + xx * math.cos(a) + zz * math.sin(a), z0 - xx * math.sin(a) + zz * math.cos(a)))
        return wp.polyline(pts).close()

    def _wing(self, p):
        """The right half lofted through its stations, mirrored. Incidence is the root's; washout twists the tip."""
        st = self.wing_stations(p)
        wp = cq.Workplane("XZ")
        prev_y = 0.0
        for k, (y, x_le, c, z, tw) in enumerate(st):
            if k:
                wp = wp.workplane(offset=-(y - prev_y))
            wp = self._section(wp, c, x_le, z, p["incidence_deg"] + tw, p["camber"], p["camber_pos"], p["thickness"])
            prev_y = y
        half = wp.loft(combine=True, ruled=False).mirror("XZ")              # built toward +y? XZ normal is -y: mirror to +y
        return half.union(half.mirror("XZ"))

    def _fuselage(self, p):
        L, R = p["fuselage_length"], p["fuselage_diameter"] / 2
        r_end = p["tail_end_diameter"] / 2
        nose = ogive(p["nose_length"], R, 18)
        cone = ogive(p["tail_cone"], R - r_end, 18)[::-1]
        cone = np.column_stack([L - cone[:, 0], cone[:, 1] + r_end])
        prof = np.vstack([nose, cone, [[L, r_end]]])
        pts = [(float(x), float(r)) for x, r in prof]
        pts = [pts[0]] + [q for a, q in zip(pts, pts[1:]) if abs(q[0] - a[0]) + abs(q[1] - a[1]) > 1e-6] + [(L, 0.0)]
        body = cq.Workplane("XY").polyline(pts).close().revolve(360.0, (0, 0, 0), (1, 0, 0))
        if p["skid"]:
            skid = cq.Workplane("XY").box(220.0, 14.0, 8.0).edges("|Y").fillet(3.0).translate((p["wing_x"] + 60.0, 0, -R - 2.0))
            body = body.union(skid)
        return body

    def _surface(self, root_x, root_c, taper, span, sweep_deg, t, along="y", z0=0.0):
        """A tapered, swept tail surface: ``along`` 'y' (the stabiliser, both halves) or 'z' (the fin, up)."""
        tip_c = root_c * taper
        dx = span * math.tan(math.radians(sweep_deg))
        up, lo = naca4(0.0, 0.4, t, n=20)
        sec = lambda c, x0: [(x0 + x * c, z * c) for x, z in up + lo[::-1][1:]]
        if along == "y":
            half = (cq.Workplane("XZ").workplane(offset=0).polyline([(x, z + z0) for x, z in sec(root_c, root_x)]).close()
                    .workplane(offset=-span / 2).polyline([(x, z + z0) for x, z in sec(tip_c, root_x + dx / 2 * 0 + (span / 2) * math.tan(math.radians(sweep_deg)))]).close()
                    .loft(combine=True, ruled=True)).mirror("XZ")
            return half.union(half.mirror("XZ"))
        fin = (cq.Workplane("XY").workplane(offset=z0).polyline(sec(root_c, root_x)).close()
               .workplane(offset=span).polyline(sec(tip_c, root_x + dx)).close().loft(combine=True, ruled=True))
        return fin

    def _tail(self, p):
        L = self.layout(p)
        h = self._surface(p["tail_le_x"], p["htail_root_chord"], p["htail_taper"], p["htail_span"], p["htail_sweep_deg"], p["tail_thickness"],
                          "y", z0=p["htail_z"])
        f = self._surface(L["x_fin"], p["fin_root_chord"], p["fin_taper"], p["fin_height"], p["fin_sweep_deg"], p["tail_thickness"], "z",
                          z0=0.6 * L["R"] * 0)
        return h.union(f)

    def _camera(self, p):
        L = self.layout(p)
        x, y, z = L["camera"]
        ball = cq.Workplane("XY").sphere(p["camera_diameter"] / 2).translate((x, y, z))
        if p["propulsion"] == "tractor":                                  # the chin turret's strut
            strut = cq.Workplane("XY").box(30.0, 14.0, p["chin_drop"]).translate((x, 0, z / 2))
            ball = ball.union(strut)
        elif p["propulsion"] == "tractor_lowcam":                         # a short blister under the nose, the ball at its front
            r = p["camera_diameter"] / 2 + 3.0
            blister = (cq.Workplane("YZ").circle(r).extrude(110.0).translate((x, 0, z)).faces(">X").edges().fillet(r * 0.9))
            web = cq.Workplane("XY").box(110.0, 2 * r * 0.8, abs(z)).translate((x + 55.0, 0, z / 2))
            ball = ball.union(blister).union(web)
        return ball

    def _motors(self, p):
        L = self.layout(p)
        out = None
        if p["propulsion"] == "twin":
            for y, z in L["props"]:
                n = (cq.Workplane("YZ").circle(p["nacelle_diameter"] / 2).extrude(p["nacelle_length"])
                     .translate((L["x_prop"] + 6.0, y, z)).edges("<X").fillet(p["nacelle_diameter"] * 0.3))
                out = n if out is None else out.union(n)
            return out
        x0 = L["x_prop"] - 6.0 if p["propulsion"] == "pusher" else L["x_prop"] + 6.0
        can = cq.Workplane("YZ").circle(p["motor_diameter"] / 2).extrude(p["motor_length"] if p["propulsion"] != "pusher" else -p["motor_length"])
        hub = cq.Workplane("YZ").circle(11.0).extrude(12.0 if p["propulsion"] == "pusher" else -12.0)
        return can.translate((x0, 0, 0)).union(hub.translate((L["x_prop"] - 6.0 if p["propulsion"] == "pusher" else L["x_prop"] + 6.0, 0, 0)))

    def _props(self, p):
        L = self.layout(p)
        out = None
        for y, z in L["props"]:
            d = cq.Workplane("YZ").circle(L["prop_d"] / 2).extrude(1.5).translate((L["x_prop"], y, z))
            out = d if out is None else out.union(d)
        return out

    def _hook(self, p):
        L = self.layout(p)
        x = L["x_ac"] - 10.0
        arm = cq.Workplane("XY").box(8.0, 6.0, 70.0).translate((x, 0, L["R"] + 35.0))
        claw = cq.Workplane("XY").box(40.0, 6.0, 8.0).translate((x - 16.0, 0, L["R"] + 70.0))
        return arm.union(claw)

    def parts(self, p) -> dict:
        out = {"fuselage": self._fuselage(p), "wing": self._wing(p), "tail": self._tail(p), "camera": self._camera(p),
               "motors": self._motors(p), "props": self._props(p)}
        if p["perch_hook"]:
            out["hook"] = self._hook(p)
        return out

    def build(self, p):
        if p["part"] == "wing":
            return self._wing(p)
        if p["part"] == "tail":
            return self._tail(p)
        if p["part"] == "fuselage":
            return self._fuselage(p)
        ac = self._fuselage(p).union(self._wing(p)).union(self._tail(p)).union(self._camera(p)).union(self._motors(p))
        if p["perch_hook"]:
            ac = ac.union(self._hook(p))
        if p["show_props"]:
            ac = ac.union(self._props(p))
        return ac


# ------------------------------------------------------------------------------------------------------------ variants
VARIANTS = {
    # goshawk, revised after the first CFD (notebook 39): the measured static margin was 4-7 % MAC, not the 12 % aimed at
    # (the lattice, with the stabiliser in the wing's plane, put the neutral point ~13 mm too far aft). Fixed by a longer
    # tail arm (fuselage 700 -> 760 mm: V_h 0.39 -> 0.51) and the stabiliser raised 30 mm out of the wing's wake
    # (a cruciform tail); the CG design point stays ~330 mm from the nose. Expected margin ~13 % MAC (HAND: tail-volume
    # estimate on the CFD's neutral point, +-4 %) — to be confirmed by the CFD rerun.
    "goshawk": {"span": 860.0, "root_chord": 320.0, "taper": 0.47, "le_sweep_deg": 20.0, "dihedral_deg": 3.0, "washout_deg": 3.0,
                "wing_x": 205.0, "fuselage_length": 760.0, "htail_z": 30.0, "cg_x": 330.955},
    "zero": {"span": 1000.0, "root_chord": 280.0, "taper": 0.55, "le_sweep_deg": 5.0, "dihedral_deg": 5.0, "washout_deg": 2.0,
             "wing_x": 225.0, "fuselage_length": 720.0, "cg_x": 294.166},
    # backpack (the Backpack product): ARIGO-1 hybrid at 0.8 scale, the wing in two plug-in halves, an 8-inch propeller;
    # no CFD yet (lattice + hand until arigo_cfd.sh runs it); CG by the static margin
    "backpack": {"span": 760.0, "root_chord": 240.0, "taper": 0.50, "le_sweep_deg": 14.0, "dihedral_deg": 4.0, "washout_deg": 2.5,
                 "wing_x": 168.0, "fuselage_length": 576.0, "fuselage_diameter": 66.0, "nose_length": 120.0, "tail_cone": 264.0,
                 "tail_end_diameter": 26.0, "htail_span": 410.0, "htail_root_chord": 184.0, "fin_height": 189.0,
                 "fin_root_chord": 205.0, "prop_diameter": 203.2, "camera_diameter": 34.0, "wing_z": -10.0},
    "hybrid": {"span": 950.0, "root_chord": 300.0, "taper": 0.50, "le_sweep_deg": 14.0, "dihedral_deg": 4.0, "washout_deg": 2.5,
               "wing_x": 210.0, "fuselage_length": 720.0, "cg_x": 309.012},
}


TAIL_SCALE = 1.35          # the tail grown proportionally from its first sizing (V_h 0.24-0.30 -> about 0.5): agility wants authority


# CG moves for one layout on one wing (mm, + aft), after its CFD (the pack is placed to give it):
# - ARIGO-0 hybrid (tractor, camera through the disc): the CFD margin in the cruise range was 6 % MAC (its Cm flat from
#   4 to 8 deg); 15 mm forward (6.1 % MAC) brings it to ~12 % — the nose motor makes a forward CG easy to reach.
CG_SHIFT_MM = {("hybrid", "tractor_lowcam"): -15.0}


def variant(name: str = "hybrid", propulsion: str = "pusher", **over) -> dict:
    """A variant; the stabiliser's trailing edge 10 mm ahead of the fuselage's end (clear of the pusher's disc)."""
    q = {**VARIANTS[name], "propulsion": propulsion, **over}
    if "cg_x" not in over and q.get("cg_x") and (name, propulsion) in CG_SHIFT_MM:
        q["cg_x"] = q["cg_x"] + CG_SHIFT_MM[(name, propulsion)]
    q.setdefault("tail_le_x", q["fuselage_length"] - 10.0 - q.get("htail_root_chord", 230.0))
    return q


def resolve(p=None) -> dict:
    return Arigo1().resolve(**dict(p or {}))


N_SPAN_PER_STRIP = 3          # ogma.aero.vlm's n_span for these strips (arigo1_model passes it)


def planform(p=None, n_strip: int = 16) -> dict:
    """The lifting surfaces as flat quadrilaterals [m] for ``ogma.aero.vlm`` and their incidences [deg]: each half-wing
    in ``n_strip`` strips of EQUAL width interpolated through its stations (each strip at its own twist, so washout is
    in the lattice), the stabiliser's two halves. The fin is not in the (symmetric) lattice.

    Equal widths matter: strips of unequal width (the first version, one per station) put trailing legs next to
    neighbours' control points and made the lattice ill-conditioned — lift slopes from 2.1 to 8.5 per rad for small
    geometry changes. With equal strips (16 per half, 3 spanwise panels each) it converges (+-1 %), and the hybrid's
    neutral point comes out at 341 mm against the CFD's 341.5 mm."""
    p = resolve(p)
    L = Arigo1.layout(p)
    st = np.array([s_[:5] for s_ in L["stations"]])
    b2 = p["span"] / 2
    ys = np.linspace(0.0, b2, n_strip + 1)
    xl, ch, zz, tw = (np.interp(ys, st[:, 0], st[:, k]) for k in (1, 2, 3, 4))
    quads, inc, is_wing = [], [], []
    for s in (1, -1):
        for k in range(n_strip):
            q = np.array([[xl[k], s * ys[k], zz[k]], [xl[k + 1], s * ys[k + 1], zz[k + 1]], [xl[k + 1] + ch[k + 1], s * ys[k + 1], zz[k + 1]],
                          [xl[k] + ch[k], s * ys[k], zz[k]]]) / 1000.0
            quads.append(q if s > 0 else q[[1, 0, 3, 2]])
            inc.append(p["incidence_deg"] + 0.5 * (tw[k] + tw[k + 1]))
            is_wing.append(True)
    hb2, hcr = p["htail_span"] / 2, p["htail_root_chord"]
    hct = hcr * p["htail_taper"]
    dx = hb2 * math.tan(math.radians(p["htail_sweep_deg"]))
    for s in (1, -1):
        zt = p["htail_z"]
        q = np.array([[p["tail_le_x"], 0.0, zt], [p["tail_le_x"] + dx, s * hb2, zt], [p["tail_le_x"] + dx + hct, s * hb2, zt],
                      [p["tail_le_x"] + hcr, 0.0, zt]]) / 1000.0
        quads.append(q if s > 0 else q[[1, 0, 3, 2]])
        inc.append(0.0)
        is_wing.append(False)
    return {"quads": quads, "incidence_deg": inc, "is_wing": is_wing, "S_ref": L["S_mm2"] / 1e6, "c_ref": L["mac"] / 1e3, "b": p["span"] / 1e3}


def camera_occlusion(p=None, fov_h_deg: float = 110.0, fov_v_deg: float = 80.0, n: int = 121) -> dict:
    """How much of the observation camera's forward view a propeller disc covers: rays over the field of view (the
    camera looking along -x, level), each tested against every disc ahead of the camera. The share of rays blocked,
    the share of the central 40 x 30 degrees blocked."""
    p = resolve(p)
    L = Arigo1.layout(p)
    cx, cy, cz = L["camera"]
    hs = np.radians(np.linspace(-fov_h_deg / 2, fov_h_deg / 2, n))
    vs = np.radians(np.linspace(-fov_v_deg / 2, fov_v_deg / 2, n))
    H, Vv = np.meshgrid(hs, vs)
    d = np.stack([-np.ones_like(H), np.tan(H), np.tan(Vv)], axis=-1)            # forward is -x
    blocked = np.zeros_like(H, bool)
    for y, z in L["props"]:
        dx = L["x_prop"] - cx
        if dx >= 0:                                                            # the disc is behind the camera: out of a forward view
            continue
        t = dx / d[..., 0]
        py, pz = cy + t * d[..., 1], cz + t * d[..., 2]
        blocked |= (py - y) ** 2 + (pz - z) ** 2 <= (L["prop_d"] / 2) ** 2
    centre = (np.abs(np.degrees(H)) <= 20.0) & (np.abs(np.degrees(Vv)) <= 15.0)
    return {"blocked_share": float(blocked.mean()), "central_blocked_share": float(blocked[centre].mean()), "fov_deg": (fov_h_deg, fov_v_deg),
            "camera_mm": L["camera"]}
