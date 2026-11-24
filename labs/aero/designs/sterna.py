"""STERNA — a 1.5 m electric survey fixed wing for coastal work, after the tern and the A6M Zero (notebook 51).

The tern's lesson: a slender body and long, lightly loaded outer panels fly slowly and cheaply. The Zero's: mass
discipline and big control surfaces. STERNA is a shoulder-wing tractor with a folding propeller, a payload pallet in
the belly just ahead of the CG, plug-in wing panels on one straight joiner tube and a low conventional tail. It is
hand- or bungee-launched and lands on two belly rails.

The configuration was chosen against a low-wing tractor and a pod-and-boom pusher: the pusher sees ahead and keeps
its propeller off the ground, but on a slender pod the boom limits the disc to under about 200 mm, the thrust line
rises and the payload ends up far from the CG. The cost of the tractor is in ``sterna_mass`` and the notebook.

The fuselage is not a body of revolution: its top line, bottom line and half-width are tables over the length
(``TOP``, ``BOT``, ``HALF_WIDTH``), the sections superellipses between them. Frame: x aft from the spinner tip,
y right, z up from the thrust line; mm. ``part``: ``aircraft``, ``wing``, ``tail``, ``fuselage``, ``joiner`` (one
half of the wing joiner tube, from the fuselage wall to its end: the FEA's part).
"""
from __future__ import annotations

import math

import cadquery as cq
import numpy as np

from vegeta.dedalus import Design, Parameter

from fixed_wing import naca4

# ---- the fuselage's lines [mm]: (x, z) of the top and the bottom, (x, half-width)
TOP = [(0, 0), (10, 12), (45, 21), (110, 36), (170, 46), (250, 50), (350, 50), (450, 50), (600, 50), (700, 44), (800, 37),
       (900, 31), (1000, 26), (1085, 22)]
BOT = [(0, 0), (10, -12), (45, -21), (110, -46), (190, -72), (215, -75), (300, -75), (400, -75), (500, -75), (600, -62),
       (700, -40), (800, -22), (900, -8), (1000, 0), (1085, 8)]
HALF_WIDTH = [(0, 0), (8, 10), (45, 21), (130, 40), (200, 50), (300, 50), (420, 50), (520, 50), (600, 44), (750, 28),
              (900, 16), (1050, 9), (1085, 6)]
SPINNER = [(0.0, 0.5), (6.0, 9.0), (16.0, 15.0), (30.0, 19.5), (45.0, 21.0)]          # (x, radius)
FIN = [(28.0, 895.0, 1100.0), (130.0, 940.0, 1100.0), (212.0, 975.0, 1100.0), (226.0, 990.0, 1098.0), (233.0, 1010.0, 1090.0)]  # (z, x_le, x_te)
RAILS = (170.0, 520.0, 28.0, 8.0)             # x from, x to, each rail's y from the centre plane, width = height
PALLET = (190.0, 310.0)                       # the payload pallet in the belly: x from, x to
HATCHES = ((170.0, 375.0), (425.0, 640.0))    # the two top hatches: x from, x to
BATTERY_BAY = (320.0, 495.0)                  # the tray's travel under the joiner
JOINER = {"x": 402.0, "z": 38.0, "od": 14.0, "wall": 1.0, "length": 380.0, "in_panel": 140.0}


def _line(table, x):
    xs, ys = zip(*table)
    return float(np.interp(x, xs, ys))


def _smooth(xs, table, passes=3):
    y = np.array([_line(table, x) for x in xs])
    for _ in range(passes):
        y[1:-1] = (y[:-2] + 2 * y[1:-1] + y[2:]) / 4
    return y


class Sterna(Design):
    """STERNA: superellipse fuselage, shoulder wing (flat centre, cranked outer panels, rounded tips, washout), a low
    conventional tail, belly rails, a spinner; the propeller as a thin disc with ``show_props``."""

    parameters = [
        Parameter("part", "aircraft", choices=("aircraft", "wing", "tail", "fuselage", "joiner"), description="what to build"),
        Parameter("show_props", False, description="the propeller as a thin disc (pictures)"),
        # ---- wing
        Parameter("span", 1500.0, "mm", min=800.0),
        Parameter("root_chord", 240.0, "mm", min=80.0),
        Parameter("tip_chord", 150.0, "mm", min=40.0, description="at the start of the tip's rounding"),
        Parameter("centre_span", 500.0, "mm", min=100.0, description="the constant-chord, flat centre section (both sides)"),
        Parameter("dihedral_deg", 5.0, "deg", min=0.0, max=12.0, description="of the outer panels, from the break"),
        Parameter("washout_deg", 2.0, "deg", min=0.0, max=6.0, description="tip incidence below the root's, linear from the break"),
        Parameter("incidence_deg", 2.0, "deg", min=-2.0, max=6.0, description="root incidence to the thrust line"),
        Parameter("camber", 0.03, "", min=0.0, max=0.06, description="NACA 4-digit stand-in for the SD7062/SD7037 class"),
        Parameter("camber_pos", 0.4, "", min=0.2, max=0.6),
        Parameter("thickness", 0.12, "", min=0.08, max=0.16),
        Parameter("wing_x", 330.0, "mm", min=100.0, description="root leading edge from the spinner tip"),
        Parameter("wing_z", 38.0, "mm", description="the root chord line above the thrust line (shoulder wing)"),
        Parameter("aileron_from", 420.0, "mm"), Parameter("aileron_to", 720.0, "mm"),
        Parameter("flap_from", 60.0, "mm"), Parameter("flap_to", 410.0, "mm"),
        Parameter("control_chord", 0.25, "", min=0.1, max=0.4, description="aileron and flap chord / local chord"),
        # ---- tail
        Parameter("htail_span", 500.0, "mm", min=100.0),
        Parameter("htail_root_chord", 130.0, "mm", min=40.0),
        Parameter("htail_tip_chord", 100.0, "mm", min=30.0),
        Parameter("htail_x", 948.0, "mm", description="the stabiliser's root leading edge from the spinner tip"),
        Parameter("htail_z", 18.0, "mm", description="the stabiliser's plane above the thrust line"),
        Parameter("tail_thickness", 0.09, "", min=0.04, max=0.14),
        Parameter("elevator_chord", 0.35, "", min=0.15, max=0.6), Parameter("rudder_chord", 0.30, "", min=0.15, max=0.6),
        # ---- propulsion and balance
        Parameter("prop_diameter", 280.0, "mm", min=150.0, description="folding, 11 in class"),
        Parameter("cg_x", 401.0, "mm", description="the design CG from the spinner tip: 30 % MAC"),
        # ---- size (the variants: sterna_long)
        Parameter("scale", 1.0, "", min=0.5, max=3.0, description="the fuselage, fin, rails and spinner scaled by this (their tables are STERNA's); the wing and tailplane are set by their own parameters"),
        Parameter("mtow_g", 2450.0, "g", min=500.0, description="the design mass limit"),
    ]

    # ------------------------------------------------------------------------------------------------ geometry
    @staticmethod
    def wing_stations(p) -> list:
        """The right half-wing's stations (y, x of the leading edge, chord, z, twist): the centre line, the break, the
        straight taper about an unswept quarter chord, and the tip's rounding."""
        b2, yb = p["span"] / 2, p["centre_span"] / 2
        cr, ct = p["root_chord"], p["tip_chord"]
        x_qc = p["wing_x"] + 0.25 * cr
        y_r = b2 * 735.0 / 750.0                                   # the rounding's start
        tan_d = math.tan(math.radians(p["dihedral_deg"]))

        def at(y, c):
            z = p["wing_z"] + max(y - yb, 0.0) * tan_d
            tw = -p["washout_deg"] * max(y - yb, 0.0) / (b2 - yb)
            return (y, x_qc - 0.25 * c, c, z, tw)
        chord = lambda y: cr if y <= yb else cr + (ct - cr) * (y - yb) / (y_r - yb)
        out = [at(0.0, cr), at(yb, cr), at(0.5 * (yb + y_r), chord(0.5 * (yb + y_r))), at(700.0 * b2 / 750.0, chord(700.0 * b2 / 750.0)), at(y_r, ct)]
        y, x_le, c, z, tw = at(b2 * 747.0 / 750.0, 0.79 * ct)
        out.append((y, x_le + ct / 15.0, c, z, tw))                # the rounded tip: the chord falls, the leading edge runs aft
        y, x_le, c, z, tw = at(b2, 0.40 * ct)
        out.append((y, x_le + ct / 7.5, c, z, tw))
        return out

    @staticmethod
    def layout(p) -> dict:
        """Areas, the mean chord and its station, aerodynamic centres, tail arms and volumes [mm, mm2]."""
        st = Sterna.wing_stations(p)
        y = np.array([s[0] for s in st]); c = np.array([s[2] for s in st]); xl = np.array([s[1] for s in st])
        S = 2 * float(np.trapezoid(c, y))
        mac = 2 * float(np.trapezoid(c * c, y)) / S
        y_mac = 2 * float(np.trapezoid(c * y, y)) / S
        x_mac = float(np.interp(y_mac, y, xl))
        x_ac = x_mac + 0.25 * mac
        hb2, hcr, hct = p["htail_span"] / 2, p["htail_root_chord"], p["htail_tip_chord"]
        S_h = (hcr + hct) * hb2
        x_ac_h = p["htail_x"] + 0.25 * hcr                         # unswept quarter chord
        k = p["scale"]
        zf = k * np.array([f[0] for f in FIN]); cf = k * np.array([f[2] - f[1] for f in FIN]); xf = k * np.array([f[1] for f in FIN])
        S_v = float(np.trapezoid(cf, zf))
        z_ac_v = float(np.trapezoid(cf * zf, zf)) / S_v
        x_ac_v = float(np.interp(z_ac_v, zf, xf + 0.25 * cf))
        return {"S_mm2": S, "AR": p["span"] ** 2 / S, "mac": mac, "y_mac": y_mac, "x_mac_le": x_mac, "x_ac": x_ac,
                "S_h_mm2": S_h, "x_ac_h": x_ac_h, "S_v_mm2": S_v, "x_ac_v": x_ac_v, "z_ac_v": z_ac_v,
                "l_h": x_ac_h - x_ac, "l_v": x_ac_v - x_ac, "V_h": S_h * (x_ac_h - x_ac) / (S * mac),
                "V_v": S_v * (x_ac_v - x_ac) / (S * p["span"]), "cg_mac": (p["cg_x"] - x_mac) / mac,
                "length": 1100.0 * k, "height": (233.0 + 75.0 + RAILS[3]) * k, "prop_below_rails": p["prop_diameter"] / 2 - (75.0 + RAILS[3]) * k,
                "scale": k, "fin_root_chord": 205.0 * k,
                "stations": st}

    # ---------------------------------------------------------------------------------------------------- solids
    @staticmethod
    def _section(wp, chord, x0, z0, twist_deg, camber, camber_pos, t, n=24):
        up, lo = naca4(camber, camber_pos, t, n=n)
        a = math.radians(twist_deg)
        pts = []
        for x, z in up + lo[::-1][1:]:
            xx, zz = (x - 0.25) * chord, z * chord                         # about the quarter chord
            pts.append((x0 + 0.25 * chord + xx * math.cos(a) + zz * math.sin(a), z0 - xx * math.sin(a) + zz * math.cos(a)))
        return wp.polyline(pts).close()

    def _wing(self, p):
        """The right half as ruled lofts between its stations (the taper is straight, the break sharp), mirrored."""
        st = self.wing_stations(p)
        half = None
        for a, b in zip(st, st[1:]):
            wp = cq.Workplane("XZ").workplane(offset=-a[0])
            wp = self._section(wp, a[2], a[1], a[3], p["incidence_deg"] + a[4], p["camber"], p["camber_pos"], p["thickness"])
            wp = wp.workplane(offset=-(b[0] - a[0]))
            wp = self._section(wp, b[2], b[1], b[3], p["incidence_deg"] + b[4], p["camber"], p["camber_pos"], p["thickness"])
            seg = wp.loft(combine=True, ruled=True)
            half = seg if half is None else half.union(seg)
        return half.union(half.mirror("XZ"))

    def _fuselage(self, p, n_around: int = 28, exponent: float = 2.7):
        sc = p["scale"]
        xs = np.concatenate([np.linspace(45.0, 215.0, 9), np.linspace(250.0, 500.0, 6), np.linspace(550.0, 1085.0, 12)])
        top, bot, hw = sc * _smooth(xs, TOP), sc * _smooth(xs, BOT), sc * _smooth(xs, HALF_WIDTH)
        xs = sc * xs
        e = 2.0 / exponent
        wp = None
        for k, x in enumerate(xs):
            zc, hh = (top[k] + bot[k]) / 2, (top[k] - bot[k]) / 2
            pts = []
            for i in range(n_around):
                a = 2 * math.pi * i / n_around
                ca, sa = math.cos(a), math.sin(a)
                pts.append((hw[k] * math.copysign(abs(ca) ** e, ca), zc + hh * math.copysign(abs(sa) ** e, sa)))
            wp = (cq.Workplane("YZ").workplane(offset=float(x)) if wp is None else wp.workplane(offset=float(x - xs[k - 1]))).polyline(pts).close()
        body = wp.loft(combine=True, ruled=True)
        spinner = cq.Workplane("XY").polyline([(0.0, 0.0)] + [(sc * x, sc * r) for x, r in SPINNER] + [(45.0 * sc, 0.0)]).close().revolve(360.0, (0, 0, 0), (1, 0, 0))
        body = body.union(spinner)
        x0, x1, yr, w = (sc * v for v in RAILS)
        for s in (1, -1):
            body = body.union(cq.Workplane("XY").box(x1 - x0, w, w + 10.0).translate(((x0 + x1) / 2, s * yr, -75.0 * sc - w / 2 + 5.0)))   # up into the curved belly
        return body

    def _tail(self, p):
        up, lo = naca4(0.0, 0.4, p["tail_thickness"], n=16)
        sec = lambda c, x0, k=1.0: [(x0 + x * c, k * z * c) for x, z in up + lo[::-1][1:]]
        hb2, hcr, hct = p["htail_span"] / 2, p["htail_root_chord"], p["htail_tip_chord"]
        x_qc = p["htail_x"] + 0.25 * hcr
        z0 = p["htail_z"]
        half = (cq.Workplane("XZ").polyline([(x, z + z0) for x, z in sec(hcr, p["htail_x"])]).close()
                .workplane(offset=-hb2).polyline([(x, z + z0) for x, z in sec(hct, x_qc - 0.25 * hct)]).close().loft(combine=True, ruled=True))
        tail = half.union(half.mirror("XZ"))
        wp = None
        fin = [tuple(p["scale"] * v for v in f) for f in FIN]
        for k, (z, xl, xt) in enumerate(fin):
            wp = (cq.Workplane("XY").workplane(offset=z) if wp is None else wp.workplane(offset=z - fin[k - 1][0])).polyline(sec(xt - xl, xl, 0.8)).close()
        return tail.union(wp.loft(combine=True, ruled=True))

    def _joiner(self, p):
        """One half of the joiner tube, from the fuselage wall (y = 50) to its end in the panel, along +y."""
        j, k = JOINER, p["scale"]
        return (cq.Workplane("XZ").workplane(offset=-50.0 * k).center(p["wing_x"] + 0.30 * p["root_chord"], p["wing_z"]).circle(j["od"] / 2).circle(j["od"] / 2 - j["wall"])
                .extrude(-j["in_panel"]))

    def _props(self, p):
        return cq.Workplane("YZ").workplane(offset=24.0 * p["scale"]).circle(p["prop_diameter"] / 2).extrude(1.5)

    def parts(self, p) -> dict:
        return {"fuselage": self._fuselage(p), "wing": self._wing(p), "tail": self._tail(p), "props": self._props(p)}

    def build(self, p):
        if p["part"] in ("wing", "tail", "fuselage", "joiner"):
            return getattr(self, "_" + p["part"])(p)
        ac = self._fuselage(p).union(self._wing(p)).union(self._tail(p))
        if p["show_props"]:
            ac = ac.union(self._props(p))
        return ac


def resolve(p=None) -> dict:
    return Sterna().resolve(**dict(p or {}))


N_SPAN_PER_STRIP = 3


def planform(p=None, n_strip: int = 16) -> dict:
    """The lifting surfaces as flat quadrilaterals [m] for ``ogma.aero.vlm`` and their incidences [deg]: each half-wing
    in ``n_strip`` equal strips through its stations (so the crank and the washout are in the lattice), the
    stabiliser's two halves. The fin is not in the (symmetric) lattice."""
    p = resolve(p)
    L = Sterna.layout(p)
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
    hb2, hcr, hct = p["htail_span"] / 2, p["htail_root_chord"], p["htail_tip_chord"]
    x0, x_qc, zt = p["htail_x"], p["htail_x"] + 0.25 * p["htail_root_chord"], p["htail_z"]
    for s in (1, -1):
        q = np.array([[x0, 0.0, zt], [x_qc - 0.25 * hct, s * hb2, zt], [x_qc + 0.75 * hct, s * hb2, zt], [x0 + hcr, 0.0, zt]]) / 1000.0
        quads.append(q if s > 0 else q[[1, 0, 3, 2]])
        inc.append(0.0)
        is_wing.append(False)
    return {"quads": quads, "incidence_deg": inc, "is_wing": is_wing, "S_ref": L["S_mm2"] / 1e6, "c_ref": L["mac"] / 1e3, "b": p["span"] / 1e3}
