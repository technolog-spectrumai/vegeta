"""PEREGRINE — a falcon-inspired electric farm drone whose wings sweep back in flight (notebook 28).

PEREGRINE patrols fields and drives wild animals and pest bird flocks off the crops with the noise of its propeller (the
sound does the job: no acoustics here). It cruises, stoops like a falcon, flies between trees and houses, follows and
herds flocks in the air, rests on rooftops and starts again by hanging on its propeller or from a hand throw. The
question of the notebook: **does a wing that sweeps back at the root in flight (the falcon's tuck) pay for its
engineering cost?** So this file builds three wings on one body:

- ``A`` — a large fixed wing (cruise efficiency; the cheapest),
- ``B`` — a small fixed wing (high wing loading: the cheap way to be fast),
- ``C`` — A's wing on two vertical root hinges: each outer panel rotates aft about its hinge by ``fold_deg``
  (0 = spread, ~60 = the stoop tuck, ~85 = stowed along the body on the roof).

Nothing is designed twice: the body is MERLIN's tractor fuselage (notebook 26, ``merlin.Merlin``: profile, motor bell,
spinner, tail cone, the nose fairing for printing), the wing sections, the tail plates and the wing loft are notebook
09's ``FixedWing``. New here: the hinged wing, the hinge lug (the printed part that carries the panel), the planform in
top view (``planform``, the vortex lattice's input in ``peregrine_flight``) and a drag build-up that knows how much of
a folded panel hides over the body.

Frame (as ``FixedWing``): wing root leading edge at the origin, chord along +X (the free stream along +X, the nose at
-X), span along Y, up +Z; the fuselage axis at ``z = -fuselage_diameter/2``; units mm. The hinge axes are vertical
through ``(hinge_x_frac * root_chord, +-yc)`` with ``yc = fuselage_diameter/2 + 5`` (FixedWing's centre-section
half width).

``part``: ``aircraft`` (the whole machine, for pictures, masses and CFD; the propeller is not built), ``wing`` (the
wing alone, for the FEA), ``hinge`` (one hinge lug, printed, for its FEA and the slicer), ``nose`` (the motor
fairing, printed — MERLIN's tractor nose).
"""
import math

import cadquery as cq
import numpy as np
from vegeta.dedalus import Parameter

from fixed_wing import FixedWing, naca4
from merlin import Merlin

VARIANTS = {
    "A": dict(span=1100.0, root_chord=250.0, taper=0.40, wing_hinge="none"),
    "B": dict(span=800.0, root_chord=195.0, taper=0.45, wing_hinge="none"),
    "C": dict(span=1100.0, root_chord=250.0, taper=0.40, wing_hinge="root"),
}
VARIANT_NAME = {"A": "A — large fixed wing", "B": "B — small fixed wing", "C": "C — folding wing"}
FOLD_TUCK, FOLD_STOW = 60.0, 85.0           # the stoop tuck and the roof stow [deg]
HINGE_DRAG_FRACTION = 0.03                   # per hinge: unsealed 1 mm root gap + knuckle fairing, as a share of the
                                             # wing's profile drag (Hoerner-level estimate; an assumption)


def _mp(name, default=None, **kw):
    """MERLIN's parameter ``name`` (which may itself be FixedWing's), with a new default where PEREGRINE differs."""
    p = next(q for q in Merlin.parameters if q.name == name)
    f = dict(name=p.name, default=p.default, units=p.units, min=p.min, max=p.max, description=p.description, choices=p.choices)
    if default is not None:
        f["default"] = default
    f.update(kw)
    return Parameter(**f)


class Peregrine(Merlin):
    """PEREGRINE: MERLIN's tractor body with a falcon wing that can sweep back at the root (``wing_hinge='root'``)."""

    parameters = [
        Parameter("part", "aircraft", choices=("aircraft", "wing", "hinge", "nose"), description="what to build"),
        _mp("propulsion", "tractor", choices=("tractor",), description="one tractor propeller on the nose"),
        _mp("span", VARIANTS["A"]["span"]),
        _mp("root_chord", VARIANTS["A"]["root_chord"]),
        _mp("taper", VARIANTS["A"]["taper"], min=0.2),
        _mp("dihedral_deg", 3.0),
        _mp("camber", 0.02), _mp("camber_pos", 0.4), _mp("thickness", 0.10),
        Parameter("wing_hinge", "none", choices=("none", "root"),
                  description="none: a fixed wing (A, B); root: each outer panel on a vertical hinge at the root (C)"),
        Parameter("fold_deg", 0.0, "deg", min=0.0, max=90.0,
                  description="root sweep of the outer panels about their hinges (aft positive); needs wing_hinge='root'"),
        Parameter("hinge_y_frac", 0.30, "", min=0.0, max=0.8,
                  description="hinge station: 0 at the wing root (the centre section's side), 1 at the tip, as a fraction "
                              "of the half span outside the centre section"),
        Parameter("hinge_x_frac", 0.20, "", min=0.02, max=0.6,
                  description="hinge axis behind the local leading edge, as a fraction of the local chord (on the spar)"),
        _mp("fuselage_diameter", 56.0, description="fuselage (battery and electronics bay) diameter"),
        _mp("nose_length", 150.0),
        _mp("aft_length", 520.0),
        _mp("tail_cone", 260.0),
        _mp("tail_end_diameter", 14.0),
        _mp("fairing_length", 70.0),
        _mp("motor_diameter", 28.0),
        _mp("prop_gap", 3.0),
        _mp("hub_diameter", 16.0),
        _mp("hub_height", 8.0),
        _mp("spinner_length", 16.0),
        _mp("tail_span", 320.0), _mp("tail_chord", 100.0), _mp("fin_height", 110.0), _mp("tail_thickness", 4.0),
        Parameter("hinge_pin_diameter", 3.0, "mm", min=1.0, description="steel hinge pin"),
        Parameter("lug_thickness", 5.0, "mm", min=1.5, description="hinge lug plate thickness (part='hinge')"),
        Parameter("lug_width", 18.0, "mm", min=6.0, description="hinge lug width around the pin (part='hinge')"),
        Parameter("lug_length", 30.0, "mm", min=10.0, description="hinge lug: root block face to the pin centre (part='hinge')"),
        _mp("angle_of_attack_deg"),
    ]

    # ------------------------------------------------------------------------------------------- the wing
    @staticmethod
    def stations(p) -> dict:
        """The wing's numbers [mm]: chords, the centre-section half width ``yc``, the half span, the tip's leading
        edge offset and dihedral rise (FixedWing's straight quarter-chord line), and the hinge: its station ``yh``,
        the local chord ``ch`` there, its leading edge ``(xle_h, zh)`` and the hinge axis ``xh`` on the spar."""
        c0 = p["root_chord"]
        c1 = c0 * p["taper"]
        yc, b2 = p["fuselage_diameter"] / 2 + 5.0, p["span"] / 2
        dx, dz = (c0 - c1) / 4, (b2 - yc) * math.tan(math.radians(p["dihedral_deg"]))
        f = p["hinge_y_frac"]
        yh, ch = yc + f * (b2 - yc), c0 + f * (c1 - c0)
        return {"c0": c0, "c1": c1, "yc": yc, "b2": b2, "dx": dx, "dz": dz, "yh": yh, "ch": ch, "xle_h": f * dx,
                "zh": f * dz, "xh": f * dx + p["hinge_x_frac"] * ch}

    @staticmethod
    def _check(p):
        if p["wing_hinge"] == "none" and p["fold_deg"] != 0.0:
            raise ValueError("fold_deg needs wing_hinge='root': a fixed wing does not fold")

    def _wing(self, p):
        """Fixed: FixedWing's wing. Hinged: the same centre section, a fixed inner panel out to the hinge station (the
        falcon's arm wing; none when ``hinge_y_frac`` is 0) and the outer panel (the hand wing) rotated about its
        vertical hinge by ``fold_deg`` (aft), with a hinge knuckle (the fairing over the pin and its bearings, standing
        2 mm proud of both skins), mirrored to the other side. All lofts follow FixedWing's straight taper, so the
        spread hinged wing is FixedWing's wing."""
        self._check(p)
        if p["wing_hinge"] == "none":
            return FixedWing._wing(self, p)
        s = self.stations(p)
        c0, c1, yc, b2, yh, ch, xh = (s[k] for k in ("c0", "c1", "yc", "b2", "yh", "ch", "xh"))
        if b2 <= yc + 10:
            raise ValueError("span too small for the fuselage diameter")
        section = self._sections(p)
        # XZ workplane: normal -Y, positive offsets toward -Y; build the -Y side and mirror it
        centre = section(cq.Workplane("XZ").workplane(offset=-yc), c0, 0.0, 0.0).extrude(2 * yc)
        side = None
        if yh > yc + 1.0:
            inner = section(cq.Workplane("XZ").workplane(offset=yc), c0, 0.0, 0.0).workplane(offset=yh - yc).moveTo(0, 0)
            side = section(inner, ch, s["xle_h"], s["zh"]).loft(combine=True, ruled=True)
        outer = section(cq.Workplane("XZ").workplane(offset=yh), ch, s["xle_h"], s["zh"]).workplane(offset=b2 - yh).moveTo(0, 0)
        outer = section(outer, c1, s["dx"], s["dz"]).loft(combine=True, ruled=True)
        t_loc, z_cam = _thickness_at(p, p["hinge_x_frac"])
        rk, hk = min(0.45 * t_loc * ch, 0.08 * ch), t_loc * ch + 4.0
        knuckle = cq.Workplane("XY").workplane(offset=s["zh"] + z_cam * ch - hk / 2).center(xh, -yh).circle(rk).extrude(hk)
        outer = outer.union(knuckle)
        if p["fold_deg"]:
            outer = outer.rotate((xh, -yh, 0), (xh, -yh, 1), p["fold_deg"])     # +Z rotation sweeps the -Y panel aft
        side = outer if side is None else side.union(outer)
        return centre.union(side).union(side.mirror("XZ"))

    def _hinge_lug(self, p):
        """One hinge lug: a plate ``lug_thickness`` thick and ``lug_width`` wide reaching ``lug_length`` from its root
        block to the pin centre, a round end around the pin bore, and the root block (20 mm along the span, 12 mm
        deep) that is bonded to the spar. Its own frame: the pin axis along Z through (0, lug_length, 0), the root
        block's bonded face at y = -20."""
        w, t, L, d = p["lug_width"], p["lug_thickness"], p["lug_length"], p["hinge_pin_diameter"]
        plate = cq.Workplane("XY").center(0, L / 2).rect(w, L).extrude(t)
        end = cq.Workplane("XY").center(0, L).circle(w / 2).extrude(t)
        block = cq.Workplane("XY").box(w + 8.0, 20.0, 12.0, centered=(True, False, False)).translate((0, -20.0, 0))
        lug = plate.union(end).union(block)
        bore = cq.Workplane("XY").center(0, L).circle(d / 2 + 0.1).extrude(t)       # 0.1 mm clearance on the radius
        return lug.cut(bore)

    def build(self, p):
        self._check(p)
        if p["part"] == "hinge":
            return self._hinge_lug(p)
        q = self._fw_p(p)
        if p["part"] == "wing":
            return self._wing(q)
        if p["part"] == "nose":
            return self._motor_fairing(p)
        hstab, fin = self._tail(q, separate=True)
        aircraft = self._fuselage_solid(p).union(self._wing(q)).union(hstab).union(fin)
        if p["angle_of_attack_deg"]:
            aircraft = aircraft.rotate((0, 0, 0), (0, 1, 0), p["angle_of_attack_deg"])
        return aircraft


# --------------------------------------------------------------------------------------------------- helpers
def resolve(p=None, **kw) -> dict:
    """The full parameter set: the defaults (variant A), the variant's own values when ``p`` names one
    (``{"variant": "B"}``), then ``p`` and ``kw``."""
    d = dict(p or {})
    d.update(kw)
    v = d.pop("variant", None)
    base = dict(VARIANTS[v]) if v else {}
    base.update(d)
    return Peregrine().resolve(**base)


def _thickness_at(p, x_c):
    """The NACA section's thickness and camber-line height at chord fraction ``x_c`` (fractions of the chord)."""
    t, m, pc = p["thickness"], p["camber"], p["camber_pos"]
    yt = 5 * t * (0.2969 * math.sqrt(x_c) - 0.1260 * x_c - 0.3516 * x_c ** 2 + 0.2843 * x_c ** 3 - 0.1015 * x_c ** 4)
    yc = (m / pc ** 2 * (2 * pc * x_c - x_c ** 2)) if x_c < pc else (m / (1 - pc) ** 2 * (1 - 2 * pc + 2 * pc * x_c - x_c ** 2))
    return 2 * yt, yc


def _rotate_xy(pts, xh, yh, deg):
    a = math.radians(deg)
    out = np.array(pts, float).copy()
    rx, ry = out[:, 0] - xh, out[:, 1] - yh
    out[:, 0] = xh + rx * math.cos(a) - ry * math.sin(a)
    out[:, 1] = yh + rx * math.sin(a) + ry * math.cos(a)
    return out


def planform(p=None, fold_deg=None) -> dict:
    """The lifting surfaces as flat quadrilaterals [m] in the aircraft frame (x aft, y right, z up), each
    (leading edge inboard, leading edge outboard, trailing edge outboard, trailing edge inboard) — the vortex lattice's
    input. ``wing``: the two halves of the centre section, the two inner panels (a hinged wing with ``hinge_y_frac`` >
    0), the two outer panels (folded by ``fold_deg`` about their hinges); ``hidden``: per wing quad, the station
    |y| inside which a folded outer panel lies over the inner wing or the body (0 for the others); ``tail``: the
    horizontal tail's two halves at the fuselage axis height. The fin carries no lift in symmetric flight and is left
    out. The reference values: ``S_ref`` (the spread planform), ``c_ref`` (its mean aerodynamic chord), ``b_ref`` (the
    spread span), ``b`` (the span as folded) and the hinge points."""
    p = resolve(p)
    fold = p["fold_deg"] if fold_deg is None else fold_deg
    if p["wing_hinge"] == "none" and fold:
        raise ValueError("fold_deg needs wing_hinge='root'")
    s = Peregrine.stations(p)
    c0, c1, yc, b2, dx, dz = (s[k] for k in ("c0", "c1", "yc", "b2", "dx", "dz"))
    hinged = p["wing_hinge"] == "root"
    yh, ch, xle, zh, xh = (s[k] for k in ("yh", "ch", "xle_h", "zh", "xh")) if hinged else (yc, c0, 0.0, 0.0, 0.0)
    centre_l = np.array([[0, 0, 0], [0, -yc, 0], [c0, -yc, 0], [c0, 0, 0]], float)
    quads, hidden = [centre_l], [0.0]
    if yh > yc + 1.0:
        quads.append(np.array([[0, -yc, 0], [xle, -yh, zh], [xle + ch, -yh, zh], [c0, -yc, 0]], float)); hidden.append(0.0)
    outer = np.array([[xle, -yh, zh], [dx, -b2, dz], [dx + c1, -b2, dz], [xle + ch, -yh, zh]], float)
    if fold:
        outer = _rotate_xy(outer, xh, -yh, fold)
    quads.append(outer); hidden.append(yh if fold else 0.0)
    mirror = np.array([1, -1, 1])
    wing = quads + [q * mirror for q in quads]
    hidden = hidden + hidden
    L = Merlin.layout(p)
    xt, R = L["tail_le"], p["fuselage_diameter"] / 2
    ts, tc = p["tail_span"] / 2, p["tail_chord"]
    tail_l = np.array([[xt, 0, -R], [xt, -ts, -R], [xt + tc, -ts, -R], [xt + tc, 0, -R]], float)
    lam = p["taper"]
    S_ref = (p["span"] * c0 * (1 + lam) / 2) * 1e-6
    mac = (2 / 3) * c0 * (1 + lam + lam ** 2) / (1 + lam) / 1000
    span_now = 2 * max(np.abs(q[:, 1]).max() for q in wing) / 1000
    return {"wing": [q / 1000 for q in wing], "hidden": [h / 1000 for h in hidden], "tail": [tail_l / 1000, tail_l * mirror / 1000],
            "S_ref": S_ref, "c_ref": mac, "b_ref": p["span"] / 1000, "b": span_now, "fold_deg": fold,
            "hinges": [(xh / 1000, -yh / 1000), (xh / 1000, yh / 1000)] if hinged else [], "yc": yc / 1000, "yh": yh / 1000,
            "tail_area": p["tail_span"] * tc * 1e-6, "fin_area": p["fin_height"] * tc * 1e-6}


def fold_limit(p=None) -> dict:
    """How far the outer panels can fold before they collide: behind its hinge a panel's root swings inboard by
    ``(1 - hinge_x_frac) ch sin(fold)``; it may lie over the inner wing and the body (stowed over the back, as a
    falcon's), but it must not cross the symmetry plane into the other panel. ``max_fold_deg`` (90 when it never does)."""
    p = resolve(p)
    s = Peregrine.stations(p)
    behind = (1 - p["hinge_x_frac"]) * s["ch"]
    ratio = s["yh"] / behind
    return {"hinge_y_mm": s["yh"], "chord_behind_hinge_mm": behind, "max_fold_deg": 90.0 if ratio >= 1 else math.degrees(math.asin(ratio))}


def _quad_area(q):
    x, y = q[:, 0], q[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _clip_y(q, y_lim, keep_below=True):
    """Sutherland–Hodgman: the part of polygon ``q`` (x, y) with y <= y_lim (``keep_below``) or y >= y_lim."""
    out = []
    pts = list(map(tuple, q[:, :2]))
    inside = (lambda pt: pt[1] <= y_lim) if keep_below else (lambda pt: pt[1] >= y_lim)
    for i in range(len(pts)):
        a, b = pts[i - 1], pts[i]
        ia, ib = inside(a), inside(b)
        if ia != ib:
            t = (y_lim - a[1]) / (b[1] - a[1])
            out.append((a[0] + t * (b[0] - a[0]), y_lim))
        if ib:
            out.append(b)
    return np.array(out) if len(out) >= 3 else np.zeros((0, 2))


def exposed_wing_area(p=None, fold_deg=None) -> dict:
    """The wing's planform area outside the body [m^2]: the inner panels, the outer panels' area outboard of the hinge
    station (a folded panel's root part lies over the inner wing or the body: those skins are hidden), plus the centre
    section's 5 mm strips beside the fuselage."""
    p = resolve(p)
    pl = planform(p, fold_deg)
    n = len(pl["wing"]) // 2
    total = 0.0
    for q, h in zip(pl["wing"][1:n], pl["hidden"][1:n]):        # the -Y side, without the centre half
        if h > 0:
            out = _clip_y(q, -h, keep_below=True)
            total += _quad_area(out) if len(out) else 0.0
        else:
            total += _quad_area(q[:, :2])
    strips = 2 * 0.005 * p["root_chord"] / 1000
    panel = _quad_area(pl["wing"][n - 1][:, :2])
    return {"exposed": 2 * total + strips, "outer_panel": panel, "hidden_per_side": (panel + sum(_quad_area(q[:, :2]) for q in pl["wing"][1:n - 1])) - total}


def _frusta(prof):
    x, r = prof[:, 0], prof[:, 1]
    return float(np.sum(np.pi * (r[1:] + r[:-1]) * np.hypot(np.diff(x), np.diff(r))))


def wetted_areas(p=None, fold_deg=None) -> dict:
    """Wetted areas [m^2] and reference lengths [m] per component (as ``merlin.wetted_areas``): the exposed wing (both
    skins, 1.02 x planform for the curvature), the fuselage (its profile revolved), the tail plates (both faces)."""
    p = resolve(p)
    fold = p["fold_deg"] if fold_deg is None else fold_deg
    pl = planform(p, fold)
    ex = exposed_wing_area(p, fold)
    c0, lam = p["root_chord"] / 1000, p["taper"]
    prof = Merlin.fuselage_profile(p)
    return {"wing": 2 * 1.02 * ex["exposed"], "fuselage": _frusta(prof) * 1e-6,
            "tail": 2 * (p["tail_span"] * p["tail_chord"] + p["fin_height"] * p["tail_chord"]) * 1e-6,
            "wing_mac": (2 / 3) * c0 * (1 + lam + lam ** 2) / (1 + lam), "fuselage_length": (prof[-1, 0] - prof[0, 0]) / 1000,
            "tail_chord": p["tail_chord"] / 1000, "planform": pl["S_ref"], "span": pl["b"], "span_ref": pl["b_ref"],
            "aspect_ratio": pl["b"] ** 2 / pl["S_ref"], "fuselage_diameter": p["fuselage_diameter"] / 1000,
            "thickness": p["thickness"], "tail_t_c": p["tail_thickness"] / p["tail_chord"]}


def drag_buildup(p, speed: float, nu: float = 1.5e-5, fold_deg=None) -> dict:
    """Parasite drag area ``Cd0 S`` [m^2] by components — ``merlin.drag_buildup``'s Raymer build-up (turbulent
    flat-plate friction on each component's length, its form factor, its wetted area, +10 % interference) on the
    folded geometry, plus, for a hinged wing, ``HINGE_DRAG_FRACTION`` of the wing's profile drag per hinge (the gap
    and the knuckle). ``cd0`` is referred to the spread planform ``S_ref`` at every fold angle."""
    p = resolve(p)
    a = wetted_areas(p, fold_deg)

    def cf(length):
        re = max(speed * length / nu, 1e4)
        return 0.455 / math.log10(re) ** 2.58

    t = a["thickness"]
    f = a["fuselage_length"] / a["fuselage_diameter"]
    parts = {"wing": cf(a["wing_mac"]) * (1 + 2 * t + 60 * t ** 4) * a["wing"],
             "fuselage": cf(a["fuselage_length"]) * (1 + 60 / f ** 3 + f / 400) * a["fuselage"],
             "tail": cf(a["tail_chord"]) * (1 + 2 * a["tail_t_c"] + 60 * a["tail_t_c"] ** 4) * a["tail"]}
    if p["wing_hinge"] == "root":
        parts["hinges"] = 2 * HINGE_DRAG_FRACTION * parts["wing"]
    total = 1.10 * sum(parts.values())
    return {"parts_m2": parts, "cd_area_m2": total, "cd0": total / a["planform"], "planform": a["planform"],
            "aspect_ratio": a["aspect_ratio"], "span": a["span"], "wetted": a}


def outline(p=None, fold_deg=None) -> dict:
    """Polygons [m] for drawings and the movie: ``side`` (x, z: fuselage, the wing's root section, the fin) and
    ``top`` (x, y: fuselage, the wing's four quads as folded, the tail), the propeller plane ``propulsor_x``."""
    p = resolve(p)
    prof = Merlin.fuselage_profile(p)
    R = p["fuselage_diameter"] / 2
    fus_side = np.vstack([prof, prof[::-1] * [1, -1]]) + [0, -R]
    c = p["root_chord"]
    up, lo = naca4(p["camber"], p["camber_pos"], p["thickness"])
    wing_sec = np.array([(x * c, y * c) for x, y in up + lo[::-1]])
    L = Merlin.layout(p)
    fin = np.array([[L["tail_le"], -R], [L["x_tail"], -R], [L["x_tail"], -R + p["fin_height"]], [L["tail_le"], -R + p["fin_height"]]])
    fus_top = np.vstack([prof, prof[::-1] * [1, -1]])
    pl = planform(p, fold_deg)
    top = [fus_top / 1000] + [q[:, :2] for q in pl["wing"]] + [q[:, :2] for q in pl["tail"]]
    return {"side": [fus_side / 1000, wing_sec / 1000, fin / 1000], "top": top, "propulsor_x": L["prop_x"] / 1000,
            "axis_z": -R / 1000}


__all__ = ["Peregrine", "VARIANTS", "VARIANT_NAME", "FOLD_TUCK", "FOLD_STOW", "HINGE_DRAG_FRACTION", "resolve", "planform",
           "fold_limit", "exposed_wing_area", "wetted_areas", "drag_buildup", "outline"]
