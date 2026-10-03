"""MERLIN — a compact electric fixed-wing for rapid atmospheric sampling (notebook 26).

MERLIN dashes from a launch stand to a reported wildfire hotspot, flies crosswind passes through the smoke column with
its forward gas sensor, and comes home for a belly landing. The question of the notebook is which propulsor gets it to
the fire fastest — the nose **electric ducted fan** (the jet leaves as a ring around the fuselage), a **tractor**
propeller on the nose, or a **pusher** propeller behind the tail — so one design file builds all three.

Nothing here is new physics or new geometry: the wing, the tail and the NACA sections are notebook 09's ``FixedWing``
(this class subclasses it and calls its ``_wing`` and ``_tail``), the duct with its bell-mouth, shroud, area-ruled
nozzle and stator vanes is notebook 25's ``ducted_fan.EDFHousing`` (its profiles are revolved here around the
fuselage instead of its own centre body). Only the fuselage profile is MERLIN's own, because its front end depends on
the propulsor.

Frame (as ``FixedWing``): wing root leading edge at the origin, chord along +X, span along Y, up +Z, the free stream
along +X (the nose at -X); the fuselage axis at ``z = -fuselage_diameter/2`` (the wing sits on top of it); units mm.
The propulsor plane (the EDF rotor plane or the propeller disc) is at ``x = -nose_length`` for ``edf`` and ``tractor``
and behind the tail for ``pusher``, whose nose is the sensor nose at ``x = -nose_length``.

``part``: ``aircraft`` (the whole machine, solid: CFD, masses, pictures; the EDF rotor and the propellers are not built
— the CFD carries them as a rotor disk), ``wing`` (the wing alone with its centre section, for the FEA), ``nose`` (the
propulsor's standing part for printing: the EDF duct with its vanes and centre ring, or the motor fairing).

Beside the CAD: ``layout`` (the axial stations and the propulsor's place), ``fuselage_profile``, ``wetted_areas`` and
``drag_buildup`` (a flat-plate parasite drag estimate from the same profiles: the polar when the CFD is not run),
``edf_housing_params`` (the parameters of the EDF housing the nose is made of: pass them to
``ducted_fan.housing_geometry``) and ``outline`` (the aircraft's side view for the movie).
"""
import math

import cadquery as cq
import numpy as np
from vegeta.dedalus import Parameter

import ducted_fan as df
from fixed_wing import FixedWing

PROPULSION = ("edf", "tractor", "pusher")

# the EDF of notebook 25 (90 mm, 12 blades, 7 stators): the rotor's blade parameters under ducted_fan's names
EDF_ROTOR = dict(diameter=90.0, hub_diameter=40.0, hub_height=14.0, pitch=160.0, chord_root=14.0, chord_max=15.0,
                 chord_tip=12.0, thickness=0.08, camber=0.04)
EDF_HOUSING = dict(EDF_ROTOR, tip_clearance=0.8, exit_area_ratio=0.9, stator_vanes=7, axial_gap=2.0, stator_chord=16.0,
                   stator_thickness=0.08, stator_gap=14.0)


def _fw(name, default=None, **kw):
    """FixedWing's parameter ``name``, with a new default (and other fields) where MERLIN differs."""
    p = next(q for q in FixedWing.parameters if q.name == name)
    f = dict(name=p.name, default=p.default, units=p.units, min=p.min, max=p.max, description=p.description, choices=p.choices)
    if default is not None:
        f["default"] = default
    f.update(kw)
    return Parameter(**f)


class Merlin(FixedWing):
    """MERLIN in one of three propulsion layouts (``propulsion``), with ``FixedWing``'s wing and tail."""

    parameters = [
        Parameter("part", "aircraft", choices=("aircraft", "wing", "nose"), description="what to build"),
        Parameter("propulsion", "edf", choices=PROPULSION,
                  description="edf: nose ducted fan, annular jet around the fuselage; tractor: nose propeller; pusher: propeller behind the tail"),
        _fw("span", 1400.0),
        _fw("root_chord", 270.0),
        _fw("taper", 0.6),
        _fw("dihedral_deg", 2.0),
        _fw("camber"), _fw("camber_pos"), _fw("thickness"),
        _fw("fuselage_diameter", 76.0, description="fuselage (battery and sensor bay) diameter"),
        Parameter("nose_length", 230.0, "mm", min=50,
                  description="wing leading edge forward to the propulsor plane (edf, tractor) or to the nose tip (pusher)"),
        Parameter("aft_length", 560.0, "mm", min=100, description="wing leading edge back to the fuselage's tail end"),
        Parameter("tail_cone", 260.0, "mm", min=20, description="the fuselage's tapering rear, ending at tail_end_diameter"),
        Parameter("tail_end_diameter", 30.0, "mm", min=6, description="the fuselage's end (the pusher's motor bell)"),
        Parameter("fairing_length", 110.0, "mm", min=10,
                  description="the front fairing: from the EDF centre body (or the motor bell, or the nose tip) to the full diameter"),
        Parameter("motor_diameter", 30.0, "mm", min=8, description="tractor/pusher: motor bell at the propeller"),
        Parameter("prop_gap", 4.0, "mm", min=0.5, description="tractor/pusher: hub face to the motor bell"),
        Parameter("pusher_gap", 30.0, "mm", min=2, description="pusher: tail trailing edge to the propeller plane"),
        Parameter("hub_diameter", 22.0, "mm", min=4, description="tractor/pusher: propeller hub (built with its spinner)"),
        Parameter("hub_height", 10.0, "mm", min=2),
        Parameter("spinner_length", 22.0, "mm", min=0),
        _fw("tail_span", 420.0), _fw("tail_chord", 120.0), _fw("fin_height", 150.0), _fw("tail_thickness", 5.0),
        _fw("angle_of_attack_deg"),
    ]

    # ------------------------------------------------------------------------------------------- stations
    @staticmethod
    def layout(p) -> dict:
        """Axial stations (mm, aircraft frame) of the fuselage and the propulsor."""
        R = p["fuselage_diameter"] / 2
        x_tail = p["aft_length"]
        out = {"R": R, "axis_z": -R, "x_tail": x_tail, "tail_le": x_tail - p["tail_chord"], "kind": p["propulsion"]}
        if p["propulsion"] == "edf":
            hp = edf_housing_params(p)
            L = df.EDFHousing.layout(hp)
            x0 = -p["nose_length"]                                   # the rotor plane
            out.update(prop_x=x0, edf_x0=x0, lip_x=x0 + L["x_l"], exit_x=x0 + L["x_e"], body_x=x0 + L["x_b0"],
                       front_x=x0 - EDF_ROTOR["hub_height"] / 2 - p["spinner_length"], r_front=hp["hub_diameter"] / 2,
                       nacelle_r=L["R_s"] + hp["wall_thickness"], prop_d=hp["diameter"])
            if out["exit_x"] + p["fairing_length"] > 0:
                raise ValueError("the fairing behind the EDF nozzle must reach the full diameter ahead of the wing: shorten "
                                 "fairing_length or lengthen nose_length")
        elif p["propulsion"] == "tractor":
            x0 = -p["nose_length"]
            out.update(prop_x=x0, front_x=x0 + p["hub_height"] / 2 + p["prop_gap"], r_front=p["motor_diameter"] / 2,
                       spinner_x=x0 - p["hub_height"] / 2 - p["spinner_length"])
        else:
            prop_x = x_tail + p["pusher_gap"]
            out.update(prop_x=prop_x, front_x=-p["nose_length"], r_front=0.0,
                       motor_end_x=prop_x - p["hub_height"] / 2 - p["prop_gap"])
            if out["motor_end_x"] <= x_tail:
                raise ValueError("pusher_gap too small: the motor bell must reach behind the tail")
        return out

    @staticmethod
    def fuselage_profile(p, n=14) -> np.ndarray:
        """Half profile (x, r) of the fuselage in its own axis frame (r from the axis), front to back, closed on the
        axis at both ends. edf: from the spinner tip, the hub, the centre body (radius hub/2) through the nozzle, then the
        fairing out to the full radius; tractor: spinner, hub, motor bell, fairing; pusher: an elliptic sensor nose. All:
        the cylinder, the tail cone and (pusher) the motor bell with the hub and the spinner behind it."""
        L = Merlin.layout(p)
        R, s = L["R"], np.linspace(0, 1, n)
        pts = []
        if p["propulsion"] == "edf":
            rb, h = L["r_front"], EDF_ROTOR["hub_height"]                # the EDF rotor's hub, not the propeller's
            x_hub = L["prop_x"] - h / 2
            sp = p["spinner_length"]
            pts += [(x_hub - sp * math.cos(a), rb * math.sin(a)) for a in s * math.pi / 2]
            x_f0 = L["exit_x"]                                        # the fairing starts at the nozzle exit
            pts += [(x_f0 + p["fairing_length"] * t, rb + (R - rb) * (3 * t ** 2 - 2 * t ** 3)) for t in s]
        elif p["propulsion"] == "tractor":
            rh, h = p["hub_diameter"] / 2, p["hub_height"]
            x_hub = L["prop_x"] - h / 2
            sp = p["spinner_length"]
            pts += [(x_hub - sp * math.cos(a), rh * math.sin(a)) for a in s * math.pi / 2]
            pts += [(L["prop_x"] + h / 2, rh), (L["front_x"], rh), (L["front_x"], L["r_front"])]
            rm = L["r_front"]
            pts += [(L["front_x"] + p["fairing_length"] * math.sin(a), rm + (R - rm) * math.sin(a)) for a in s[1:] * math.pi / 2]
        else:
            x0, fl = L["front_x"], p["fairing_length"]
            pts += [(x0 + fl * (1 - math.cos(a)), R * math.sin(a)) for a in s * math.pi / 2]
        x_c1 = L["x_tail"] - p["tail_cone"]
        re = p["tail_end_diameter"] / 2
        pts += [(x_c1 + p["tail_cone"] * t, R - (R - re) * (3 * t ** 2 - 2 * t ** 3)) for t in s]
        if p["propulsion"] == "pusher":
            rh, h, sp = p["hub_diameter"] / 2, p["hub_height"], p["spinner_length"]
            pts += [(L["motor_end_x"], re), (L["motor_end_x"], rh), (L["prop_x"] + h / 2, rh)]
            pts += [(L["prop_x"] + h / 2 + sp * math.sin(a), rh * math.cos(a)) for a in s[1:] * math.pi / 2]
        else:
            pts += [(L["x_tail"], 0.0)]
        pts = [pts[0]] + [q for i, q in enumerate(pts[1:], 1) if math.dist(q, pts[i - 1]) > 1e-6]
        return np.array(pts)

    # ------------------------------------------------------------------------------------------- parts
    def _fuselage_solid(self, p):
        prof = self.fuselage_profile(p)
        prof[:, 1] = np.where(prof[:, 1] < 1e-6, 0.0, prof[:, 1])
        if prof[0, 1] > 0:
            prof = np.vstack([[prof[0, 0], 0.0], prof])
        if prof[-1, 1] > 0:
            prof = np.vstack([prof, [prof[-1, 0], 0.0]])
        body = cq.Workplane("XY").polyline([tuple(q) for q in prof]).close().revolve(360, (0, 0, 0), (1, 0, 0))
        return body.translate((0, 0, -p["fuselage_diameter"] / 2))

    def _edf_shell(self, p):
        """The duct (``EDFHousing.duct_profile``) and the stator vanes (``EDFHousing.vane_section``), in the
        aircraft frame, cut to reach into the fuselage's centre body as in the housing."""
        hp = edf_housing_params(p)
        Lh = df.EDFHousing.layout(hp)
        duct = (cq.Workplane("XY").polyline([tuple(q) for q in df.EDFHousing.duct_profile(hp)["closed"]]).close()
                .revolve(360, (0, 0, 0), (1, 0, 0)).val())
        r0, r1 = Lh["r_b"] - 1.0, Lh["R_s"] + 0.4 * hp["wall_thickness"]
        sec = df.EDFHousing.vane_section(hp)
        prism = cq.Workplane("XZ").polyline([tuple(q) for q in sec]).close().extrude(-(r1 + 1.0)).val()
        xa, la = sec[:, 0].min() - 1.0, np.ptp(sec[:, 0]) + 2.0
        ring = cq.Solid.makeCylinder(r1, la, cq.Vector(xa, 0, 0), cq.Vector(1, 0, 0)).cut(
            cq.Solid.makeCylinder(r0, la, cq.Vector(xa, 0, 0), cq.Vector(1, 0, 0)))
        vane = prism.intersect(ring)
        n = int(hp["stator_vanes"])
        parts = [duct] + [vane.rotate((0, 0, 0), (1, 0, 0), 360.0 * k / n) for k in range(n)]
        # the centre ring the vanes stand on (the nose part's own, for printing)
        ring_x0, ring_x1 = Lh["s_le"] - 2.0, Lh["s_te"] + 2.0
        parts.append(cq.Solid.makeCylinder(Lh["r_b"], ring_x1 - ring_x0, cq.Vector(ring_x0, 0, 0), cq.Vector(1, 0, 0)).cut(
            cq.Solid.makeCylinder(Lh["r_b"] - 2.0, ring_x1 - ring_x0, cq.Vector(ring_x0, 0, 0), cq.Vector(1, 0, 0))))
        shell = parts[0].fuse(*parts[1:]).clean()
        return cq.Workplane("XY").add(shell).translate((-p["nose_length"], 0, -p["fuselage_diameter"] / 2))

    def _motor_fairing(self, p):
        """tractor/pusher printing part: the fuselage's front fairing (tractor) or tail cone (pusher), hollow."""
        L = self.layout(p)
        R, w = p["fuselage_diameter"] / 2, 1.6
        if p["propulsion"] == "tractor":
            x0, x1 = L["front_x"], L["front_x"] + p["fairing_length"]
        else:
            x0, x1 = L["x_tail"] - p["tail_cone"], L["motor_end_x"]
        prof = self.fuselage_profile(p)
        keep = prof[(prof[:, 0] >= x0 - 1e-9) & (prof[:, 0] <= x1 + 1e-9)]
        inner = keep.copy(); inner[:, 1] = np.maximum(inner[:, 1] - w, 0.5)
        poly = np.vstack([keep, inner[::-1]])
        shell = cq.Workplane("XY").polyline([tuple(q) for q in poly]).close().revolve(360, (0, 0, 0), (1, 0, 0))
        return shell.translate((0, 0, -R))

    def _fw_p(self, p):
        """``p`` with the names FixedWing's ``_wing`` and ``_tail`` read (the fuselage from its front to its end)."""
        L = self.layout(p)
        x_front = float(self.fuselage_profile(p)[0, 0])
        return dict(p, nose_length=-x_front, fuselage_length=L["x_tail"] - x_front)

    def build(self, p):
        q = self._fw_p(p)
        if p["part"] == "wing":
            return self._wing(q)
        if p["part"] == "nose":
            return self._edf_shell(p) if p["propulsion"] == "edf" else self._motor_fairing(p)
        hstab, fin = self._tail(q, separate=True)
        aircraft = self._fuselage_solid(p).union(self._wing(q)).union(hstab).union(fin)
        if p["propulsion"] == "edf":
            aircraft = aircraft.union(self._edf_shell(p))
        if p["angle_of_attack_deg"]:
            aircraft = aircraft.rotate((0, 0, 0), (0, 1, 0), p["angle_of_attack_deg"])
        return aircraft


# --------------------------------------------------------------------------------------------------- helpers
def _resolve(p):
    return Merlin().resolve(**dict(p))


def edf_housing_params(p) -> dict:
    """The ``ducted_fan.EDFHousing`` parameters of MERLIN's nose duct: notebook 25's EDF (``EDF_HOUSING``) with a tail
    cone long enough to stay cylindrical through the nozzle — the fuselage continues there, so the nozzle converges
    around a cylinder and the jet leaves as a ring around the fuselage."""
    return df.EDFHousing().resolve(**dict(EDF_HOUSING, tail_length=1.0e5))


def _frusta(prof):
    x, r = prof[:, 0], prof[:, 1]
    return float(np.sum(np.pi * (r[1:] + r[:-1]) * np.hypot(np.diff(x), np.diff(r))))


def wetted_areas(p) -> dict:
    """Wetted areas [m^2] and reference lengths [m] per component, from the same profiles and planform the CAD
    builds: the exposed wing (both skins, 1.02 x planform for the curvature, outside the fuselage), the fuselage (its
    profile revolved; for the EDF only behind the nozzle exit — the part ahead of it is inside the duct), the tail
    plates (both faces) and the EDF nacelle outside (``housing_geometry``'s external area)."""
    p = _resolve(p)
    L = Merlin.layout(p)
    b, c0, lam = p["span"] / 1000, p["root_chord"] / 1000, p["taper"]
    S = b * c0 * (1 + lam) / 2
    d = p["fuselage_diameter"] / 1000
    S_exposed = S - d * c0
    prof = Merlin.fuselage_profile(p)
    if p["propulsion"] == "edf":
        prof = prof[prof[:, 0] >= L["exit_x"] - 1e-9]
    fus = _frusta(prof) * 1e-6
    tail = 2 * (p["tail_span"] * p["tail_chord"] + p["fin_height"] * p["tail_chord"]) * 1e-6
    out = {"wing": 2 * 1.02 * S_exposed, "fuselage": fus, "tail": tail,
           "wing_mac": (2 / 3) * c0 * (1 + lam + lam ** 2) / (1 + lam), "fuselage_length": (prof[-1, 0] - prof[0, 0]) / 1000,
           "tail_chord": p["tail_chord"] / 1000, "planform": S, "aspect_ratio": b ** 2 / S, "span": b,
           "fuselage_diameter": d, "thickness": p["thickness"], "tail_t_c": p["tail_thickness"] / p["tail_chord"]}
    if p["propulsion"] == "edf":
        g = df.housing_geometry(edf_housing_params(p))
        out["nacelle"] = g["external_wetted_area"] * 1e-6
        out["nacelle_length"] = g["total_length"] / 1000
    return out


def drag_buildup(p, speed: float, nu: float = 1.5e-5) -> dict:
    """Parasite drag area ``Cd0 S`` [m^2] by components (Raymer's component build-up): turbulent flat-plate
    ``Cf = 0.455 / (log10 Re)^2.58`` on each component's length, times its form factor (wing and tail
    ``1 + 2 t/c + 60 (t/c)^4``; fuselage ``1 + 60/f^3 + f/400`` with the fineness f = L/d), times its wetted area, plus
    10 % for interference, gaps and the sensor intakes. The EDF nacelle's outside is ``boreas.ducted.nacelle_drag``'s
    job (the propulsor model carries it), so it is listed but not summed here."""
    a = wetted_areas(p)

    def cf(length):
        re = max(speed * length / nu, 1e4)
        return 0.455 / math.log10(re) ** 2.58

    t = a["thickness"]
    f = a["fuselage_length"] / a["fuselage_diameter"]
    parts = {"wing": cf(a["wing_mac"]) * (1 + 2 * t + 60 * t ** 4) * a["wing"],
             "fuselage": cf(a["fuselage_length"]) * (1 + 60 / f ** 3 + f / 400) * a["fuselage"],
             "tail": cf(a["tail_chord"]) * (1 + 2 * a["tail_t_c"] + 60 * a["tail_t_c"] ** 4) * a["tail"]}
    total = 1.10 * sum(parts.values())
    return {"parts_m2": parts, "cd_area_m2": total, "cd0": total / a["planform"], "planform": a["planform"],
            "aspect_ratio": a["aspect_ratio"]}


def outline(p) -> dict:
    """Side view (x, z) polygons in metres of the fuselage, the wing's root section and the fin, for drawings."""
    p = _resolve(p)
    prof = Merlin.fuselage_profile(p)
    R = p["fuselage_diameter"] / 2
    fus = np.vstack([prof * [1, 1], prof[::-1] * [1, -1]]) + [0, -R]
    c = p["root_chord"]
    from fixed_wing import naca4
    up, lo = naca4(p["camber"], p["camber_pos"], p["thickness"])
    wing = np.array([(x * c, y * c) for x, y in up + lo[::-1]])
    L = Merlin.layout(p)
    fin = np.array([[L["tail_le"], -R], [L["x_tail"], -R], [L["x_tail"], -R + p["fin_height"]], [L["tail_le"], -R + p["fin_height"]]])
    return {"side": [fus / 1000, wing / 1000, fin / 1000], "propulsor_x": L["prop_x"] / 1000, "axis_z": -R / 1000}


__all__ = ["Merlin", "PROPULSION", "EDF_ROTOR", "EDF_HOUSING", "edf_housing_params", "wetted_areas", "drag_buildup", "outline"]
