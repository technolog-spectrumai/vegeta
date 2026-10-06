"""AGUYA — a turbojet-powered fast sampler (notebook 29): MERLIN's job (reach a reported wildfire, fly crosswind
passes through the smoke column with a forward gas sensor, come home) at about 150 m/s instead of about 50.

What the speed asks of the airframe:

- **The engine** is a model turbojet (``vegeta.boreas.microjet``, notebook 28) in a **dorsal pod** on a pylon behind
  the wing: the intake sees clean air above the fuselage, the hot jet leaves above the tail cone and between the two
  halves of a **V-tail**, and the sensor nose stays far ahead of anything the engine breathes out.
- **The fuel** (kerosene, 0.8 kg/L) is in the fuselage between the wing spars, on the centre of gravity, so the trim
  does not change as it burns: the fuselage diameter is set by the tank volume of the longest mission.
- **The wing** is small and thin (NACA 1410): at 150 m/s the lift coefficient is ~0.02 and the wing's job is to hold
  the launch, the loiter over the fire and the turns; its area sets the stall speed and so the catapult.
- **Recovery** by parachute from a belly hatch (no belly landing with a hot engine and kerosene over dry forest).

Frame (as ``FixedWing``): wing root leading edge at the origin, chord along +X, span along Y, up +Z, the free stream
along +X; the fuselage axis at ``z = -fuselage_diameter/2`` (the wing on top of it); units mm. The nacelle's axis is
``pylon_height + nacelle_diameter/2`` above the fuselage's top.

``part``: ``aircraft`` (the whole machine, solid: CFD, masses, pictures), ``wing`` (wing with its centre section, for the
FEA), ``nacelle`` (the engine pod with its pylon).

Geometry only: the analysis helpers (``for_engine``, ``tank_volume_l``, ``wetted_areas``, ``drag_buildup``,
``cfd_surfaces``, ``outline``) stay in ``notebooks/designs/aguya.py``.
"""
import math

import cadquery as cq
import numpy as np
from vegeta.dedalus import Parameter

from .fixed_wing import FixedWing, naca4


def _fw(name, default=None, **kw):
    p = next(q for q in FixedWing.parameters if q.name == name)
    f = dict(name=p.name, default=p.default, units=p.units, min=p.min, max=p.max, description=p.description, choices=p.choices)
    if default is not None:
        f["default"] = default
    f.update(kw)
    return Parameter(**f)


class Aguya(FixedWing):
    parameters = [
        Parameter("part", "aircraft", choices=("aircraft", "wing", "nacelle"), description="what to build"),
        _fw("span", 1300.0),
        _fw("root_chord", 290.0),
        _fw("taper", 0.5),
        _fw("dihedral_deg", 1.0),
        _fw("camber", 0.01), _fw("camber_pos"), _fw("thickness", 0.10),
        _fw("fuselage_diameter", 120.0, description="fuselage diameter (the fuel tank, the sensor and avionics bays)"),
        Parameter("fuselage_length", 1350.0, "mm", min=300),
        Parameter("nose_length", 520.0, "mm", min=50, description="nose tip to the wing's leading edge"),
        Parameter("nose_ogive", 260.0, "mm", min=20, description="the ogive sensor nose's length"),
        Parameter("tail_cone", 420.0, "mm", min=20),
        Parameter("tail_end_diameter", 30.0, "mm", min=4),
        Parameter("nacelle_diameter", 126.0, "mm", min=20, description="pod outside (engine casing + wall + air gap)"),
        Parameter("nacelle_length", 340.0, "mm", min=50, description="intake lip to nozzle exit"),
        Parameter("nacelle_x", 170.0, "mm", description="intake lip behind the wing's leading edge"),
        Parameter("intake_diameter", 84.0, "mm", min=10, description="intake opening (the engine's bell-mouth)"),
        Parameter("intake_depth", 30.0, "mm", min=2, description="intake face (engine front) behind the lip"),
        Parameter("nozzle_diameter", 44.0, "mm", min=10, description="nozzle exit (from the cycle's A8)"),
        Parameter("boattail", 110.0, "mm", min=10, description="pod taper from full diameter to the nozzle"),
        Parameter("pylon_height", 20.0, "mm", min=0, description="fuselage top to the pod's underside"),
        Parameter("pylon_chord", 210.0, "mm", min=20),
        Parameter("pylon_thickness", 0.14, "", min=0.06, max=0.3, description="pylon section thickness / chord"),
        Parameter("vtail_span", 290.0, "mm", min=50, description="each V-tail panel, root to tip"),
        Parameter("vtail_chord", 190.0, "mm", min=20),
        Parameter("vtail_dihedral_deg", 40.0, "deg", min=20, max=60),
        Parameter("tail_thickness", 6.0, "mm", min=1),
        _fw("angle_of_attack_deg"),
    ]

    # ----------------------------------------------------------------------------------------- stations
    @staticmethod
    def layout(p) -> dict:
        R = p["fuselage_diameter"] / 2
        x_nose, x_end = -p["nose_length"], -p["nose_length"] + p["fuselage_length"]
        Rn = p["nacelle_diameter"] / 2
        z_n = p["pylon_height"] + Rn                                         # nacelle axis, above the wing root (z = 0)
        x_lip, x_exit = p["nacelle_x"], p["nacelle_x"] + p["nacelle_length"]
        out = {"R": R, "axis_z": -R, "x_nose": x_nose, "x_end": x_end, "x_cone": x_end - p["tail_cone"],
               "Rn": Rn, "nacelle_z": z_n, "x_lip": x_lip, "x_face": x_lip + p["intake_depth"], "x_exit": x_exit, "x_nozzle_face": x_exit - 5.0,
               "x_tail_le": x_end - p["vtail_chord"], "r_intake": p["intake_diameter"] / 2, "r_nozzle": p["nozzle_diameter"] / 2}
        if p["intake_diameter"] >= p["nacelle_diameter"] - 6 or p["nozzle_diameter"] >= p["nacelle_diameter"]:
            raise ValueError("intake and nozzle must fit inside the nacelle")
        if x_exit > x_end:
            raise ValueError("the nozzle must end ahead of the fuselage's tail end")
        if p["nose_ogive"] + p["tail_cone"] >= p["fuselage_length"]:
            raise ValueError("nose_ogive + tail_cone exceed the fuselage length")
        return out

    @staticmethod
    def fuselage_profile(p, n=16) -> np.ndarray:
        L = Aguya.layout(p)
        R, s = L["R"], np.linspace(0, 1, n)
        lo = p["nose_ogive"]
        # tangent ogive nose: r = R sqrt(1 - (1 - x/lo)^2) blunted by the sensor inlet radius
        pts = [(L["x_nose"] + lo * t, max(R * math.sqrt(max(1 - (1 - t) ** 2, 0.0)), 0.0)) for t in s]
        re = p["tail_end_diameter"] / 2
        pts += [(L["x_cone"] + p["tail_cone"] * t, R - (R - re) * (3 * t ** 2 - 2 * t ** 3)) for t in s]
        pts += [(L["x_end"], 0.0)]
        return np.array(pts)

    @staticmethod
    def nacelle_profile(p, n=10) -> np.ndarray:
        """Half profile (x, r) of the pod's outside, about its own axis: an elliptic lip from the intake radius to
        full diameter, the cylinder, the boattail down to the nozzle (with a 2 mm wall)."""
        L = Aguya.layout(p)
        Rn, ri, rn = L["Rn"], L["r_intake"], L["r_nozzle"]
        t = np.linspace(0, math.pi / 2, n)
        lip = 0.35 * p["nacelle_diameter"]
        pts = [(L["x_lip"] + lip * (1 - math.cos(a)), ri + (Rn - ri) * math.sin(a)) for a in t]
        x_bt = L["x_exit"] - p["boattail"]
        s = np.linspace(0, 1, n)
        pts += [(x_bt + p["boattail"] * u, Rn - (Rn - rn - 2.0) * (3 * u ** 2 - 2 * u ** 3)) for u in s]
        return np.array(pts)

    # ----------------------------------------------------------------------------------------- parts
    def _fuselage(self, p):
        prof = self.fuselage_profile(p)
        prof = np.vstack([[prof[0, 0], 0.0], prof[1:]]) if prof[0, 1] > 1e-9 else prof
        pts = [(float(x), float(r)) for x, r in prof]
        pts = [pts[0]] + [q for i, q in enumerate(pts[1:], 1) if math.dist(q, pts[i - 1]) > 1e-6]
        body = cq.Workplane("XY").polyline(pts).close().revolve(360, (0, 0, 0), (1, 0, 0))
        return body.translate((0, 0, -p["fuselage_diameter"] / 2))

    def _nacelle(self, p):
        L = self.layout(p)
        prof = self.nacelle_profile(p)
        ri, rn = L["r_intake"], L["r_nozzle"]
        pts = [(L["x_face"], 0.0), (L["x_face"], ri), (L["x_lip"], ri)] + [tuple(q) for q in prof[1:]]
        pts += [(L["x_exit"], rn), (L["x_nozzle_face"], rn), (L["x_nozzle_face"], 0.0)]   # the nozzle face, 5 mm in
        pod = cq.Workplane("XY").polyline(pts).close().revolve(360, (0, 0, 0), (1, 0, 0)).translate((0, 0, L["nacelle_z"]))
        # the pylon: a symmetric section from inside the fuselage up into the pod
        c = p["pylon_chord"]
        up, lo = naca4(0.0, 0.4, p["pylon_thickness"])
        sec = [(x * c, y * c) for x, y in up + lo[::-1][1:]]
        x0 = max(L["x_exit"] - p["boattail"] - c, L["x_face"] + 10.0)    # behind the engine face (the intake stays clear)
        z0, z1 = -0.25 * p["fuselage_diameter"], L["nacelle_z"] - 0.5 * L["Rn"]
        pylon = (cq.Workplane("XY").workplane(offset=z0).polyline([(x0 + x, y) for x, y in sec]).close()
                 .extrude(z1 - z0))
        return pod.union(pylon)

    def _vtail(self, p):
        L = self.layout(p)
        ch, b, t = p["vtail_chord"], p["vtail_span"], p["tail_thickness"]
        plate = cq.Workplane("XY").box(ch, t, b, centered=(False, True, False))      # along +z from the axis
        g = 90.0 - p["vtail_dihedral_deg"]
        left = plate.rotate((0, 0, 0), (1, 0, 0), g)
        right = plate.rotate((0, 0, 0), (1, 0, 0), -g)
        return left.union(right).translate((L["x_tail_le"], 0, -p["fuselage_diameter"] / 2))

    def _fw_p(self, p):
        return dict(p, nose_length=p["nose_length"], fuselage_length=p["fuselage_length"])

    def build(self, p):
        if p["part"] == "wing":
            return self._wing(self._fw_p(p))
        if p["part"] == "nacelle":
            return self._nacelle(p)
        aircraft = (self._fuselage(p).union(self._wing(self._fw_p(p))).union(self._nacelle(p)).union(self._vtail(p)))
        if p["angle_of_attack_deg"]:
            aircraft = aircraft.rotate((0, 0, 0), (0, 1, 0), p["angle_of_attack_deg"])
        return aircraft



__all__ = ["Aguya"]
