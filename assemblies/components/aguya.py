"""AGUYA — a turbojet-powered fast sampler (notebook 29): MERLIN's job (reach a reported wildfire, fly crosswind
passes through the smoke column with a forward gas sensor, come home) at about 150 m/s instead of about 50.

Promoted from ``notebooks/designs/aguya.py`` (notebook 29) as it was proven there; the notebook copy may move on.

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
FEA), ``nacelle`` (the engine pod with its pylon). ``cfd_surfaces`` splits the aircraft into the three STLs of Aeromant's
``jet_external`` template (the body, the intake face at the engine's front, the nozzle exit face).
"""
import math
from pathlib import Path

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


# ----------------------------------------------------------------------------------------------- helpers
def _resolve(p):
    return Aguya().resolve(**dict(p or {}))


def for_engine(engine, **overrides) -> dict:
    """Aguya's parameters around a ``vegeta.boreas.microjet.Microjet``: the pod 14 mm wider than the casing, 40 mm
    longer than the engine, the intake at the engine's bell-mouth, the nozzle exit from the cycle's ``a8``."""
    d = engine.outer_diameter * 1000
    out = dict(nacelle_diameter=d + 14.0, nacelle_length=engine.length * 1000 + 40.0, intake_diameter=0.75 * d,
               nozzle_diameter=2 * math.sqrt(engine.a8 / math.pi) * 1000)
    out.update(overrides)
    return Aguya().resolve(**out)


def tank_volume_l(p, fill=0.85) -> float:
    """Usable fuel volume [L] of the fuselage tank: the cylinder between the nose ogive's end and the tail cone, less
    the 2 mm wall, less the bays for the sensor (150 mm behind the ogive) and the parachute (120 mm ahead of the
    cone), times ``fill`` (the bladder, the baffles, the unusable fuel)."""
    p = _resolve(p)
    L = Aguya.layout(p)
    x0 = L["x_nose"] + p["nose_ogive"] + 150.0
    x1 = L["x_cone"] - 120.0
    r = p["fuselage_diameter"] / 2 - 2.0
    return max(x1 - x0, 0.0) * math.pi * r * r * 1e-6 * fill


def _frusta(prof):
    x, r = prof[:, 0], prof[:, 1]
    return float(np.sum(np.pi * (r[1:] + r[:-1]) * np.hypot(np.diff(x), np.diff(r))))


def wetted_areas(p) -> dict:
    p = _resolve(p)
    L = Aguya.layout(p)
    b, c0, lam = p["span"] / 1000, p["root_chord"] / 1000, p["taper"]
    S = b * c0 * (1 + lam) / 2
    d = p["fuselage_diameter"] / 1000
    nac = Aguya.nacelle_profile(p)
    return {"wing": 2 * 1.02 * (S - d * c0), "fuselage": _frusta(Aguya.fuselage_profile(p)) * 1e-6,
            "nacelle": _frusta(nac) * 1e-6, "pylon": 2 * p["pylon_chord"] * p["pylon_height"] * 1e-6,
            "vtail": 2 * 2 * p["vtail_span"] * p["vtail_chord"] * 1e-6,
            "wing_mac": (2 / 3) * c0 * (1 + lam + lam ** 2) / (1 + lam), "fuselage_length": p["fuselage_length"] / 1000,
            "nacelle_length": p["nacelle_length"] / 1000, "nacelle_diameter": p["nacelle_diameter"] / 1000,
            "planform": S, "aspect_ratio": b ** 2 / S, "span": b, "fuselage_diameter": d, "thickness": p["thickness"],
            "tail_chord": p["vtail_chord"] / 1000, "tail_t_c": p["tail_thickness"] / p["vtail_chord"],
            "pylon_chord": p["pylon_chord"] / 1000}


def drag_buildup(p, speed: float, nu: float = 1.5e-5, mach: float = 0.0) -> dict:
    """Parasite drag area [m^2] by Raymer's component build-up (as MERLIN's): turbulent flat-plate skin friction on
    each component's length (corrected for Mach, ``(1 + 0.144 M^2)^-0.65``), its form factor (the wing's with Raymer's
    compressibility factor ``1.34 M^0.18``), its wetted area, plus 10 % for interference, gaps and the sensor inlet.
    The pod's outside counts here; its intake spillage and the jet's effect on the afterbody are the CFD's."""
    a = wetted_areas(p)

    def cf(length):
        re = max(speed * length / nu, 1e4)
        return 0.455 / math.log10(re) ** 2.58 / (1 + 0.144 * mach ** 2) ** 0.65

    def ff_wing(tc):
        return (1 + 0.6 / 0.3 * tc + 100 * tc ** 4) * (1.34 * mach ** 0.18 if mach > 0.2 else 1.0)

    def ff_body(f):
        return 1 + 60 / f ** 3 + f / 400

    parts = {"wing": cf(a["wing_mac"]) * ff_wing(a["thickness"]) * a["wing"],
             "fuselage": cf(a["fuselage_length"]) * ff_body(a["fuselage_length"] / a["fuselage_diameter"]) * a["fuselage"],
             "nacelle": cf(a["nacelle_length"]) * (1 + 0.35 / (a["nacelle_length"] / a["nacelle_diameter"])) * a["nacelle"],
             "pylon": cf(a["pylon_chord"]) * ff_wing(0.14) * a["pylon"],
             "vtail": cf(a["tail_chord"]) * ff_wing(a["tail_t_c"]) * a["vtail"]}
    total = 1.10 * sum(parts.values())
    return {"parts_m2": parts, "cd_area_m2": total, "cd0": total / a["planform"], "planform": a["planform"],
            "aspect_ratio": a["aspect_ratio"]}


def cfd_surfaces(p, outdir, tolerance=0.2) -> dict:
    """The three STLs (mm) of Aeromant's ``jet_external``: ``body`` (everything outside the engine), ``intake`` (the
    engine face at the bottom of the intake) and ``exhaust`` (the nozzle exit disc, 5 mm inside the nozzle lip). Returns their paths, the intake and
    nozzle areas [m^2] and the nacelle axis."""
    p = _resolve(dict(p, part="aircraft", angle_of_attack_deg=0.0))
    L = Aguya.layout(p)
    shape = Aguya().build(p).val()
    groups = {"body": [], "intake": [], "exhaust": []}
    zc, tol = L["nacelle_z"], 1e-3 * p["nacelle_diameter"]
    for f in shape.Faces():
        v = np.array([[q.x, q.y, q.z] for q in f.tessellate(1.0)[0]])
        r = np.hypot(v[:, 1], v[:, 2] - zc)
        if f.geomType() == "PLANE" and np.all(np.abs(v[:, 0] - L["x_face"]) < tol) and np.all(r < L["r_intake"] + tol):
            groups["intake"].append(f)
        elif f.geomType() == "PLANE" and np.all(np.abs(v[:, 0] - L["x_nozzle_face"]) < tol) and np.all(r < L["r_nozzle"] + tol):
            groups["exhaust"].append(f)
        else:
            groups["body"].append(f)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, faces in groups.items():
        if not faces:
            raise RuntimeError(f"no {name} faces found on the aircraft")
        path = outdir / f"{name}.stl"
        cq.exporters.export(cq.Compound.makeCompound(faces), str(path), tolerance=tolerance, angularTolerance=0.2)
        paths[name] = path
    return {"paths": paths, "intake_area_m2": math.pi * L["r_intake"] ** 2 * 1e-6,
            "nozzle_area_m2": math.pi * L["r_nozzle"] ** 2 * 1e-6,
            "nacelle_axis_m": [0.0, 0.0, zc / 1000], "faces": {k: len(v) for k, v in groups.items()}}


def outline(p) -> dict:
    """Side view (x, z) polygons in metres: fuselage, nacelle, the wing's root section, one V-tail panel's projection."""
    p = _resolve(p)
    L = Aguya.layout(p)
    prof = Aguya.fuselage_profile(p)
    R = L["R"]
    fus = np.vstack([prof, prof[::-1] * [1, -1]]) + [0, -R]
    nac = Aguya.nacelle_profile(p)
    nac = np.vstack([nac, nac[::-1] * [1, -1]]) + [0, L["nacelle_z"]]
    up, lo = naca4(p["camber"], p["camber_pos"], p["thickness"])
    c = p["root_chord"]
    wing = np.array([(x * c, y * c) for x, y in up + lo[::-1]])
    h = p["vtail_span"] * math.sin(math.radians(p["vtail_dihedral_deg"]))
    tail = np.array([[L["x_tail_le"], -R], [L["x_end"], -R], [L["x_end"], -R + h], [L["x_tail_le"], -R + h]])
    return {"side": [fus / 1000, nac / 1000, wing / 1000, tail / 1000], "nozzle_x": L["x_exit"] / 1000,
            "nozzle_z": L["nacelle_z"] / 1000}


__all__ = ["Aguya", "for_engine", "tank_volume_l", "wetted_areas", "drag_buildup", "cfd_surfaces", "outline"]
