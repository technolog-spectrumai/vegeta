"""ALBATROS — a maritime turbojet transport drone (notebook 50): it leaves a ship's catapult or a runway ashore, cruises
fast and high to a vessel in trouble a few hundred kilometres away, slows over it and releases a cargo multirotor out
of its belly, and flies home to land ashore.

What the mission asks of the airframe:

- **The bay** is the fuselage. The cargo multirotor (``albatros_bay``: one hexacopter, or a row of quadcopters) lies
  folded in a belly bay under the wing, its middle on the wing's aerodynamic centre, so the trim does not jump when it
  leaves. The fuselage is a rounded box around the bay, not a circle; two doors open downwards.
- **The fuel** (kerosene) is in two tanks, ahead of and behind the bay, burned together: the centre of gravity stays.
- **The wing** is small, swept and slender. A gust changes the lift by ``rho V U a S / 2``, the same for any wing
  loading, so the heavier each square metre is loaded, the less the aircraft is thrown about; sweep lowers the lift
  slope ``a`` again; full-span flaperons take out most of what is left (``albatros_gust``). The catapult's end speed
  sets how heavily it can be loaded.
- **The engines** are model-class turbojets (``vegeta.boreas.microjet``) in two pods on the aft fuselage shoulders,
  intakes above the wing's wake and above the spray, each with a two-axis vectoring nozzle.
- **The layout** (``layout``; ``albatros_flight`` sizes them): ``catapult`` is the baseline, wing-borne from launch
  to landing; ``vectored`` has lift jets in the fuselage and swivelling nozzles; ``tailsitter`` stands on its tail
  fins with its engines in a ring; ``lift_cruise`` (set aside) has four electric lift rotors on two booms. The CAD
  draws what the parameters say (``engines`` pods; booms and rotors when ``lift_rotors`` > 0).
- **Water**: a sealed hull with the heavy things low floats upright if it is ditched; the high wing, the pods and the
  V-tail stay dry in a calm sea.

Frame (as ``FixedWing`` and AGUYA): wing root leading edge at the origin, chord along +X, the free stream along +X (the
nose at -X), span along Y, up +Z; the fuselage's top line at z = 0 (a high wing); units mm.

``part``: ``aircraft`` (the whole machine, solid, bay closed: CFD, masses, pictures), ``aircraft_open`` (the bay cut
out, doors away), ``wing``, ``nacelle`` (one engine pod), ``multirotor`` (the bay's vehicle, unfolded, with its cargo
box). Beside the CAD: ``layout``, ``wetted_areas``, ``drag_buildup`` (Raymer's component build-up, as AGUYA's),
``for_design`` (the parameters from a sizing), ``outline`` (side and top views).
"""
import math

import cadquery as cq
import numpy as np

from ogma.aero import cf_turbulent
from vegeta.dedalus import Design, Parameter

from fixed_wing import naca4

SUPER_N = 3.2                                 # the fuselage section's superellipse exponent (2 = ellipse, large = box)
SECTION_AREA = 0.895                          # area of that superellipse / (width x height)
SECTION_PERIMETER = 0.855                     # perimeter of that superellipse / (2 (width + height))
ROTOR_DRAG = 0.0025                           # Cd.A of a stopped two-blade lift rotor / diameter²: blades locked fore-and-aft
                                              # (friction on 0.14 D² of blade), the hub above a motor recessed in the boom


class Albatros(Design):
    parameters = [
        Parameter("part", "aircraft", choices=("aircraft", "aircraft_open", "wing", "nacelle", "multirotor"), description="what to build"),
        Parameter("layout", "catapult", choices=("catapult", "vectored", "tailsitter", "lift_cruise"), description="how it leaves the ground"),
        Parameter("span", 2300.0, "mm", min=400),
        Parameter("root_chord", 460.0, "mm", min=80),
        Parameter("taper", 0.45, "", min=0.2, max=1.0, description="tip chord / root chord"),
        Parameter("sweep_deg", 20.0, "deg", min=0, max=45, description="quarter-chord sweep"),
        Parameter("dihedral_deg", -2.0, "deg", min=-8, max=8, description="negative: anhedral (a high swept wing)"),
        Parameter("camber", 0.02, "", min=0, max=0.06), Parameter("camber_pos", 0.4, "", min=0.1, max=0.9),
        Parameter("thickness", 0.12, "", min=0.06, max=0.18, description="wing section thickness / chord"),
        Parameter("fuselage_length", 2900.0, "mm", min=600),
        Parameter("fuselage_width", 660.0, "mm", min=120), Parameter("fuselage_height", 420.0, "mm", min=120),
        Parameter("nose_length", 1250.0, "mm", min=100, description="nose tip to the wing's root leading edge"),
        Parameter("nose_shape", 620.0, "mm", min=60, description="length of the rounded nose"),
        Parameter("tail_cone", 950.0, "mm", min=60, description="length of the tapering tail (its top line stays level)"),
        Parameter("tail_end", 0.22, "", min=0.08, max=0.6, description="section scale at the tail end"),
        Parameter("bay_length", 880.0, "mm", min=100), Parameter("bay_width", 600.0, "mm", min=60),
        Parameter("bay_height", 280.0, "mm", min=40),
        Parameter("bay_x", 0.0, "mm", description="bay centre behind the wing root's leading edge (for_design: on the centre of gravity)"),
        Parameter("engines", 2, "", min=1, max=6, description="engine pods on the aft fuselage"),
        Parameter("nacelle_diameter", 144.0, "mm", min=30), Parameter("nacelle_length", 400.0, "mm", min=80),
        Parameter("intake_diameter", 98.0, "mm", min=10), Parameter("nozzle_diameter", 52.0, "mm", min=10),
        Parameter("boattail", 120.0, "mm", min=10), Parameter("nacelle_x", 620.0, "mm", description="intake lip behind the wing root's leading edge"),
        Parameter("lift_rotors", 0, "", min=0, max=4, description="electric lift rotors (0, or 4: two on each boom; the set-aside layout)"),
        Parameter("rotor_diameter", 900.0, "mm", min=100),
        Parameter("boom_y", 700.0, "mm", min=100, description="lift boom from the symmetry plane"),
        Parameter("boom_diameter", 80.0, "mm", min=20),
        Parameter("rotor_x_front", -700.0, "mm", description="front rotor axis"), Parameter("rotor_x_rear", 1150.0, "mm", description="rear rotor axis"),
        Parameter("vtail_span", 620.0, "mm", min=50, description="each V-tail panel, root to tip"),
        Parameter("vtail_chord", 330.0, "mm", min=20), Parameter("vtail_dihedral_deg", 42.0, "deg", min=20, max=60),
        Parameter("tail_thickness", 0.09, "", min=0.03, max=0.2, description="tail panel thickness / chord"),
        Parameter("multirotor_rotors", 6, "", min=3, max=8), Parameter("multirotor_arm", 590.0, "mm", min=50),
        Parameter("multirotor_prop", 560.0, "mm", min=50), Parameter("cargo_height", 120.0, "mm", min=0),
        Parameter("angle_of_attack_deg", 0.0, "deg", min=-10, max=20),
    ]

    # ----------------------------------------------------------------------------------------- stations
    @staticmethod
    def layout(p) -> dict:
        W, H = p["fuselage_width"], p["fuselage_height"]
        x_nose, x_end = -p["nose_length"], -p["nose_length"] + p["fuselage_length"]
        b2, c0, lam = p["span"] / 2, p["root_chord"], p["taper"]
        tan_le = math.tan(math.radians(p["sweep_deg"])) + (1 - lam) * c0 / (4 * b2)       # leading-edge sweep
        mac = (2 / 3) * c0 * (1 + lam + lam ** 2) / (1 + lam)
        y_mac = b2 / 3 * (1 + 2 * lam) / (1 + lam)
        Rn = p["nacelle_diameter"] / 2
        out = {"W": W, "H": H, "x_nose": x_nose, "x_end": x_end, "x_cone": x_end - p["tail_cone"], "tan_le": tan_le,
               "mac": mac, "y_mac": y_mac, "x_ac": y_mac * tan_le + 0.25 * mac, "Rn": Rn,
               "x_lip": p["nacelle_x"], "x_exit": p["nacelle_x"] + p["nacelle_length"],
               "nacelle_y": W / 2 * 0.80 + Rn, "nacelle_z": 0.35 * Rn,
               "r_intake": p["intake_diameter"] / 2, "r_nozzle": p["nozzle_diameter"] / 2,
               "x_tail_le": x_end - p["vtail_chord"], "bay_x0": p["bay_x"] - p["bay_length"] / 2,
               "bay_x1": p["bay_x"] + p["bay_length"] / 2, "boom_z": -0.02 * c0 - p["boom_diameter"] / 2}
        if p["intake_diameter"] >= p["nacelle_diameter"] - 6 or p["nozzle_diameter"] >= p["nacelle_diameter"]:
            raise ValueError("intake and nozzle must fit inside the nacelle")
        if p["nose_shape"] + p["tail_cone"] >= p["fuselage_length"]:
            raise ValueError("nose_shape + tail_cone exceed the fuselage length")
        if out["bay_x0"] < x_nose + p["nose_shape"] or out["bay_x1"] > out["x_cone"] + 0.4 * p["tail_cone"]:
            raise ValueError("the bay does not fit between the nose and the tail cone")
        if p["bay_width"] > W - 40 or p["bay_height"] > H - 60:
            raise ValueError("the bay does not fit the fuselage's cross-section (walls, the wing's carry-through)")
        if p["lift_rotors"] not in (0, 4):
            raise ValueError("lift_rotors is 0 or 4")
        if p["lift_rotors"] and p["boom_y"] - p["rotor_diameter"] / 2 < W / 2 + 20:
            raise ValueError("the lift rotors reach into the fuselage: move the booms out")
        return out

    @staticmethod
    def section_scale(p, x) -> tuple:
        """(scale, centre z) of the fuselage's cross-section at station x: an elliptic nose, the constant body, the tail
        cone tapering to ``tail_end`` with its top line level (the belly sweeps up: clear of the deck and the water)."""
        L = Albatros.layout(p)
        H = p["fuselage_height"]
        if x < L["x_nose"] + p["nose_shape"]:
            t = max((x - L["x_nose"]) / p["nose_shape"], 0.0)
            return max(math.sqrt(max(1 - (1 - t) ** 2, 0.0)), 0.0), -H / 2
        if x <= L["x_cone"]:
            return 1.0, -H / 2
        t = min((x - L["x_cone"]) / p["tail_cone"], 1.0)
        s = 1 - (1 - p["tail_end"]) * (3 * t ** 2 - 2 * t ** 3)
        return s, -s * H / 2

    @staticmethod
    def stations(p, n_nose=7, n_tail=7) -> np.ndarray:
        L = Albatros.layout(p)
        nose = L["x_nose"] + p["nose_shape"] * (1 - np.cos(np.linspace(0.12, 1, n_nose) * math.pi / 2))
        tail = L["x_cone"] + p["tail_cone"] * np.linspace(0, 1, n_tail)
        return np.concatenate([nose, tail])

    @staticmethod
    def nacelle_profile(p, n=10) -> np.ndarray:
        """Half profile (x, r) of a pod's outside about its own axis (as AGUYA's: an elliptic lip, the cylinder, the
        boattail down to the nozzle)."""
        L = Albatros.layout(p)
        Rn, ri, rn = L["Rn"], L["r_intake"], L["r_nozzle"]
        lip = 0.35 * p["nacelle_diameter"]
        pts = [(L["x_lip"] + lip * (1 - math.cos(a)), ri + (Rn - ri) * math.sin(a)) for a in np.linspace(0, math.pi / 2, n)]
        x_bt = L["x_exit"] - p["boattail"]
        pts += [(x_bt + p["boattail"] * u, Rn - (Rn - rn - 2.0) * (3 * u ** 2 - 2 * u ** 3)) for u in np.linspace(0, 1, n)]
        return np.array(pts)

    # ----------------------------------------------------------------------------------------- parts
    @staticmethod
    def _superellipse(w, h, zc, n=40):
        e = 2.0 / SUPER_N
        pts = []
        for a in np.linspace(0, 2 * math.pi, n, endpoint=False):
            c, s = math.cos(a), math.sin(a)
            pts.append((math.copysign(abs(c) ** e, c) * w / 2, zc + math.copysign(abs(s) ** e, s) * h / 2))
        return pts

    def _fuselage(self, p):
        W, H = p["fuselage_width"], p["fuselage_height"]
        wires = []
        for x in self.stations(p):
            s, zc = self.section_scale(p, float(x))
            pts = [cq.Vector(float(x), y, z) for y, z in self._superellipse(W * s, H * s, zc)]
            wires.append(cq.Wire.assembleEdges([cq.Edge.makeSpline(pts, periodic=True)]))
        return cq.Workplane("XY").newObject([cq.Solid.makeLoft(wires)])

    def _section(self, wp, c, dx, dz, p):
        up, lo = naca4(p["camber"], p["camber_pos"], p["thickness"])
        pu = [(dx + x * c, dz + y * c) for x, y in up]
        pl = [(dx + x * c, dz + y * c) for x, y in lo]
        return wp.moveTo(*pu[0]).spline(pu[1:], includeCurrent=True).spline(pl[::-1][1:], includeCurrent=True).close()

    def _wing(self, p):
        L = self.layout(p)
        c0, c1, b2 = p["root_chord"], p["root_chord"] * p["taper"], p["span"] / 2
        dx, dz = L["tan_le"] * b2, b2 * math.tan(math.radians(p["dihedral_deg"]))
        z0 = 0.03 * c0                                              # the root section sits just into the fuselage's top
        panel = self._section(cq.Workplane("XZ"), c0, 0.0, z0, p).workplane(offset=b2).moveTo(0, 0)
        panel = self._section(panel, c1, dx, z0 + dz, p).loft(combine=True, ruled=True)
        return panel.union(panel.mirror("XZ"))

    def _pod(self, p):
        """One pod about the x axis (not placed): the intake face ``depth`` in, the nozzle face 5 mm in."""
        L = self.layout(p)
        ri, rn = L["r_intake"], L["r_nozzle"]
        prof = self.nacelle_profile(p)
        xf = L["x_lip"] + 0.3 * p["nacelle_diameter"]
        pts = [(xf, 0.0), (xf, ri), (L["x_lip"], ri)] + [tuple(q) for q in prof[1:]]
        pts += [(L["x_exit"], rn), (L["x_exit"] - 5.0, rn), (L["x_exit"] - 5.0, 0.0)]
        return cq.Workplane("XY").polyline(pts).close().revolve(360, (0, 0, 0), (1, 0, 0))

    def _nacelles(self, p):
        """The pods on the aft fuselage: one on a dorsal pylon, two on the shoulders, or three to six in a ring around
        the tail (the tail-sitter's cluster)."""
        L = self.layout(p)
        pod, Rn, H, n = self._pod(p), L["Rn"], L["H"], int(p["engines"])
        y0, z0 = L["nacelle_y"], L["nacelle_z"]
        x0, x1 = L["x_lip"] + 0.35 * p["nacelle_length"], L["x_exit"] - p["boattail"]
        s_mid, zc = self.section_scale(p, 0.5 * (x0 + x1))
        if n == 1:
            places = [(0.0, Rn + 0.25 * H)]
        elif n == 2:
            places = [(-y0, z0), (y0, z0)]
        else:
            ring = 0.5 * s_mid * max(L["W"], H) + Rn
            places = [(ring * math.sin(2 * math.pi * (k + 0.5) / n), zc + ring * math.cos(2 * math.pi * (k + 0.5) / n)) for k in range(n)]
        out = None
        for y, z in places:
            if n == 2:                                              # a stub wing from the centreline out into the pod
                stub = (cq.Workplane("XY").box(x1 - x0, abs(y), 0.5 * Rn, centered=(False, False, True))
                        .translate((x0, min(y, 0.0), min(z, 0.0) - 0.3 * Rn)))
            else:                                                   # a strut from the fuselage's axis out to the pod
                r, a = math.hypot(y, z - zc), math.degrees(math.atan2(y, z - zc))
                stub = (cq.Workplane("XY").box(x1 - x0, 0.5 * Rn, r, centered=(False, True, False))
                        .rotate((0, 0, 0), (1, 0, 0), -a).translate((x0, 0, zc)))
            unit = pod.translate((0, y, z)).union(stub)
            out = unit if out is None else out.union(unit)
        return out

    def _booms(self, p):
        L = self.layout(p)
        r, D = p["boom_diameter"] / 2, p["rotor_diameter"]
        x0, x1 = p["rotor_x_front"] - 0.12 * D, p["rotor_x_rear"] + 0.12 * D
        boom = (cq.Workplane("YZ").circle(r).extrude(x1 - x0).translate((x0, 0, 0))
                .union(cq.Workplane("XY").sphere(r).translate((x0, 0, 0))).union(cq.Workplane("XY").sphere(r).translate((x1, 0, 0))))
        can_r, can_h = 0.06 * D, 0.07 * D
        for x in (p["rotor_x_front"], p["rotor_x_rear"]):
            can = cq.Workplane("XY").circle(can_r).extrude(r + can_h).translate((x, 0, 0))
            blade = cq.Workplane("XY").box(D, 0.07 * D, 0.012 * D, centered=(True, True, False)).translate((x, 0, r + can_h - 0.006 * D))
            boom = boom.union(can).union(blade)
        z = L["boom_z"]
        # a short pylon up into the wing at each boom
        b2 = p["span"] / 2
        x_le = L["tan_le"] * p["boom_y"]
        c = p["root_chord"] * (1 - (1 - p["taper"]) * p["boom_y"] / b2)
        pyl = cq.Workplane("XY").box(0.6 * c, 0.25 * p["boom_diameter"] + 6, abs(z) + 0.05 * c, centered=(False, True, False)).translate((x_le + 0.2 * c, 0, 0))
        unit = boom.union(pyl)
        dzw = p["boom_y"] * math.tan(math.radians(p["dihedral_deg"]))
        return unit.translate((0, p["boom_y"], z + dzw)).union(unit.translate((0, -p["boom_y"], z + dzw)))

    def _vtail(self, p):
        L = self.layout(p)
        ch, b, t = p["vtail_chord"], p["vtail_span"], p["tail_thickness"] * p["vtail_chord"]
        plate = cq.Workplane("XY").box(ch, t, b + 0.1 * L["H"], centered=(False, True, False)).translate((0, 0, -0.1 * L["H"]))
        g = 90.0 - p["vtail_dihedral_deg"]
        s_end, _ = self.section_scale(p, L["x_end"] - 0.5 * ch)
        both = plate.rotate((0, 0, 0), (1, 0, 0), g).union(plate.rotate((0, 0, 0), (1, 0, 0), -g))
        return both.translate((L["x_tail_le"], 0, -0.25 * s_end * L["H"]))

    def _bay_cut(self, p):
        L = self.layout(p)
        return (cq.Workplane("XY").box(p["bay_length"], p["bay_width"], p["bay_height"] + 5.0, centered=(False, True, False))
                .translate((L["bay_x0"], 0, -p["fuselage_height"] - 5.0)))

    def _multirotor(self, p):
        n, arm, D = int(p["multirotor_rotors"]), p["multirotor_arm"], p["multirotor_prop"]
        hub = max(0.28 * arm, 60.0)
        body = cq.Workplane("XY").circle(hub).extrude(40.0)
        for k in range(n):
            a = 360.0 * (k + 0.5) / n
            one = (cq.Workplane("XY").box(arm, 0.045 * arm + 8, 22.0, centered=(False, True, False)).translate((0, 0, 9.0))
                   .union(cq.Workplane("XY").circle(0.05 * D + 8).extrude(55.0).translate((arm, 0, 0)))
                   .union(cq.Workplane("XY").box(D, 0.08 * D, 5.0, centered=(True, True, False)).translate((arm, 0, 55.0))
                          .rotate((arm, 0, 0), (arm, 0, 1), 90.0)))
            body = body.union(one.rotate((0, 0, 0), (0, 0, 1), a))
        if p["cargo_height"] > 0:
            side = 1.6 * hub
            body = body.union(cq.Workplane("XY").box(side, side, p["cargo_height"] + 5.0, centered=(True, True, False))
                              .translate((0, 0, -p["cargo_height"])))
        return body

    def build(self, p):
        if p["part"] == "wing":
            return self._wing(p)
        if p["part"] == "nacelle":
            return self._pod(p)
        if p["part"] == "multirotor":
            return self._multirotor(p)
        aircraft = self._fuselage(p).union(self._wing(p)).union(self._nacelles(p)).union(self._vtail(p))
        if p["lift_rotors"]:
            aircraft = aircraft.union(self._booms(p))
        if p["part"] == "aircraft_open":
            aircraft = aircraft.cut(self._bay_cut(p))
        if p["angle_of_attack_deg"]:
            aircraft = aircraft.rotate((0, 0, 0), (0, 1, 0), p["angle_of_attack_deg"])
        return aircraft


# ----------------------------------------------------------------------------------------------- helpers
def _resolve(p):
    return Albatros().resolve(**dict(p or {}))


def for_design(*, wing_area_m2: float, aspect_ratio: float, bay: dict, fuel_kg: float, engine, engines: int = 2,
               layout: str = "catapult", lift_rotors: int = 0, rotor_diameter_m: float = 0.9, sweep_deg: float = 20.0,
               taper: float = 0.45, multirotor=None, buried_engines: int = 0, buried_diameter_m: float = 0.0,
               fuel_density: float = 800.0,
               tank_fill: float = 0.8, **overrides) -> dict:
    """Albatros's parameters from a sizing: the wing from its area and aspect ratio; the fuselage a rounded box around
    the bay (``albatros_bay.BayOption.bay()``), as long as the nose, the avionics bay, the two fuel tanks, the bay,
    the buried lift engines and the tail cone need; the pods around the ``vegeta.boreas.microjet.Microjet``; the
    booms and rotors of the lift system. The bay's middle is put on the wing's aerodynamic centre."""
    b = math.sqrt(aspect_ratio * wing_area_m2) * 1000
    c0 = 2 * wing_area_m2 * 1e6 / (b * (1 + taper))
    W, H = bay["fuselage_width_m"] * 1000, bay["fuselage_height_m"] * 1000
    section = SECTION_AREA * W * H * 1e-6                                           # m²
    tank_len = fuel_kg / fuel_density / tank_fill / (0.80 * section) * 1000         # both tanks together, mm
    nose_shape, avionics, tail_cone = 1.45 * H, 300.0, 2.0 * max(W, H)
    lift_bay = buried_engines * ((buried_diameter_m or engine.outer_diameter) * 1000 + 60.0)
    bay_len = bay["length_m"] * 1000
    mac = (2 / 3) * c0 * (1 + taper + taper ** 2) / (1 + taper)
    y_mac = b / 6 * (1 + 2 * taper) / (1 + taper)
    tan_le = math.tan(math.radians(sweep_deg)) + (1 - taper) * c0 / (2 * b)
    x_ac = y_mac * tan_le + 0.25 * mac
    ahead = nose_shape + avionics + lift_bay / 2 + tank_len / 2                      # nose tip to the bay's front
    length = ahead + bay_len + tank_len / 2 + lift_bay / 2 + 0.6 * tail_cone
    length = max(length, ahead + bay_len + tail_cone * 0.75 + 0.9 * engine.length * 1000)
    d = engine.outer_diameter * 1000
    nac_len = engine.length * 1000 + 60.0
    x_end = -(ahead + bay_len / 2 - x_ac) + length
    D = rotor_diameter_m * 1000
    boom_y = max(W / 2 + D / 2 + 60.0, 0.30 * b) if lift_rotors else 0.30 * b
    out = dict(layout=layout, span=b, root_chord=c0, taper=taper, sweep_deg=sweep_deg, fuselage_length=length, fuselage_width=W,
               fuselage_height=H, nose_length=ahead + bay_len / 2 - x_ac, nose_shape=nose_shape, tail_cone=tail_cone,
               bay_length=bay_len, bay_width=bay["width_m"] * 1000, bay_height=bay["height_m"] * 1000, bay_x=x_ac,
               engines=engines, nacelle_diameter=d + 14.0, nacelle_length=nac_len, intake_diameter=0.75 * d,
               nozzle_diameter=2 * math.sqrt(engine.a8 / math.pi) * 1000, boattail=0.85 * d,
               nacelle_x=x_end - 0.55 * tail_cone - nac_len, lift_rotors=lift_rotors, rotor_diameter=D, boom_y=boom_y,
               boom_diameter=max(0.085 * D, 50.0), rotor_x_front=x_ac - 0.62 * D - 0.45 * c0, rotor_x_rear=x_ac + 0.62 * D + 0.45 * c0,
               vtail_span=0.5 * math.sqrt(0.24 * wing_area_m2 * 1.6) * 1000 * 1.25, vtail_chord=0.5 * math.sqrt(0.24 * wing_area_m2 / 1.6) * 1000 * 1.6)
    if multirotor is not None:
        cargo_h = max(multirotor.folded_m[2] - 0.12, 0.0) * 1000
        out.update(multirotor_rotors=multirotor.rotors, multirotor_arm=multirotor.arm_m * 1000,
                   multirotor_prop=multirotor.prop_diameter_m * 1000, cargo_height=cargo_h)
    out.update(overrides)
    return Albatros().resolve(**out)


def wetted_areas(p) -> dict:
    p = _resolve(p)
    L = Albatros.layout(p)
    b, c0, lam = p["span"] / 1000, p["root_chord"] / 1000, p["taper"]
    S = b * c0 * (1 + lam) / 2
    W, H = p["fuselage_width"] / 1000, p["fuselage_height"] / 1000
    xs = np.linspace(L["x_nose"], L["x_end"], 80)
    sc = np.array([Albatros.section_scale(p, float(x))[0] for x in xs])
    fus = float(np.trapezoid(SECTION_PERIMETER * 2 * (W + H) * sc, xs / 1000))
    nac = Albatros.nacelle_profile(p)
    nac_w = float(np.sum(np.pi * (nac[1:, 1] + nac[:-1, 1]) * np.hypot(np.diff(nac[:, 0]), np.diff(nac[:, 1])))) * 1e-6
    D = p["rotor_diameter"] / 1000
    boom_len = (p["rotor_x_rear"] - p["rotor_x_front"]) / 1000 + 0.24 * D
    return {"wing": 2 * 1.02 * (S - W * c0), "fuselage": fus, "nacelles": p["engines"] * nac_w,
            "booms": (2 * math.pi * p["boom_diameter"] / 1000 * boom_len) if p["lift_rotors"] else 0.0,
            "vtail": 2 * 2 * p["vtail_span"] * p["vtail_chord"] * 1e-6, "planform": S, "aspect_ratio": b ** 2 / S, "span": b,
            "wing_mac": L["mac"] / 1000, "fuselage_length": p["fuselage_length"] / 1000,
            "fuselage_diameter": math.sqrt(4 * SECTION_AREA * W * H / math.pi), "frontal_area": SECTION_AREA * W * H,
            "nacelle_length": p["nacelle_length"] / 1000, "nacelle_diameter": p["nacelle_diameter"] / 1000,
            "boom_length": boom_len, "boom_diameter": p["boom_diameter"] / 1000, "thickness": p["thickness"],
            "tail_chord": p["vtail_chord"] / 1000, "tail_t_c": p["tail_thickness"], "sweep_deg": p["sweep_deg"],
            "rotors": p["lift_rotors"], "rotor_diameter": D}


def drag_buildup(p, speed: float, nu: float = 1.5e-5, mach: float = 0.0) -> dict:
    """Parasite drag area [m^2] by Raymer's component build-up (as AGUYA's): turbulent flat-plate friction on each
    component's length, its form factor, its wetted area; the stopped lift rotors as ``ROTOR_DRAG`` x diameter^2 each
    (two blades locked fore-and-aft, the motor's can); the tail-sitter's four leg fairings as a fifth of its tail; plus
    12 % for interference, the bay doors' gaps, antennas and the flotation seals."""
    a = wetted_areas(p)
    p = _resolve(p)

    def cf(length):
        return cf_turbulent(speed * length / nu, mach, re_min=1e4)

    def ff_wing(tc, sweep=0.0):
        return (1 + 0.6 / 0.3 * tc + 100 * tc ** 4) * (1.34 * mach ** 0.18 * math.cos(math.radians(sweep)) ** 0.28 if mach > 0.2 else 1.0)

    def ff_body(f):
        return 1 + 60 / f ** 3 + f / 400

    parts = {"wing": cf(a["wing_mac"]) * ff_wing(a["thickness"], a["sweep_deg"]) * a["wing"],
             "fuselage": cf(a["fuselage_length"]) * ff_body(a["fuselage_length"] / a["fuselage_diameter"]) * a["fuselage"],
             "nacelles": cf(a["nacelle_length"]) * (1 + 0.35 / (a["nacelle_length"] / a["nacelle_diameter"])) * a["nacelles"] * 1.3,
             "vtail": cf(a["tail_chord"]) * ff_wing(a["tail_t_c"]) * a["vtail"]}
    if a["rotors"]:
        parts["lift booms"] = cf(a["boom_length"]) * ff_body(a["boom_length"] / a["boom_diameter"]) * a["booms"] * 1.3
        parts["stopped lift rotors"] = a["rotors"] * ROTOR_DRAG * a["rotor_diameter"] ** 2
    if p["layout"] == "tailsitter":
        parts["landing legs"] = 0.2 * parts["vtail"] + 4 * 0.0006
    total = 1.12 * sum(parts.values())
    return {"parts_m2": parts, "cd_area_m2": total, "cd0": total / a["planform"], "planform": a["planform"],
            "aspect_ratio": a["aspect_ratio"]}


def outline(p) -> dict:
    """Side (x, z) and top (x, y) polygons in metres: fuselage, wing, pods, booms and rotor disks, the bay."""
    p = _resolve(p)
    L = Albatros.layout(p)
    xs = np.linspace(L["x_nose"], L["x_end"], 60)
    sc = np.array([Albatros.section_scale(p, float(x)) for x in xs])
    W, H = p["fuselage_width"], p["fuselage_height"]
    top, bot = sc[:, 1] + sc[:, 0] * H / 2, sc[:, 1] - sc[:, 0] * H / 2
    side = [np.column_stack([np.r_[xs, xs[::-1]], np.r_[top, bot[::-1]]]) / 1000]
    plan = [np.column_stack([np.r_[xs, xs[::-1]], np.r_[sc[:, 0] * W / 2, -sc[::-1, 0] * W / 2]]) / 1000]
    b2, c0, c1 = p["span"] / 2, p["root_chord"], p["root_chord"] * p["taper"]
    dx = L["tan_le"] * b2
    plan.append(np.array([[0, 0], [dx, b2], [dx + c1, b2], [c0, 0], [dx + c1, -b2], [dx, -b2]]) / 1000)
    up, lo = naca4(p["camber"], p["camber_pos"], p["thickness"])
    side.append(np.array([(x * c0, y * c0 + 0.03 * c0) for x, y in up + lo[::-1]]) / 1000)
    nac = Albatros.nacelle_profile(p)
    for sgn in ((-1, 1) if p["engines"] > 1 else (0,)):
        plan.append((np.vstack([nac, nac[::-1] * [1, -1]]) + [0, sgn * L["nacelle_y"]]) / 1000)
    side.append((np.vstack([nac, nac[::-1] * [1, -1]]) + [0, L["nacelle_z"]]) / 1000)
    h = p["vtail_span"] * math.sin(math.radians(p["vtail_dihedral_deg"]))
    w = p["vtail_span"] * math.cos(math.radians(p["vtail_dihedral_deg"]))
    side.append(np.array([[L["x_tail_le"], 0], [L["x_end"], 0], [L["x_end"], h], [L["x_tail_le"] + 0.3 * p["vtail_chord"], h]]) / 1000)
    for sgn in (-1, 1):
        plan.append(np.array([[L["x_tail_le"], 0], [L["x_end"], 0], [L["x_end"], sgn * w], [L["x_tail_le"] + 0.3 * p["vtail_chord"], sgn * w]]) / 1000)
    rotors = []
    if p["lift_rotors"]:
        r = p["boom_diameter"] / 2
        x0, x1 = p["rotor_x_front"] - 0.12 * p["rotor_diameter"], p["rotor_x_rear"] + 0.12 * p["rotor_diameter"]
        for sgn in (-1, 1):
            y = sgn * p["boom_y"]
            plan.append(np.array([[x0, y - r], [x1, y - r], [x1, y + r], [x0, y + r]]) / 1000)
            rotors += [(p["rotor_x_front"] / 1000, y / 1000), (p["rotor_x_rear"] / 1000, y / 1000)]
        side.append(np.array([[x0, L["boom_z"] - r], [x1, L["boom_z"] - r], [x1, L["boom_z"] + r], [x0, L["boom_z"] + r]]) / 1000)
    bay = np.array([[L["bay_x0"], -p["bay_width"] / 2], [L["bay_x1"], -p["bay_width"] / 2], [L["bay_x1"], p["bay_width"] / 2],
                    [L["bay_x0"], p["bay_width"] / 2]]) / 1000
    bay_side = np.array([[L["bay_x0"], -H], [L["bay_x1"], -H], [L["bay_x1"], -H + p["bay_height"]], [L["bay_x0"], -H + p["bay_height"]]]) / 1000
    return {"side": side, "plan": plan, "rotors": rotors, "rotor_radius": p["rotor_diameter"] / 2000, "bay_plan": bay,
            "bay_side": bay_side, "x_ac": L["x_ac"] / 1000, "nozzle_x": L["x_exit"] / 1000}


__all__ = ["Albatros", "for_design", "wetted_areas", "drag_buildup", "outline", "ROTOR_DRAG", "SECTION_AREA"]
