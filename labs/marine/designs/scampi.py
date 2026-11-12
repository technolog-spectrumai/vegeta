"""Scampi: a small work-class electric AUV for offshore wind — swims first, lands on the seabed to work. The Sikarian
Lobster's successor (``origin/dev_lobster``: Nefri / Ornatus, the underwater walker with pincers and a tail
thruster), grown into a hover-capable vehicle: no walking legs, four fixed landing skids; a hold arm and a work arm
at the front; swappable tool skids under the belly; an NVIDIA Jetson AGX Orin for autonomy.

The requirements (``REQUIREMENTS``, from the design questions) and what the sizing (``sizing``) says against them:

* offshore wind first — monopiles, J-tubes, cable crossings, scour; inspection, marine-growth cleaning, light
  intervention (ropes, debris, cable protection), CP probes, sensors and small anodes;
* small work class, 60-120 kg in air, ~1.5 m long, launched from a workboat davit;
* universal vehicle + tool skids (belly bay, wet-mate 48 V + Ethernet);
* untethered, acoustic modem for status, an optional fibre micro-tether for supervised arm work;
* one 6-DOF work arm and one 4-DOF hold (grabber) arm;
* rated to 300 m (design pressure ``SAFETY`` x 3.1 MPa);
* 6-8 h per dive.

Layout (x forward, y left, z up; z = 0 the skids' soles): an open HDPE frame under a free-flooding fairing; syntactic
foam in the top third (centre of buoyancy high), two aluminium pressure housings low in the frame (electronics,
battery: centre of gravity low, so the vehicle is stable in roll and pitch without control); eight thrusters, four
horizontal at 45 deg in the corners (surge, sway, yaw) and four vertical (heave, roll, pitch); the two arms at the nose,
folded under the fairing in transit; four skids with compliant pads at the corners.

``part``: ``vehicle`` (everything, arms posed by ``arm_pose``), ``frame``, ``fairing``, ``foam``, ``housing``
(``housing`` selects which), ``thruster`` (one), ``work_arm``, ``hold_arm``, ``skid`` (one), ``tool_skid``.
"""
from __future__ import annotations

import math

try:
    import cadquery as cq
except ImportError:                 # the sizing runs without the CAD
    cq = None
from vegeta.dedalus import Design, Parameter

RHO_SW = 1025.0                     # kg/m^3 seawater
G = 9.81
SAFETY = 1.5                        # on the rated depth's pressure (DNV-ST-E407 class practice for housings)
AL6061 = dict(E=68.9e9, nu=0.33, yield_=276e6, rho=2700.0)
FOAM_RHO = 420.0                    # kg/m^3: syntactic foam rated ~600 m (2x the rating; supplier class, assumed)

REQUIREMENTS = {
    "market": "offshore wind (fixed-bottom, floating moorings later)",
    "size": "small work class, 60-120 kg in air, about 1.5 m",
    "approach": "universal vehicle + swappable tool skids",
    "comms": "untethered; acoustic modem; optional fibre micro-tether for supervised arm work",
    "arms": "one 6-DOF work arm + one 4-DOF hold arm",
    "depth_m": 300.0,
    "endurance_h": (6.0, 8.0),
    "jobs": ["monopile and cable inspection", "marine-growth cleaning", "light intervention", "sensor / CP anode work"],
    "current_m_s": 1.0,             # station keeping in a tidal current (North Sea class, assumed)
}

#: Bought parts: mass in air [kg], displaced volume [L], power [W] (average while on), source.
PARTS = {
    "Jetson AGX Orin 64GB + carrier": (0.9, 0.0, 40.0, "NVIDIA module 15-60 W; in the electronics housing"),
    "INS (FOG class) + depth sensor": (2.2, 0.9, 20.0, "subsea INS class, assumed"),
    "DVL 600 kHz": (1.6, 0.6, 15.0, "DVL class, assumed"),
    "imaging / multibeam sonar": (2.5, 1.2, 35.0, "forward-looking sonar class, assumed"),
    "acoustic modem + USBL transponder": (1.2, 0.5, 5.0, "average; 30-60 W while sending"),
    "4 cameras + 4 LED lights": (4.0, 1.6, 120.0, "lights at about half power on average"),
    "CP probe + contact probe": (0.6, 0.3, 2.0, "on the hold arm's wrist"),
    "8 thrusters (T500 class)": (8 * 1.0, 8 * 0.35, 0.0, "power from the thrust model; rating to 300 m assumed"),
    "work arm 6-DOF electric": (9.0, 3.5, 0.0, "power from the duty in ENERGY_PROFILE"),
    "hold arm 4-DOF electric": (5.0, 2.0, 0.0, ""),
    "wet-mate connectors, cables": (2.5, 0.8, 0.0, ""),
}
THRUSTER = {"forward_N": 157.0, "reverse_N": 120.0, "max_W": 1100.0, "eta_bollard_N_per_W": 0.23,
            "source": "T500-class at 24 V: ~16 kgf forward; thrust per shaft power at part load (assumed)"}


class Scampi(Design):
    parameters = [
        Parameter("part", "vehicle", choices=("vehicle", "frame", "fairing", "foam", "housing", "thruster", "work_arm",
                                              "hold_arm", "skid", "tool_skid"), description="what to build"),
        Parameter("length", 1500.0, "mm", min=600, description="frame length (without the arms)"),
        Parameter("width", 800.0, "mm", min=300),
        Parameter("height", 560.0, "mm", min=250, description="skid soles to the fairing's top"),
        Parameter("skid_height", 140.0, "mm", min=50, description="frame bottom above the soles"),
        Parameter("frame_bar", 30.0, "mm", min=10, description="HDPE frame plate thickness"),
        Parameter("foam_height", 110.0, "mm", min=30, description="syntactic foam block under the fairing's top"),
        Parameter("housing", "electronics", choices=("electronics", "battery"), description="which housing for part=housing"),
        Parameter("housing_od", 200.0, "mm", min=60, description="both pressure housings, outer diameter"),
        Parameter("housing_wall", 7.0, "mm", min=1, description="Al 6061-T6; buckling decides (sizing['housings'])"),
        Parameter("electronics_length", 600.0, "mm", min=100),
        Parameter("battery_length", 700.0, "mm", min=100),
        Parameter("cap_thickness", 18.0, "mm", min=3, description="flat end caps"),
        Parameter("battery_kwh", 4.5, "kWh", min=0.5, description="Li-ion 48 V pack (the 7 h dive + 10 % margin)"),
        Parameter("payload_foam_L", 8.0, "L", min=0, description="foam beyond neutral: lifts a recovered object (tool skids carry their own foam: neutral)"),
        Parameter("cell_wh_kg", 250.0, "Wh/kg", min=100, description="cells (21700 class)"),
        Parameter("thruster_duct", 110.0, "mm", min=40, description="thruster duct outer diameter"),
        Parameter("thruster_length", 120.0, "mm", min=40),
        Parameter("arm_reach", 900.0, "mm", min=200, description="work arm, shoulder to the jaw tips"),
        Parameter("hold_reach", 650.0, "mm", min=200),
        Parameter("arm_link", 70.0, "mm", min=20, description="arm link diameter"),
        Parameter("arm_pose", "work", choices=("folded", "work"), description="arms folded under the nose or reaching out"),
        Parameter("tool_bay_length", 600.0, "mm", min=100, description="belly tool skid"),
        Parameter("tool_bay_width", 420.0, "mm", min=100),
        Parameter("tool_bay_height", 80.0, "mm", min=30, description="the skid fits between its 60 mm floor and the frame"),
        Parameter("show_tool_skid", True, "", description="draw the tool skid in the vehicle"),
    ]

    # ------------------------------------------------------------------------------------------- sizing
    @staticmethod
    def housing_check(p, length_mm, depth_m=REQUIREMENTS["depth_m"]):
        """One Al 6061-T6 housing at ``SAFETY`` x the rated pressure: hoop stress (thin wall) and elastic buckling
        of a long tube, 2E/(1-nu^2) (t/D)^3 — conservative for a tube stiffened by its caps (Windenburg-Trilling
        gives more for L/D ~ 3)."""
        D, t = p["housing_od"] / 1000, p["housing_wall"] / 1000
        P = SAFETY * RHO_SW * G * depth_m
        hoop = P * (D / 2) / t
        p_cr = 2 * AL6061["E"] / (1 - AL6061["nu"] ** 2) * (t / D) ** 3
        L = length_mm / 1000
        tube = math.pi * D * t * L * AL6061["rho"]
        caps = 2 * math.pi * (D / 2) ** 2 * p["cap_thickness"] / 1000 * AL6061["rho"]
        return {"design_pressure_MPa": P / 1e6, "hoop_MPa": hoop / 1e6, "hoop_margin": AL6061["yield_"] / 1.5 / hoop - 1,
                "buckling_MPa": p_cr / 1e6, "buckling_margin": p_cr / P - 1, "mass_kg": tube + caps,
                "volume_L": math.pi * (D / 2) ** 2 * (L + 2 * p["cap_thickness"] / 1000) * 1000}

    @staticmethod
    def drag_area(p, axis="x", cd=0.9):
        """Cd x A [m^2] of the faired frame along an axis (open-frame work vehicles: Cd 0.8-1.1)."""
        L, W, H = p["length"] / 1000, p["width"] / 1000, (p["height"] - p["skid_height"]) / 1000
        A = {"x": W * H, "y": L * H, "z": L * W}[axis]
        return cd * A

    @classmethod
    def thrust(cls, p):
        """Surge / sway / heave thrust from the eight thrusters (four horizontal at 45 deg, four vertical) [N]."""
        f = THRUSTER["forward_N"]
        return {"x": 4 * f * math.cos(math.radians(45)), "y": 4 * f * math.sin(math.radians(45)), "z": 4 * f}

    @classmethod
    def propulsion_power(cls, p, v, axis="x"):
        """Electrical power [W] to hold speed ``v`` [m/s] through the water along ``axis``."""
        D = 0.5 * RHO_SW * cls.drag_area(p, axis) * v * v
        return D / THRUSTER["eta_bollard_N_per_W"] * (0.6 if v > 0.6 else 1.0)    # better at speed (inflow)

    @classmethod
    def energy_profile(cls, p):
        """One 7 h offshore-wind dive [h, W]: transit, inspection, cleaning, intervention, on the seabed, return."""
        hotel = sum(w for _, _, w, _ in PARTS.values())
        cur = REQUIREMENTS["current_m_s"] * 0.5                                 # mean of a tidal cycle
        return [
            ("transit out 1.0 m/s", 0.5, hotel + cls.propulsion_power(p, 1.0)),
            ("inspection along a monopile 0.3 m/s in current", 3.0, hotel + cls.propulsion_power(p, 0.3 + cur)),
            ("cleaning skid on the weld (hold arm on)", 1.0, hotel + 0.4 * cls.propulsion_power(p, cur) + 250.0),
            ("intervention, both arms", 1.0, hotel + 0.4 * cls.propulsion_power(p, cur) + 180.0),
            ("landed on the seabed: CP and sensor work", 1.0, hotel - 60.0 + 60.0),
            ("transit back 1.0 m/s", 0.5, hotel + cls.propulsion_power(p, 1.0)),
        ]

    @classmethod
    def sizing(cls, p=None):
        """Housings, mass and buoyancy (the foam that makes Scampi 0.5 kg positive), stability, thrust against the
        current, the dive's energy against the battery."""
        p = {q.name: q.default for q in cls.parameters} | (p or {})
        he = cls.housing_check(p, p["electronics_length"])
        hb = cls.housing_check(p, p["battery_length"])
        cells = p["battery_kwh"] * 1000 / p["cell_wh_kg"]
        frame_kg = 14.0                                                        # HDPE plates 30 mm (rho 950), fairing
        items = {"frame + fairing + skids (HDPE)": (frame_kg, frame_kg / 0.95)}
        items["electronics housing"] = (he["mass_kg"] + 2.0, he["volume_L"])  # + boards, power supplies
        items["battery housing + cells + BMS"] = (hb["mass_kg"] + cells + 1.5, hb["volume_L"])
        for k, (m, v, _, _) in PARTS.items():
            if not k.startswith("Jetson"):
                items[k] = (m, v)
        items["Jetson AGX Orin 64GB + carrier"] = (PARTS["Jetson AGX Orin 64GB + carrier"][0], 0.0)
        m_dry = sum(m for m, _ in items.values())
        v_dry = sum(v for _, v in items.values()) / 1000
        net_up = 0.5                                                           # kg positive: floats up if it dies
        # foam: (m_dry + rho_f V) = rho_sw (v_dry + V) - net_up, plus the payload foam (trimmed out by ballast weights
        # without a tool skid: lead in the skid posts, ~the payload's in-water weight)
        V_n = (m_dry + net_up - RHO_SW * v_dry) / (RHO_SW - FOAM_RHO)
        V_p = p["payload_foam_L"] / 1000
        V_f = V_n + V_p
        ballast_kg = V_p * (RHO_SW - FOAM_RHO) / (1 - RHO_SW / 11340.0)          # lead to trim it out, bare
        m_total = m_dry + FOAM_RHO * V_f + ballast_kg
        foam_avail = (p["length"] - 2 * p["frame_bar"]) * (p["width"] - 2 * p["frame_bar"]) * p["foam_height"] / 1e9
        th = cls.thrust(p)
        v_max = {a: math.sqrt(2 * th[a] / (RHO_SW * cls.drag_area(p, a))) for a in th}
        prof = cls.energy_profile(p)
        e_wh = sum(h * w for _, h, w in prof)
        usable = 0.9 * p["battery_kwh"] * 1000
        return {
            "housings": {"electronics": he, "battery": hb},
            "mass_in_air_kg": m_total, "dry_mass_kg": m_dry, "foam_L": V_f * 1000, "foam_space_L": foam_avail * 1000,
            "foam_fits": V_f <= foam_avail, "displacement_L": (v_dry + V_f) * 1000, "net_buoyancy_kg": net_up,
            "payload_in_water_kg": V_p * (RHO_SW - FOAM_RHO), "ballast_bare_kg": ballast_kg,
            "items": items, "thrust_N": th, "v_max_m_s": v_max,
            "station_keeping_current_m_s": v_max["y"],                       # the worst axis: the side
            "dive_energy_Wh": e_wh, "dive_hours": sum(h for _, h, _ in prof), "battery_usable_Wh": usable,
            "battery_margin": usable / e_wh - 1, "profile": prof,
            "hotel_W": sum(w for _, _, w, _ in PARTS.values()),
        }

    # ------------------------------------------------------------------------------------------- CAD
    def _box(self, lx, ly, lz, c):
        return cq.Workplane("XY").box(lx, ly, lz).translate(c)

    def _frame(self, p):
        L, W, b, z0 = p["length"], p["width"], p["frame_bar"], p["skid_height"]
        H = p["height"] - p["foam_height"] - z0
        out = None
        for s in (1, -1):                                                      # two side plates with windows
            plate = self._box(L, b, H, (0, s * (W / 2 - b / 2), z0 + H / 2))
            for k in range(3):
                plate = plate.cut(self._box(L * 0.22, b * 2, H * 0.55, (L * (-0.3 + 0.3 * k), s * (W / 2 - b / 2), z0 + H / 2)))
            out = plate if out is None else out.union(plate)
        for x in (L / 2 - b / 2, -L / 2 + b / 2, 0.0):                         # cross members, bottom and top
            out = out.union(self._box(b, W - 2 * b, b, (x, 0, z0 + b / 2)))
            out = out.union(self._box(b, W - 2 * b, b, (x, 0, z0 + H - b / 2)))
        return out

    def _fairing(self, p):
        L, W, Hf, top = p["length"], p["width"], p["foam_height"], p["height"]
        shell = (cq.Workplane("XY").box(L + 20, W + 20, Hf + 20).edges("|X or |Y").chamfer(min(60.0, Hf / 2))
                 .translate((0, 0, top - Hf / 2)))
        return shell.cut(self._box(L - 10, W - 10, Hf, (0, 0, top - Hf / 2 - 10)))

    def _foam(self, p):
        L, W, b, Hf, top = p["length"], p["width"], p["frame_bar"], p["foam_height"], p["height"]
        return self._box(L - 2 * b - 20, W - 2 * b - 20, Hf - 25, (0, 0, top - Hf / 2 - 5))

    def _housing(self, p, which):
        r = p["housing_od"] / 2
        Lh = p["electronics_length"] if which == "electronics" else p["battery_length"]
        y = 0.22 * p["width"] * (1 if which == "electronics" else -1)
        z = p["skid_height"] + p["frame_bar"] + r + 10
        return cq.Workplane("YZ").center(y, z).circle(r).extrude(Lh / 2, both=True)

    def _thruster(self, p, at=(0, 0, 0), axis=(1, 0, 0)):
        d, Lt = p["thruster_duct"], p["thruster_length"]
        a = cq.Vector(*axis).normalized()
        base = cq.Vector(*at) - a * (Lt / 2)
        duct = cq.Solid.makeCylinder(d / 2, Lt, base, a).cut(cq.Solid.makeCylinder(d / 2 - 6, Lt, base, a))
        hub = cq.Solid.makeCylinder(d * 0.18, Lt * 0.8, base + a * (Lt * 0.1), a)
        return cq.Workplane("XY").add(duct.fuse(hub))

    def thruster_layout(self, p):
        """(position mm, axis) of the eight thrusters: four horizontal at the corners (45 deg), four vertical."""
        L, W, z = p["length"], p["width"], p["skid_height"] + 0.45 * (p["height"] - p["skid_height"])
        d = p["thruster_duct"]
        out = []
        for sx in (1, -1):
            for sy in (1, -1):
                out.append(((sx * (L / 2 - d * 0.9), sy * (W / 2 - d * 0.9), z), (math.cos(math.radians(45)) * sx, sy * math.sin(math.radians(45)), 0.0)))
        zt = p["height"] - p["foam_height"] / 2
        for sx in (1, -1):
            for sy in (1, -1):
                out.append(((sx * L * 0.33, sy * (W / 2 - d * 0.75), zt), (0.0, 0.0, 1.0)))
        return out

    def _arm(self, p, reach, y, links, pose):
        """A chain of ``links`` link fractions; folded: tucked down under the nose, work: reaching forward and down."""
        r = p["arm_link"] / 2
        x0, z0 = p["length"] / 2 - 40, p["skid_height"] + 0.30 * (p["height"] - p["skid_height"])
        angles = [-20, -35, -55, -70] if pose == "work" else [-175, 4, -176, 4]    # folded: flat zig-zag under the nose
        pts = [cq.Vector(x0, y, z0)]
        out = cq.Workplane("XY").add(cq.Solid.makeSphere(r * 1.3, pts[0]))
        for frac, ang in zip(links, angles):
            a = math.radians(ang)
            d = cq.Vector(math.cos(a), 0, math.sin(a)) * (reach * frac)
            nxt = pts[-1] + d
            out = out.add(cq.Solid.makeCylinder(r * (1 - 0.15 * len(pts)), d.Length, pts[-1], d))
            out = out.add(cq.Solid.makeSphere(r * (1.1 - 0.15 * len(pts)), nxt))
            pts.append(nxt)
        tip = pts[-1]
        for s in (1, -1):                                                      # jaws
            jaw = cq.Vector(math.cos(math.radians(angles[-1])), s * 0.25, math.sin(math.radians(angles[-1]))) * (0.12 * reach)
            out = out.add(cq.Solid.makeCylinder(r * 0.3, jaw.Length, tip, jaw))
        return cq.Workplane("XY").add(out.vals()).combine()

    def _skid(self, p, sx=1, sy=1):
        L, W, h = p["length"], p["width"], p["skid_height"]
        x, y = sx * (L / 2 - 80), sy * (W / 2 - 40)
        post = cq.Workplane("XY").center(x, y).circle(22).extrude(h - 20).translate((0, 0, 20))
        pad = cq.Workplane("XY").center(x, y).ellipse(90, 60).extrude(20)       # compliant pad (PU), soft seabed
        return post.union(pad)

    def _tool_skid(self, p):
        """The cleaning skid: a box under the belly (its bottom 60 mm above the soles) and a rotary brush whose bristles
        stop 20 mm above them — Scampi lands on its skids, never on the tool."""
        hb = p["tool_bay_height"]
        z_bot = 60.0
        box = self._box(p["tool_bay_length"], p["tool_bay_width"], hb, (0, 0, z_bot + hb / 2))
        brush = cq.Workplane("XY").center(p["tool_bay_length"] / 2 - 80, 0).circle(70).extrude(-(z_bot - 20.0)).translate((0, 0, z_bot))
        return box.union(brush)

    def build(self, p):
        if cq is None:
            raise ImportError("Scampi's CAD needs CadQuery")
        part = p["part"]
        if part == "frame":
            return self._frame(p)
        if part == "fairing":
            return self._fairing(p)
        if part == "foam":
            return self._foam(p)
        if part == "housing":
            return self._housing(p, p["housing"])
        if part == "thruster":
            return self._thruster(p)
        if part == "work_arm":
            return self._arm(p, p["arm_reach"], -0.22 * p["width"], (0.38, 0.32, 0.18, 0.12), p["arm_pose"])
        if part == "hold_arm":
            return self._arm(p, p["hold_reach"], 0.22 * p["width"], (0.45, 0.40, 0.15), p["arm_pose"])
        if part == "skid":
            return self._skid(p)
        if part == "tool_skid":
            return self._tool_skid(p)
        body = self._frame(p).union(self._fairing(p)).union(self._foam(p))
        body = body.union(self._housing(p, "electronics")).union(self._housing(p, "battery"))
        for at, ax in self.thruster_layout(p):
            body = body.union(self._thruster(p, at, ax))
        for sx in (1, -1):
            for sy in (1, -1):
                body = body.union(self._skid(p, sx, sy))
        body = body.union(self._arm(p, p["arm_reach"], -0.22 * p["width"], (0.38, 0.32, 0.18, 0.12), p["arm_pose"]))
        body = body.union(self._arm(p, p["hold_reach"], 0.22 * p["width"], (0.45, 0.40, 0.15), p["arm_pose"]))
        if p["show_tool_skid"]:
            body = body.union(self._tool_skid(p))
        return body


if __name__ == "__main__":
    s = Scampi.sizing()
    for k in ("electronics", "battery"):
        h = s["housings"][k]
        print(f"{k:12s} housing: hoop {h['hoop_MPa']:.0f} MPa (margin {h['hoop_margin']:+.2f}), buckling {h['buckling_MPa']:.1f} MPa "
              f"vs {h['design_pressure_MPa']:.1f} (margin {h['buckling_margin']:+.2f}), {h['mass_kg']:.1f} kg, {h['volume_L']:.1f} L")
    print(f"mass in air {s['mass_in_air_kg']:.1f} kg (dry {s['dry_mass_kg']:.1f}, ballast bare {s['ballast_bare_kg']:.1f}), "
          f"foam {s['foam_L']:.0f} L of {s['foam_space_L']:.0f} L space, displacement {s['displacement_L']:.0f} L, "
          f"payload {s['payload_in_water_kg']:.1f} kg in water")
    print("thrust", {k: round(v) for k, v in s["thrust_N"].items()}, "v_max", {k: round(v, 2) for k, v in s["v_max_m_s"].items()})
    for name, h, w in s["profile"]:
        print(f"  {name:48s} {h:4.1f} h {w:6.0f} W")
    print(f"dive {s['dive_hours']:.1f} h: {s['dive_energy_Wh']:.0f} Wh vs usable {s['battery_usable_Wh']:.0f} Wh "
          f"(margin {s['battery_margin']:+.2f}); hotel {s['hotel_W']:.0f} W")
