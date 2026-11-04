"""Halka — the ion-ring thruster (``docs/plasma.md``, P2): concentric emitter and ground rings on a printed frame,
air pulled through along the axis by the corona between them. The presets are Plasma Channel's Mark IV, 4B and 4C,
whose ring radii, convergence and gaps come from ``vegeta.fulgur.thruster`` (one source of truth for the CAD and
the physics; every dimension there is estimated from the video's screenshots until measured). ``scale`` grows the
whole thing for P4.

Parts (``part``):

* ``frame`` — the printed frame (clear resin, Form 4): the Mark IV's is a base plate with concentric grooves and
  six support posts; the 4B/4C's is an outer wall with six spokes, a central hub cone with an open bore and clip
  seats on the spokes for the ground rings;
* ``grounds`` — the ground rings: printed, then nickel- or copper-plated thick (the Mark IV: 6 x 2 mm strips; the
  4B/4C: 10 x 6 mm D-section rings — the video's "Mark 4B ground 4" print used 36.6 mL of resin, which this section
  reproduces) — the cathodes, sitting ``dz`` downstream of the emitters (the convergence);
* ``emitters`` — the bronze emitter rings: 0.2 mm strip, ``strip_h`` tall, with the serrated edge towards the
  grounds (``teeth``: ``full`` draws every tooth, ``none`` the plain strip — the physics takes the pitch and tip
  radius from the parameters either way);
* ``supports`` — the insulating standoffs that hold the emitters off the frame (the creepage path the spark
  margin uses runs along them);
* ``fasteners`` — the M3 screws and the HV terminals (bought);
* ``halka`` — everything.

z is the thrust axis, air flows +z; emitters at z = 0, grounds at z = dz; millimetres.
"""
from __future__ import annotations

import math

try:
    import cadquery as cq
except ImportError:
    cq = None
from vegeta.dedalus import Design, Parameter

from vegeta.fulgur.thruster import PRESETS

RESIN, BRONZE, COPPER, NICKEL = 1.15e-3, 8.8e-3, 8.96e-3, 8.9e-3      # g/mm^3
SCREW_G, TERMINAL_G = 0.5, 2.0
#: What differs between generations beyond the rings (the frame style, the plating, the ground section).
STYLE = {"mark4": dict(frame="plate", plating="copper", ground="strip", wires_g=15.0, plating_um=100.0),
         "mark4b": dict(frame="wall", plating="nickel", ground="dsection", wires_g=15.0, plating_um=250.0),
         "mark4c": dict(frame="wall", plating="nickel", ground="dsection", wires_g=15.0, plating_um=250.0)}


class Halka(Design):
    parameters = [
        Parameter("part", "halka", choices=("halka", "frame", "grounds", "emitters", "supports", "fasteners"),
                  description="what to build"),
        Parameter("preset", "mark4c", choices=tuple(PRESETS), description="the ring layout from vegeta.fulgur"),
        Parameter("scale", 1.0, "", min=0.5, max=4.0, description="the whole layout scaled (P4)"),
        Parameter("teeth", "none", choices=("none", "full"), description="draw the emitters' serrations"),
        Parameter("tooth_pitch", 1.0, "mm", min=0.5),
        Parameter("tooth_h", 1.5, "mm", min=0.5),
        Parameter("strip_h", 6.0, "mm", min=3, description="the emitter strip's height"),
        Parameter("strip_t", 0.2, "mm", min=0.1),
        Parameter("ground_h", 6.0, "mm", min=2, description="the grounds' axial height"),
        Parameter("ground_t", 2.0, "mm", min=0.8, description="the Mark IV's strip grounds' thickness"),
        Parameter("ground_w", 10.0, "mm", min=3, description="the 4B/4C's D-section grounds' radial width"),
        Parameter("plating_um", 0.0, "um", min=0, description="the plating's thickness on the grounds (0: the preset's)"),
        Parameter("frame_h", 45.0, "mm", min=10, description="the wall frame's height"),
        Parameter("wall", 4.0, "mm", min=1.2),
        Parameter("plate_t", 4.0, "mm", min=1.5, description="the plate frame's thickness"),
        Parameter("rim_h", 14.0, "mm", min=0, description="the plate frame's outer rim"),
        Parameter("spokes", 6, "", min=3, max=12),
        Parameter("spoke_w", 8.0, "mm", min=3),
        Parameter("spoke_t", 5.0, "mm", min=2),
        Parameter("support_d", 6.0, "mm", min=3, description="the emitter standoffs' diameter"),
    ]

    # ------------------------------------------------------------------------------------------- the layout
    @staticmethod
    def layout(p) -> dict:
        """The rings in mm from the fulgur preset, scaled: [(radius, role, z)], the hub and outer radii."""
        t = PRESETS[p["preset"]]
        k = 1000.0 * p["scale"]
        rings = [(r.radius * k, r.role, r.z * k) for r in t.rings]
        return {"rings": rings, "hub": t.hub_radius * k, "outer": t.outer_radius * k, "dz": max(z for _, _, z in rings),
                "style": STYLE[p["preset"]], "thruster": t}

    @staticmethod
    def _ring(r, t, h, z=0.0):
        return cq.Workplane("XY").workplane(offset=z).circle(r + t / 2).circle(r - t / 2).extrude(h)

    @staticmethod
    def _dsection(r, w, h, z=0.0, clearance=0.0):
        """A D-section ring: w wide (radial) x h tall, rounded on the outer (downstream) edges; centred on z."""
        ring = cq.Workplane("XY").workplane(offset=z - h / 2).circle(r + w / 2 + clearance).circle(r - w / 2 - clearance).extrude(h + 2 * clearance)
        return ring if clearance else ring.edges(">Z").fillet(h * 0.45)

    def _ground_solid(self, p, r, z, clearance=0.0):
        L = self.layout(p)
        if L["style"]["ground"] == "strip":
            return self._ring(r, p["ground_t"] + 2 * clearance, p["ground_h"] + 2 * clearance, z - p["ground_h"] / 2 - clearance)
        return self._dsection(r, p["ground_w"], p["ground_h"], z, clearance)

    def _spoke_angles(self, p):
        return [2 * math.pi * k / p["spokes"] for k in range(p["spokes"])]

    # ------------------------------------------------------------------------------------------- the parts
    def _frame(self, p):
        L = self.layout(p)
        if L["style"]["frame"] == "plate":
            # the Mark IV: an annular base plate at the grounds' plane with a groove under each ground ring
            z0 = L["dz"] - p["ground_h"] / 2 - p["plate_t"]
            plate = cq.Workplane("XY").workplane(offset=z0).circle(L["outer"] + p["wall"]).circle(L["hub"]).extrude(p["plate_t"])
            if p["rim_h"] > 0:
                plate = plate.union(self._ring(L["outer"] + p["wall"] / 2, p["wall"], p["rim_h"], z0 - p["rim_h"] + p["plate_t"]))
            for r, role, z in L["rings"]:
                if role == "ground":                   # a groove the ground ring's foot sits in
                    plate = plate.cut(self._ring(r, p["ground_t"] + 0.4, 1.5, z0 + p["plate_t"] - 1.5))
            return plate
        # the 4B/4C: an outer wall, spokes at the emitters' plane, a hub cone, clip seats under each ground ring
        wall = self._ring(L["outer"] + p["wall"] / 2, p["wall"], p["frame_h"], -p["frame_h"] / 2 + L["dz"] / 2)
        body = wall
        for a in self._spoke_angles(p):
            length = L["outer"] - L["hub"] - 2.0
            spoke = (cq.Workplane("XY").workplane(offset=-p["spoke_t"] / 2).center((L["hub"] + length / 2) * math.cos(a), (L["hub"] + length / 2) * math.sin(a))
                     .transformed(rotate=(0, 0, math.degrees(a))).rect(length, p["spoke_w"]).extrude(p["spoke_t"]))
            body = body.union(spoke)
            for r, role, z in L["rings"]:
                if role == "ground":                   # a clip seat: a block with a saddle the ring snaps into
                    seat = (cq.Workplane("XY").workplane(offset=0).center(r * math.cos(a), r * math.sin(a))
                            .rect(p["ground_w"] + 4, p["spoke_w"]).extrude(z + p["ground_h"] / 2))
                    seat = seat.cut(self._ground_solid(p, r, z, clearance=0.2))
                    body = body.union(seat)
        # the hub: a cone with an open bore, from the spokes' plane to the grounds' plane and beyond
        hub_h = L["dz"] + 2 * p["ground_h"] + 8.0
        cone = cq.Workplane("XY").add(cq.Solid.makeCone(L["hub"] + 3.0, L["hub"] + 1.2, hub_h, cq.Vector(0, 0, -p["spoke_t"] / 2), cq.Vector(0, 0, 1)))
        cone = cone.cut(cq.Workplane("XY").workplane(offset=-p["spoke_t"]).circle(L["hub"]).extrude(hub_h + 2))
        return body.union(cone)

    def _grounds(self, p):
        L = self.layout(p)
        out = None
        for r, role, z in L["rings"]:
            if role != "ground":
                continue
            g = self._ground_solid(p, r, z)
            out = g if out is None else out.add(g)
        return out

    def _emitters(self, p):
        L = self.layout(p)
        out = None
        for r, role, z in L["rings"]:
            if role != "emitter":
                continue
            strip = self._ring(r, p["strip_t"], p["strip_h"], z - p["strip_h"])        # the teeth's edge at z
            if p["teeth"] == "full":
                n = max(12, int(round(2 * math.pi * r / p["tooth_pitch"])))
                tooth = (cq.Workplane("XZ").polyline([(-p["tooth_pitch"] / 2, 0), (0, p["tooth_h"]), (p["tooth_pitch"] / 2, 0)])
                         .close().extrude(p["strip_t"] / 2, both=True).translate((r, 0, z)))
                teeth = [tooth.val().rotate(cq.Vector(0, 0, 0), cq.Vector(0, 0, 1), 360.0 * k / n) for k in range(n)]
                strip = strip.add(cq.Compound.makeCompound(teeth))
            out = strip if out is None else out.add(strip)
        return out

    def _supports(self, p):
        L = self.layout(p)
        out = None
        z_frame = L["dz"] - p["ground_h"] / 2 - p["plate_t"] if L["style"]["frame"] == "plate" else -p["spoke_t"] / 2
        for a in self._spoke_angles(p):
            for r, role, z in L["rings"]:
                if role != "emitter":
                    continue
                top, bottom = z - p["strip_h"], z_frame
                lo, hi = min(top, bottom), max(top, bottom)
                s = (cq.Workplane("XY").workplane(offset=lo).center(r * math.cos(a), r * math.sin(a))
                     .circle(p["support_d"] / 2).extrude(max(hi - lo, 2.0)))
                s = s.cut(self._ring(r, p["strip_t"] + 0.3, p["strip_h"] * 0.6, lo))           # the strip's slot
                out = s if out is None else out.union(s)
        return out

    def _fasteners(self, p):
        L = self.layout(p)
        out = None
        for a in self._spoke_angles(p):
            for r, role, z in L["rings"]:
                if role != "emitter":
                    continue
                s = (cq.Workplane("XY").workplane(offset=z - p["strip_h"] - 1.0).center((r + p["support_d"] / 2 + 1.5) * math.cos(a), (r + p["support_d"] / 2 + 1.5) * math.sin(a))
                     .circle(1.5).extrude(8.0))
                out = s if out is None else out.union(s)
        return out

    # ------------------------------------------------------------------------------------------- build / bom
    def build(self, p):
        parts = {"frame": self._frame, "grounds": self._grounds, "emitters": self._emitters, "supports": self._supports,
                 "fasteners": self._fasteners}
        if p["part"] in parts:
            return parts[p["part"]](p)
        out = None
        for name in ("frame", "grounds", "emitters", "supports", "fasteners"):
            s = parts[name](p)
            out = s if out is None else out.add(s)
        return out

    def masses(self, p) -> dict:
        """Each part's mass [g] from its volume and material; the plated grounds: resin core plus the plating shell."""
        L = self.layout(p)
        vol = lambda name: sum(s.Volume() for s in self.build({**p, "part": name}).solids().vals())
        plating = COPPER if L["style"]["plating"] == "copper" else NICKEL
        um = p["plating_um"] or L["style"]["plating_um"]
        g_area = sum(s.Area() for s in self.build({**p, "part": "grounds"}).solids().vals())
        n_e = sum(1 for _, role, _ in L["rings"] if role == "emitter")
        m = {"frame": vol("frame") * RESIN, "grounds": vol("grounds") * RESIN + g_area * um * 1e-3 * plating,
             "emitters": vol("emitters") * BRONZE, "supports": vol("supports") * RESIN,
             "fasteners": p["spokes"] * n_e * SCREW_G + 2 * TERMINAL_G, "wires": L["style"]["wires_g"]}
        m["total"] = sum(m.values())
        return {k: round(v, 1) for k, v in m.items()}

    @staticmethod
    def bom(p) -> dict:
        L = Halka.layout(p)
        n_g = sum(1 for _, role, _ in L["rings"] if role == "ground")
        n_e = len(L["rings"]) - n_g
        return {"printed frame (clear resin)": 1, f"ground rings, printed + {L['style']['plating']}-plated": n_g,
                "bronze emitter rings, serrated, brazed": n_e, "printed standoffs": p["spokes"] * n_e,
                "M3 screws": p["spokes"] * n_e, "HV terminals": 2, "video's mass g": round(L["thruster"].mass_kg * 1000)}
