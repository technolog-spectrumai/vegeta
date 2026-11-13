"""Gyrus — the Hall thruster (``docs/hall.md``, H2): the 300 W class sized by ``vegeta.fulgur.hall`` from the
SPT-100, in CadQuery. The channel's dimensions and the exit field come from ``scale_channel``; the magnetic
circuit's sections from ``CircuitSection`` — one source of truth for the CAD and the magnetostatic solve.

Parts (``part``):

* ``channel`` — the boron-nitride annular cup: inner and outer walls, the floor the anode sits on;
* ``anode`` — the stainless anode ring at the channel's floor with the gas distributor (a plenum ring with
  holes) and its feed stub;
* ``core`` — the soft-iron back plate and centre core (one piece), with the inner pole piece on top;
* ``coil_in`` — the inner coil on the core; ``coils_out`` — the four outer coils on their cores through the plate;
* ``screen`` — the outer iron screen (yoke) and the outer front pole ring;
* ``cathode`` — a placeholder hollow cathode on a bracket at the exit plane's edge (bought);
* ``feed`` — the propellant line to the anode plenum;
* ``gyrus`` — everything.

z is the thrust axis, the exit plane at z = L, the plume leaves +z; millimetres.
"""
from __future__ import annotations

import math

try:
    import cadquery as cq
except ImportError:
    cq = None
from vegeta.dedalus import Design, Parameter

from vegeta.fulgur.hall import CircuitSection, scale_channel

BN, STEEL, IRON, COPPER, ALUMINIUM = 2.1e-3, 7.9e-3, 7.87e-3, 8.96e-3, 2.7e-3     # g/mm^3
COIL_FILL = 0.65                                                                 # copper in the winding's volume


class Gyrus(Design):
    parameters = [
        Parameter("part", "gyrus", choices=("gyrus", "channel", "anode", "core", "coil_in", "coils_out", "screen", "cathode", "feed"),
                  description="what to build"),
        Parameter("power", 300.0, "W", min=100, max=2000, description="the discharge power the channel is scaled to"),
        Parameter("v_d", 280.0, "V", min=150, max=400),
        Parameter("wall", 2.0, "mm", min=1.0, description="the channel's wall thickness"),
        Parameter("floor", 3.0, "mm", min=1.5),
        Parameter("plate_t", 5.0, "mm", min=3),
        Parameter("core_r", 8.0, "mm", min=4),
        Parameter("coil_in_w", 6.0, "mm", min=3),
        Parameter("coil_out_w", 8.0, "mm", min=4, description="the outer coils' radial depth (the section's equivalent coil)"),
        Parameter("outer_coils", 4, "", min=2, max=8),
        Parameter("outer_core_r", 4.0, "mm", min=2),
        Parameter("yoke_t", 4.0, "mm", min=2),
        Parameter("pole_t", 4.0, "mm", min=2),
        Parameter("anode_t", 2.0, "mm", min=1),
        Parameter("anode_h", 6.0, "mm", min=4),
    ]

    @staticmethod
    def section(p) -> CircuitSection:
        ch = scale_channel(p["power"], p["v_d"])
        return CircuitSection(ch, wall=p["wall"] * 1e-3, plate_t=p["plate_t"] * 1e-3, core_r=p["core_r"] * 1e-3,
                              coil_in_w=p["coil_in_w"] * 1e-3, coil_out_w=p["coil_out_w"] * 1e-3, yoke_t=p["yoke_t"] * 1e-3,
                              pole_t=p["pole_t"] * 1e-3)

    @staticmethod
    def _mm(sec):
        """The section's regions in mm: {name: (r1, r2, z1, z2)}."""
        return {n: (r1 * 1e3, r2 * 1e3, z1 * 1e3, z2 * 1e3) for n, r1, r2, z1, z2, _ in sec.regions()}

    @staticmethod
    def _annulus(r1, r2, z1, z2):
        wp = cq.Workplane("XY").workplane(offset=z1).circle(r2)
        if r1 > 0:
            wp = wp.circle(r1)
        return wp.extrude(z2 - z1)

    # ------------------------------------------------------------------------------------------- parts
    def _channel(self, p):
        sec = self.section(p); c = sec.ch; L = c.L * 1e3
        r_in, r_out, w = c.r_in * 1e3, c.r_out * 1e3, p["wall"]
        cup = self._annulus(r_in - w, r_out + w, -p["floor"], L)
        return cup.cut(self._annulus(r_in, r_out, 0.0, L + 1))

    def _anode(self, p):
        sec = self.section(p); c = sec.ch
        r_in, r_out = c.r_in * 1e3 + 0.3, c.r_out * 1e3 - 0.3
        ring = self._annulus(r_in, r_out, 0.0, p["anode_h"])                       # the plenum ring
        ring = ring.cut(self._annulus(r_in + p["anode_t"], r_out - p["anode_t"], p["anode_t"], p["anode_h"] - p["anode_t"]))
        n = 24
        for k in range(n):                                                          # the distributor's holes, upwards
            a = 2 * math.pi * k / n
            rm = (r_in + r_out) / 2
            ring = ring.cut(cq.Workplane("XY").workplane(offset=p["anode_h"] - p["anode_t"] - 0.1).center(rm * math.cos(a), rm * math.sin(a)).circle(0.4).extrude(p["anode_t"] + 0.2))
        stub = cq.Workplane("XY").workplane(offset=-p["floor"] - 6).center((r_in + r_out) / 2, 0).circle(1.5).extrude(6 + p["anode_t"])
        return ring.union(stub)

    def _core(self, p):
        m = self._mm(self.section(p))
        plate = self._annulus(*m["plate"])
        core = self._annulus(*m["core"])
        pole = self._annulus(*m["pole_in"])
        body = plate.union(core).union(pole)
        for a in self._outer_angles(p):                                             # the outer cores' holes
            r = self._outer_r(p)
            body = body.cut(cq.Workplane("XY").workplane(offset=m["plate"][2] - 1).center(r * math.cos(a), r * math.sin(a)).circle(p["outer_core_r"] + 0.1).extrude(p["plate_t"] + 2))
        return body

    def _outer_r(self, p):
        m = self._mm(self.section(p))
        return (m["coil_out"][0] + m["coil_out"][1]) / 2

    def _outer_angles(self, p):
        return [2 * math.pi * k / p["outer_coils"] + math.pi / p["outer_coils"] for k in range(p["outer_coils"])]

    def _coil_in(self, p):
        m = self._mm(self.section(p))
        r1, r2, z1, z2 = m["coil_in"]
        return self._annulus(r1 + 0.3, r2 - 0.3, z1 + 0.3, z2 - 0.3)

    def _coils_out(self, p):
        m = self._mm(self.section(p))
        _, _, z1, z2 = m["coil_out"]
        out = None
        for a in self._outer_angles(p):
            r = self._outer_r(p)
            core = cq.Workplane("XY").workplane(offset=m["plate"][2]).center(r * math.cos(a), r * math.sin(a)).circle(p["outer_core_r"]).extrude(m["pole_out"][3] - m["plate"][2])
            coil = (cq.Workplane("XY").workplane(offset=z1 + 0.3).center(r * math.cos(a), r * math.sin(a))
                    .circle(p["outer_core_r"] + p["coil_out_w"] / 2 - 0.5).circle(p["outer_core_r"] + 0.3).extrude(z2 - z1 - 0.6))
            s = core.union(coil)
            out = s if out is None else out.union(s)
        return out

    def _screen(self, p):
        m = self._mm(self.section(p))
        yoke = self._annulus(*m["yoke"])
        pole = self._annulus(*m["pole_out"])
        body = yoke.union(pole)
        for a in self._outer_angles(p):                                             # the outer cores pass through the pole ring
            r = self._outer_r(p)
            body = body.cut(cq.Workplane("XY").workplane(offset=m["pole_out"][2] - 1).center(r * math.cos(a), r * math.sin(a)).circle(p["outer_core_r"] + 0.1).extrude(p["pole_t"] + 2))
        return body

    def _cathode(self, p):
        m = self._mm(self.section(p)); L = m["channel"][3]
        r = m["yoke"][1] + 12.0
        tube = cq.Workplane("XY").workplane(offset=L - 10).center(r, 0).circle(4.0).extrude(22.0)             # the hollow cathode body
        tip = cq.Workplane("XY").workplane(offset=L + 12).center(r, 0).circle(2.5).extrude(3.0)
        bracket = cq.Workplane("XY").workplane(offset=L - 10).center((m["yoke"][1] + r) / 2, 0).rect(r - m["yoke"][1] + 2, 8.0).extrude(3.0)
        return tube.union(tip).union(bracket)

    def _feed(self, p):
        sec = self.section(p); c = sec.ch
        rm = (c.r_in + c.r_out) / 2 * 1e3
        z0 = -p["floor"] - 6
        line = cq.Workplane("XY").workplane(offset=z0 - 20).center(rm, 0).circle(1.6).circle(1.1).extrude(20.0)
        return line

    # ------------------------------------------------------------------------------------------- build / masses
    def build(self, p):
        parts = {"channel": self._channel, "anode": self._anode, "core": self._core, "coil_in": self._coil_in,
                 "coils_out": self._coils_out, "screen": self._screen, "cathode": self._cathode, "feed": self._feed}
        if p["part"] in parts:
            return parts[p["part"]](p)
        out = None
        for name in parts:
            s = parts[name](p)
            out = s if out is None else out.add(s)
        return out

    def masses(self, p) -> dict:
        vol = lambda name: sum(s.Volume() for s in self.build({**p, "part": name}).solids().vals())
        m = {"channel (BN)": vol("channel") * BN, "anode (steel)": vol("anode") * STEEL, "core + plate (iron)": vol("core") * IRON,
             "inner coil (Cu)": vol("coil_in") * COPPER * COIL_FILL, "outer coils + cores": vol("coils_out") * (COPPER * COIL_FILL * 0.6 + IRON * 0.4),
             "screen + pole (iron)": vol("screen") * IRON, "cathode (placeholder)": 60.0, "feed (steel)": vol("feed") * STEEL}
        m["total"] = sum(m.values())
        return {k: round(v, 1) for k, v in m.items()}
