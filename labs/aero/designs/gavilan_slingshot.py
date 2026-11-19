"""GAVILAN's latex slingshot: the light net thrown by rubber bands, opposed so it does not kick (CAD, Dedalus).

One carbon tube under GAVILAN-X's belly, open at both ends. Two carriages sit back to back in its middle, held together
by a hook a servo lets go. Latex bands run along the outside of the tube from a ring at each end to pins on the nearer
carriage, through slots in the tube's wall: cocked, they are stretched from the ring to the middle; released, each set
pulls its carriage to its own end, where a stop holds the carriage and the load flies on — the net cartridge forward,
a second net (or a dead mass of the same weight) aft. Equal masses, equal speeds, opposite ways: no recoil and no jet.
``opposed`` False: one carriage, one set of bands, the rear end closed — lighter, and the aircraft takes a kick.

Sized in ``labs/aero/exp/gavilan_spring_net.py`` (20 m/s, latex: about 18 g of bands for the opposed pair); the
capture tests of the light net at lower speeds decide the speed. Masses there are hand estimates.

Frame: x along the tube, aft positive (GAVILAN's), the tube's axis on y = z = 0, its front end at x = 0 (the muzzle).
``state``: ``cocked`` (carriages in the middle, bands stretched) or ``fired`` (carriages at the ends, bands short, the
cartridges just out). ``part``: ``launcher`` (everything), ``tube``, ``carriage``.
"""
from __future__ import annotations

import math

import cadquery as cq
import numpy as np

from vegeta.dedalus import Design, Parameter

BAND_ANGLES = (45.0, 135.0, 225.0, 315.0)


def _cyl(r, length, x0, y=0.0, z=0.0, r_in=0.0):
    s = cq.Workplane("YZ").circle(r).extrude(length).translate((x0, y, z))
    if r_in > 0:
        s = s.cut(cq.Workplane("YZ").circle(r_in).extrude(length + 2).translate((x0 - 1, y, z)))
    return s


def _rod(a, b, r):
    a, b = np.asarray(a, float), np.asarray(b, float)
    v = b - a
    return cq.Workplane("XY").add(cq.Solid.makeCylinder(r, float(np.linalg.norm(v)), cq.Vector(*a), cq.Vector(*v)))


class Slingshot(Design):
    """The opposed latex net slingshot."""

    parameters = [
        Parameter("part", "launcher", choices=("launcher", "tube", "carriage"), description="what to build"),
        Parameter("state", "cocked", choices=("cocked", "fired"), description="cocked: stretched; fired: the carriages at the ends"),
        Parameter("opposed", True, description="two carriages thrown apart (no recoil); False: one, forward"),
        Parameter("bore", 32.0, "mm", min=20.0, description="tube bore (the light net's cartridge is 31.8 mm)"),
        Parameter("wall", 1.5, "mm", min=0.8, description="carbon tube wall"),
        Parameter("stroke", 150.0, "mm", min=50.0, description="each carriage's travel"),
        Parameter("cartridge_length", 70.0, "mm", min=30.0, description="the net cartridge (and the second one aft)"),
        Parameter("carriage_length", 24.0, "mm", min=10.0, description="each carriage"),
        Parameter("bands", 4, "", min=2, max=8, description="latex bands on each side, round the tube"),
        Parameter("band_diameter", 6.0, "mm", min=2.0, description="latex tube outer diameter"),
        Parameter("band_offset", 9.0, "mm", min=4.0, description="the bands' line outside the tube's wall"),
        Parameter("slot_width", 5.0, "mm", min=2.0, description="the slots the carriage pins run in"),
    ]

    @staticmethod
    def layout(p) -> dict:
        r_o = p["bore"] / 2 + p["wall"]
        half = p["stroke"] + p["carriage_length"] + p["cartridge_length"] + 10.0       # one side: stroke, carriage, load, a stop
        length = 2 * half if p["opposed"] else half + 20.0
        mid = half                                                                       # the middle (the latch) from the muzzle
        cocked = p["state"] == "cocked"
        c_front = mid - p["carriage_length"] if cocked else p["cartridge_length"] + 10.0  # the front carriage's front face
        c_rear = mid if cocked else length - p["cartridge_length"] - 10.0 - p["carriage_length"]
        return {"r_o": r_o, "r_band": r_o + p["band_offset"], "length": length, "mid": mid, "front_carriage_x": c_front,
                "rear_carriage_x": c_rear, "half": half}

    def _tube(self, p):
        L = self.layout(p)
        t = _cyl(L["r_o"], L["length"], 0.0, r_in=p["bore"] / 2)
        ends = [0.0, L["length"] - 6.0] if p["opposed"] else [0.0]
        for x in ends:                                                      # the rings the bands are hooked to
            ring = _cyl(L["r_band"] + 4.0, 6.0, x, r_in=L["r_o"] - 0.1)
            t = t.union(ring)
        if not p["opposed"]:
            t = t.union(_cyl(L["r_o"], 3.0, L["length"] - 3.0))            # the closed rear end
        for a in BAND_ANGLES[: max(2, min(4, int(p["bands"])))]:            # the slots for the carriage pins
            c, s = math.cos(math.radians(a)), math.sin(math.radians(a))
            x0 = 12.0
            x1 = L["length"] - 12.0
            slot = cq.Workplane("XY").box(x1 - x0, p["slot_width"], 2 * p["wall"] + 2.0).translate(((x0 + x1) / 2, 0, L["r_o"] - p["wall"] / 2))
            t = t.cut(slot.rotate((0, 0, 0), (1, 0, 0), a - 90.0))
        return t

    def _carriage(self, p, x0, pins=True):
        L = self.layout(p)
        c = _cyl(p["bore"] / 2 - 0.3, p["carriage_length"], x0)
        if pins:
            for a in BAND_ANGLES:
                d = np.array([0.0, math.cos(math.radians(a)), math.sin(math.radians(a))])
                xm = x0 + p["carriage_length"] / 2
                c = c.union(_rod(np.array([xm, 0, 0]) + d * 4.0, np.array([xm, 0, 0]) + d * (L["r_band"] + 2.0), 2.0))
        return c

    def _cartridge(self, p, x0, length, forward=True):
        """The net cartridge: a shell, four weights in splayed pockets at its leading end (``forward``: toward -x)."""
        r = p["bore"] / 2 - 0.3
        shell = _cyl(r, length, x0)
        lead = x0 if forward else x0 + length
        for a in BAND_ANGLES:
            d = np.array([-1.0 if forward else 1.0, 0.0, 0.0]) * math.cos(math.radians(12.0)) + \
                np.array([0.0, math.cos(math.radians(a)), math.sin(math.radians(a))]) * math.sin(math.radians(12.0))
            c = np.array([lead, 0, 0]) + np.array([0.0, math.cos(math.radians(a)), math.sin(math.radians(a))]) * 8.5
            shell = shell.union(_rod(c - d * 4.0, c + d * 12.0, 4.6))
        return shell

    def _bands(self, p, x_anchor, x_pin):
        L = self.layout(p)
        out = None
        n = int(p["bands"])
        for k in range(n):
            a = math.radians(BAND_ANGLES[k % 4] + (k // 4) * 12.0)
            y, z = L["r_band"] * math.cos(a), L["r_band"] * math.sin(a)
            b = _rod((x_anchor, y, z), (x_pin, y, z), p["band_diameter"] / 2)
            out = b if out is None else out.union(b)
        return out

    def _latch(self, p):
        L = self.layout(p)
        x = L["mid"] if p["opposed"] else L["mid"] + 10.0
        servo = cq.Workplane("XY").box(23.0, 12.0, 22.0).translate((x, 0.0, -(L["r_o"] + 11.0)))
        hook = _rod((x, 0, -L["r_o"]), (x, 0, -L["r_o"] + 6.0), 2.0)
        return servo.union(hook)

    def parts(self, p) -> dict:
        L = self.layout(p)
        cocked = p["state"] == "cocked"
        front_x = L["front_carriage_x"]
        out = {"tube": self._tube(p), "latch": self._latch(p)}
        car = self._carriage(p, front_x)
        bands = self._bands(p, 3.0, front_x + p["carriage_length"] / 2)
        x_cart = front_x - p["cartridge_length"] if cocked else -p["cartridge_length"] - 40.0     # fired: the cartridge just out of the muzzle
        cart = self._cartridge(p, x_cart, p["cartridge_length"], forward=True)
        if p["opposed"]:
            rx = L["rear_carriage_x"]
            car = car.union(self._carriage(p, rx))
            bands = bands.union(self._bands(p, L["length"] - 3.0, rx + p["carriage_length"] / 2))
            xr = rx + p["carriage_length"] if cocked else L["length"] + 40.0
            cart = cart.union(self._cartridge(p, xr, p["cartridge_length"], forward=False))
        out.update({"carriages": car, "bands": bands, "cartridges": cart})
        return out

    def build(self, p):
        if p["part"] == "tube":
            return self._tube(p)
        if p["part"] == "carriage":
            return self._carriage(p, 0.0)
        # separate parts that move against each other: a compound, not one fused solid
        solids = [v for s in self.parts(p).values() for v in s.solids().vals()]
        return cq.Workplane("XY").add(cq.Compound.makeCompound(solids))


def resolve(p=None) -> dict:
    return Slingshot().resolve(**dict(p or {}))
