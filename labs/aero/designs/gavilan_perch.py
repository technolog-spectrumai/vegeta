"""GAVILAN's perch: where it waits as a camera, and charges through its feet (CAD, Dedalus; the charging budget).

GAVILAN-X stands nose-up on the pads at its four panel tips (``gavilan.GavilanX``: ``foot_pads``). The perch is a
cross of four arms on a hub that bolts to a roof, a mast or a wall bracket; at the end of each arm, under each foot, a
**cup**: a cone that funnels the pad in on landing (it forgives a landing ``funnel_top / 2`` off in any direction and
the matching yaw error), and at the bottom of the cone a **sprung contact plate**. The pads carry brass soles wired to
the aircraft's power bus through an ideal-diode bridge, so which foot lands in which cup does not matter (the aircraft
can come down turned by any multiple of 90 degrees): two diagonal cups are +, two are −.

The perch supplies a constant-current, constant-voltage charge for the 6S pack (25.2 V); the aircraft carries a small
active balancer. While it sits on the perch the perch also carries its standby load (camera, video, flight computer),
so it watches indefinitely; off the perch it watches on its pack (``budget``).

Frame: z up, the ground (the perch's base) at z = 0, the aircraft's body axis on x = y = 0; mm. ``part``: ``perch``
(everything), ``cup`` (one cup, printed), ``hub``.
"""
from __future__ import annotations

import math

import cadquery as cq

from vegeta.dedalus import Design, Parameter

import gavilan as gv

CUP_ANGLES = (45.0, 135.0, 225.0, 315.0)


def _foot_radius() -> float:
    p = gv.x_resolve()
    return gv.GavilanX.layout(p)["foot_radius"] - p["pad_width"] / 2 + 2.0      # the pad's centre, along the panel


class Perch(Design):
    """The charging perch for GAVILAN-X."""

    parameters = [
        Parameter("part", "perch", choices=("perch", "cup", "hub"), description="what to build"),
        Parameter("foot_radius", round(_foot_radius(), 1), "mm", min=100.0, description="the pads' centres from the axis (GavilanX's)"),
        Parameter("arm_width", 30.0, "mm", min=10.0, description="each arm's width"),
        Parameter("arm_height", 18.0, "mm", min=6.0, description="each arm's height"),
        Parameter("hub_diameter", 110.0, "mm", min=40.0, description="the hub that bolts down"),
        Parameter("hub_height", 24.0, "mm", min=6.0),
        Parameter("funnel_top", 110.0, "mm", min=40.0, description="the cone's mouth: the landing tolerance is half of it"),
        Parameter("funnel_bottom", 34.0, "mm", min=20.0, description="the cone's floor, where the contact plate sits"),
        Parameter("funnel_depth", 50.0, "mm", min=15.0),
        Parameter("cup_wall", 3.0, "mm", min=1.2),
        Parameter("contact_diameter", 24.0, "mm", min=8.0, description="the sprung brass contact plate in the floor"),
        Parameter("contact_rise", 5.0, "mm", min=1.0, description="the plate's spring travel above the floor"),
        Parameter("box_size", 90.0, "mm", min=40.0, description="the charger's box on the hub's underside"),
    ]

    @staticmethod
    def layout(p) -> dict:
        z_floor = p["arm_height"] + 4.0                                                  # the cups' floors stand on the arms' tips
        return {"cups": [(p["foot_radius"] * math.cos(math.radians(a)), p["foot_radius"] * math.sin(math.radians(a))) for a in CUP_ANGLES],
                "z_floor": z_floor, "z_rim": z_floor + p["funnel_depth"], "landing_tolerance": p["funnel_top"] / 2 - 9.0,
                "yaw_tolerance_deg": math.degrees(math.atan2(p["funnel_top"] / 2 - 9.0, p["foot_radius"])),
                "span": 2 * (p["foot_radius"] + p["funnel_top"] / 2) * math.cos(math.radians(45.0)) + p["funnel_top"]}

    def _cup(self, p, x=0.0, y=0.0, z=0.0):
        r0, r1, h, w = p["funnel_bottom"] / 2, p["funnel_top"] / 2, p["funnel_depth"], p["cup_wall"]
        outer = cq.Workplane("XY").add(cq.Solid.makeCone(r0 + w, r1 + w, h + w, cq.Vector(0, 0, -w)))
        inner = cq.Workplane("XY").add(cq.Solid.makeCone(r0, r1, h + 0.1, cq.Vector(0, 0, 0)))
        cup = outer.cut(inner)
        plate = cq.Workplane("XY").circle(p["contact_diameter"] / 2).extrude(2.0).translate((0, 0, p["contact_rise"]))
        spring = cq.Workplane("XY").circle(4.0).extrude(p["contact_rise"])
        return {"cup": cup.translate((x, y, z)), "contact": plate.union(spring).translate((x, y, z))}

    def parts(self, p) -> dict:
        L = self.layout(p)
        hub = cq.Workplane("XY").circle(p["hub_diameter"] / 2).extrude(p["hub_height"])
        arms = None
        for (x, y), a in zip(L["cups"], CUP_ANGLES):
            arm = cq.Workplane("XY").box(p["foot_radius"], p["arm_width"], p["arm_height"], centered=(False, True, False)) \
                .rotate((0, 0, 0), (0, 0, 1), a)
            post = cq.Workplane("XY").circle(p["funnel_bottom"] / 2 + p["cup_wall"]).extrude(L["z_floor"]).translate((x, y, 0))
            arm = arm.union(post)
            arms = arm if arms is None else arms.union(arm)
        frame = hub.union(arms)
        box = cq.Workplane("XY").box(p["box_size"], p["box_size"], 40.0).translate((0, 0, -20.0))
        cups, contacts = None, None
        for k, (x, y) in enumerate(L["cups"]):
            c = self._cup(p, x, y, L["z_floor"])
            cups = c["cup"] if cups is None else cups.union(c["cup"])
            contacts = c["contact"] if contacts is None else contacts.union(c["contact"])
        return {"frame": frame, "charger": box, "cups": cups, "contacts": contacts}

    def build(self, p):
        if p["part"] == "cup":
            c = self._cup(p)
            return c["cup"].union(c["contact"])
        if p["part"] == "hub":
            return cq.Workplane("XY").circle(p["hub_diameter"] / 2).extrude(p["hub_height"])
        return cq.Workplane("XY").add(cq.Compound.makeCompound([v for s in self.parts(p).values() for v in s.solids().vals()]))


def resolve(p=None) -> dict:
    return Perch().resolve(**dict(p or {}))


# ------------------------------------------------------------------------------------------------------ the budget
STANDBY_W = 12.0           # flight controller, receiver, GPS, camera and its tilt servo, video link, idle (gavilan_flight.AVIONICS_W, a hand estimate)
CHARGE_C = 1.0             # the pack's charge rate the perch gives (1C: a race pack's usual; hand choice)
CHARGE_EFFICIENCY = 0.88   # perch charger to the pack's cells (hand estimate)
ONBOARD_G = {"brass soles in the four pads and their wiring along the panels": 22.0, "ideal-diode bridge (4 MOSFETs) and its fuse": 6.0,
             "6S active balancer board": 14.0}
ONBOARD_MARGIN = 1.25
HUNT_RESERVE_WH = 10.0     # what the pack keeps for a hunt and the flight home (gavilan_flight.hawk_table: ~5 Wh a 100 m hunt, doubled; hand)


def onboard_kg() -> float:
    return sum(ONBOARD_G.values()) * ONBOARD_MARGIN / 1e3


def budget() -> dict:
    """Watching on the perch and off it, and the time to put a hunt back into the pack."""
    import emerillon_market as mk
    import gavilan_flight as gf
    bat = mk.part(gf.CONFIG_J.battery)
    wh = bat.cells * 3.7 * bat.ah
    usable = gf.USABLE * wh
    i_charge = CHARGE_C * bat.ah
    p_charge = i_charge * bat.cells * 4.2 / CHARGE_EFFICIENCY
    return {"pack_wh": wh, "usable_wh": usable, "standby_w": STANDBY_W,
            "watch_off_the_perch_h": (usable - HUNT_RESERVE_WH) / STANDBY_W, "watch_on_the_perch_h": float("inf"),
            "charge_current_a": i_charge, "perch_supply_w": p_charge + STANDBY_W,
            "recharge_after_a_100_m_hunt_min": 2 * 5.0 / (i_charge * bat.cells * 3.85) * 60.0,
            "full_charge_min": usable / (i_charge * bat.cells * 3.85) * 60.0, "onboard_kg": onboard_kg()}
