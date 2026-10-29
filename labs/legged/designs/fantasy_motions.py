"""The chest's motions (``motion_kit``): its lid opening to 100° over 1.5 s and closing again (the hinge is passive:
a hand or a manipulator moves it). The other fantasy objects are single rigid bodies with nothing to move."""
from __future__ import annotations

import math

import motion_kit as mk

__all__ = ["motions_chest", "parts_chest"]

OPEN_DEG = 100.0


def motions_chest(robot=None):
    import fantasy_robots as fr

    kin = mk.Kin(robot or fr.chest())
    a = math.radians(OPEN_DEG)
    out = []
    for name, f in (("open", lambda u: mk.smooth(u)), ("close", lambda u: 1 - mk.smooth(u))):
        out.append(mk.joint_cycle(kin, name, lambda t, f=f: {"lid_hinge": a * f(t / 1.5)}, period=1.5, frames=24,
                                  loop=False, kind="transition", source="fantasy_motions: the lid on its hinge",
                                  params={"open_deg": OPEN_DEG}))
    return kin, out


def parts_chest(machine) -> dict:
    _, ms = motions_chest()
    return mk.block(ms, 0.0)
