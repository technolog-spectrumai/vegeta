"""The fantasy humanoids as Chiron robots, for export: Talia's model (``humanoid_robot.talia``) with what the CAD
adds in the hands.

* **Ghoul** — two jaws per hand on actuated hinges at the end of the forearm (axis x: they open sideways, the
  left-side jaw to +y), ``smart servo 6 Nm`` each: 25 actuated joints.
* **Ork** — Talia's frame at the ork's numbers; the broom (an ash pole and stock, a straw head) rigid on the right
  forearm and the trash-bin lid rigid on the left, their masses the CAD volumes × the densities of
  ``fantasy_humanoid_specs.MATERIALS``. The elbows stand bent 35° (``nominal_qpos``) as in the CAD, which keeps
  the pole vertical and the broom head on the ground.
"""
from __future__ import annotations

import math

from vegeta.chiron import Geom, Joint, Link

import body_kit as bk
import fantasy_humanoid_specs as fh
import humanoid_robot as hr

__all__ = ["add_pincers", "ork", "CAD_MM3", "carried_masses", "parts_ork"]

mm = 1e-3

#: CAD volumes of the ork's carried items [mm³] (``fantasy_humanoids.broom`` / ``bin_lid`` at the defaults; the
#: test rebuilds them)
CAD_MM3 = {"broom wood": 1613449.2, "broom straw": 5095118.0, "lid": 519544.6}
MATERIAL = {"broom wood": "ash", "broom straw": "broom straw", "lid": "galvanised steel"}


def carried_masses() -> dict:
    """{item: kg} of the ork's broom (wood, straw) and lid."""
    return {k: v * 1e-9 * fh.MATERIALS[MATERIAL[k]]["density_kg_m3"] for k, v in CAD_MM3.items()}


def _link(root, name):
    for lk, _ in root.walk():
        if lk.name == name:
            return lk
    raise KeyError(name)


def add_pincers(robot, forearm_length_mm, jaw_length_mm=fh.PINCER["pincer_length"], servo_key="smart servo 6 Nm"):
    """Two jaw links per hand on ``robot`` (a ``talia`` model): hinges 75 mm past the forearm's end, 12 mm either
    side of its line, axis x; each jaw a capsule hanging along the forearm, 15 g per 100 mm of PA12-CF."""
    Lf, Lj = forearm_length_mm * mm, jaw_length_mm * mm
    lo, hi = (math.radians(a) for a in fh.PINCER["jaw_range_deg"])
    sv = bk.servo(servo_key, kp=8.0, kd=0.1)
    from ogma import actuators as act
    m_act = act.get(servo_key).mass_g / 1000.0
    for side in ("L", "R"):
        fore = _link(robot.root, f"{side}_forearm")
        for jaw, sy in (("a", 1), ("b", -1)):
            rng = (lo, hi) if sy > 0 else (-hi, -lo)                       # each jaw opens away from the other
            fore.children.append(Link(
                f"{side}_jaw_{jaw}", pos=(0.0, sy * 0.012, -Lf - 0.075),
                joints=[Joint(f"{side}_jaw_{jaw}", axis=(1, 0, 0), range=rng, tag="jaw", servo=sv)],
                geoms=[Geom(f"{side}_jaw_{jaw}", "capsule", (0.007,), fromto=(0, 0, 0, 0, 0, -Lj),
                            mass=0.15 * Lj + m_act / 2, role="link", rgba=bk.DARK)]))
    robot.notes = robot.notes.replace("21 actuated joints", "25 actuated joints (four pincer jaws)")
    return robot


def ork():
    p = {k: v for k, v in fh.ORK.items() if k in hr.P}
    r = hr.talia("Ork", p, notes="Ork: a bulky humanoid with a broom and a trash-bin lid, 21 actuated joints "
                               "(design stage).")
    P = dict(hr.P, **p)
    e = math.radians(fh.ORK["elbow_bend_deg"])
    Hp, Ht, rr = P["pelvis_height"] * mm, P["torso_height"] * mm, P["arm_diameter"] * mm / 2
    La, Lf = P["upper_arm_length"] * mm, P["forearm_length"] * mm
    hand = -Lf - 0.04                                                    # the fist on the forearm's axis
    z_hand = Hp / 2 + Ht - rr * 1.2 - La + hand * math.cos(e)             # world, pelvis at 0, elbow bent e
    z_ground = -(Hp / 2 + (P["thigh_length"] + P["shin_length"] + P["ankle_height"]) * mm)
    down = (-math.sin(e), 0.0, -math.cos(e))                              # world vertical in the bent forearm's frame
    up = tuple(-c for c in down)
    m = carried_masses()
    bl, sh = fh.ORK["bristle_length"] * mm, fh.ORK["stock_height"] * mm
    drop = z_hand - z_ground - bl - sh / 2                               # the fist to the stock's centre
    stock = tuple(c * drop for c in down)
    stock = (stock[0], 0.0, hand + stock[2])
    top = tuple(c * fh.ORK["pole_above_hand"] * mm for c in up)
    top = (top[0], 0.0, hand + top[2])
    straw = (stock[0] + down[0] * (sh / 2 + bl / 2), 0.0, stock[2] + down[2] * (sh / 2 + bl / 2))
    tilt = (math.cos(-e / 2), 0.0, math.sin(-e / 2), 0.0)                 # the broom head level in the world
    fore_r = _link(r.root, "R_forearm")
    fore_r.geoms += [
        Geom("broom_pole", "capsule", (fh.ORK["pole_diameter"] * mm / 2,), fromto=stock + top,
             mass=m["broom wood"] * 0.75, role="link", rgba=(0.6, 0.45, 0.3, 1.0)),
        Geom("broom_stock", "box", (fh.ORK["stock_depth"] * mm / 2, fh.ORK["broom_width"] * mm / 2, sh / 2), pos=stock,
             quat=tilt, mass=m["broom wood"] * 0.25, role="link", rgba=(0.6, 0.45, 0.3, 1.0)),
        Geom("broom_bristles", "box", (fh.ORK["stock_depth"] * mm * 0.7, fh.ORK["broom_width"] * mm * 0.55, bl / 2),
             pos=straw, quat=tilt, mass=m["broom straw"], role="link", rgba=(0.85, 0.75, 0.4, 1.0))]
    fore_l = _link(r.root, "L_forearm")
    fore_l.geoms.append(Geom("bin_lid", "cylinder", (fh.ORK["lid_diameter"] * mm / 2, fh.ORK["lid_dish"] * mm / 2),
                             pos=(0.0, rr + 0.004 + fh.ORK["lid_dish"] * mm / 2, -Lf * 0.45),
                             quat=(math.cos(math.pi / 4), -math.sin(math.pi / 4), 0.0, 0.0), mass=m["lid"],
                             role="link", rgba=(0.62, 0.65, 0.68, 1.0)))
    r.nominal_qpos = {"L_elbow": -e, "R_elbow": -e}
    r.sources["carried"] = "fantasy_humanoids.broom / bin_lid: CAD volumes × fantasy_humanoid_specs.MATERIALS"
    return r


def parts_ork(machine) -> dict:
    import fantasy_humanoid_motions as fm
    import motion_kit as mk
    kin, ms = fm.motions_ork()
    out = mk.block(ms, fm.base_height(kin))
    out["materials"] = {k: dict(fh.MATERIALS[k]) for k in set(MATERIAL.values())}
    out["carried"] = {k: {"material": MATERIAL[k], "volume_m3": round(v * 1e-9, 9), "mass_kg": round(carried_masses()[k], 4)}
                      for k, v in CAD_MM3.items()}
    out["tags"]["data:carried*"] = "cad"
    return out
