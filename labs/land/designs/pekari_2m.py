"""Pekari-2M and the family's CAD (Dedalus/CadQuery): the Pekari Rover's chassis (``pekari_rover.PekariRover``)
with the platform's arms on the roof in a named pose and its stabilizers (Pumba: ``pumba.Pumba``).

The geometry of the arms comes from ``pekari_arms`` (lengths, sections, pedestals) and ``pekari_family`` (mounts,
stabilizer pads) so the CAD, the mass budget and the Chiron model share one description. The arm parts are simple
solids (rectangular tubes, joint housings, palm, two jaws): enough for the envelope, collisions with the deck and
the drawings — not for manufacture.

``part``: the Pekari parts (``hull``, ``track_module``, ``sprocket``, ``idler``, ``road_wheel``, ``track_link``,
``pinion``, ``basket`` — the deck) and ``robot`` (everything, arms in ``arm_pose``, stabilizers per
``stabilizers``), ``arm`` (the first mount's arm in ``arm_pose``, the shoulder at the origin), ``upper_arm`` /
``forearm`` (tubes in their own frames: the inboard joint at the origin, along +x).
"""
import math

try:
    import cadquery as cq
except ImportError:                     # the parameters and the platform need no CAD kernel
    cq = None
from vegeta.dedalus import Parameter

import pekari_arms as pa
from pekari_rover import PekariRover

PARTS = ("robot", "hull", "track_module", "sprocket", "idler", "road_wheel", "track_link", "pinion", "basket", "arm",
         "upper_arm", "forearm")


def chassis_parameters(chassis: dict, platform_name: str) -> list:
    """Pekari's parameters (without ``part``) with the platform's chassis as defaults."""
    from dataclasses import replace

    out = []
    for q in PekariRover.parameters:
        if q.name == "part":
            continue
        if q.name in chassis and q.default != chassis[q.name]:
            desc = "the deck (the Pekari basket)" if q.name.startswith("basket") else q.description
            q = replace(q, default=chassis[q.name], description=desc or f"{platform_name} chassis")
        out.append(q)
    return out


def family_parameters() -> list:
    return [Parameter("arm_pose", "folded", choices=("folded", "deployed", "reach"), description="the arms' named pose"),
            Parameter("stabilizers", "stowed", choices=("none", "stowed", "deployed"), description="the stabilizer legs")]


class _FamilyCAD(PekariRover):
    """The common CAD of the family: ``PLATFORM`` names the ``pekari_family.Platform`` (module:attribute)."""

    PLATFORM = ""

    @classmethod
    def platform(cls):
        mod, attr = cls.PLATFORM.split(":")
        import importlib

        return getattr(importlib.import_module(mod), attr)

    # ---- arm parts
    @staticmethod
    def _tube(L, section, boss_r):
        w, d = section
        body = cq.Workplane("XY").box(L, w, d).translate((L / 2, 0, 0))
        for x in (0.0, L):
            body = body.union(cq.Workplane("XZ").center(x, 0).circle(boss_r).extrude(w * 0.6, both=True))
        return body

    def _arm(self, arm: pa.ArmSpec, q: dict):
        """One arm in pose ``q`` (rad), shoulder at the origin [mm]."""
        k = 1000.0
        uw, ud = (v * k for v in arm.upper_section)
        fw, fd = (v * k for v in arm.fore_section)
        e1 = -q["shoulder"]
        e2 = -(q["shoulder"] + q["elbow"])
        e3 = -(q["shoulder"] + q["elbow"] + q["wrist_pitch"])
        Lu, Lf, Lt = arm.upper * k, arm.fore * k, arm.tool * k
        ped_h, ped_r = arm.pedestal[0] * k, arm.pedestal[1] * k
        pedestal = cq.Workplane("XY").circle(ped_r).extrude(ped_h - 0.5 * ud).translate((0, 0, -ped_h))
        upper = self._tube(Lu, (uw, ud), 0.55 * ud).rotate((0, 0, 0), (0, 1, 0), -math.degrees(e1))
        ex, ez = Lu * math.cos(e1), Lu * math.sin(e1)
        fore = self._tube(Lf, (fw, fd), 0.55 * fd).rotate((0, 0, 0), (0, 1, 0), -math.degrees(e2)).translate((ex, 0, ez))
        wx, wz = ex + Lf * math.cos(e2), ez + Lf * math.sin(e2)
        t_mod = 0.35 * Lt
        palm_len = max(Lt - arm.jaw_length * k - t_mod, 20.0)
        tool = cq.Workplane("XZ").circle(0.55 * fd).extrude(0.55 * fw, both=True)                 # wrist pitch housing
        tool = tool.union(cq.Workplane("YZ").circle(0.5 * fw).extrude(t_mod))                      # roll module
        tool = tool.union(cq.Workplane("XY").box(palm_len, 1.2 * fw, 1.2 * fd).translate((t_mod + palm_len / 2, 0, 0)))
        jl, jw, jd = arm.jaw_length * k, max(0.6 * fw, 20.0), max(0.25 * fd, 12.0)
        for s in (1, -1):
            tool = tool.union(cq.Workplane("XY").box(jl, jw, jd).translate((t_mod + palm_len + jl / 2, 0, s * (0.3 * fd + jd / 2))))
        tool = tool.rotate((0, 0, 0), (0, 1, 0), -math.degrees(e3)).translate((wx, 0, wz))
        out = pedestal.union(upper).union(fore).union(tool)
        return out.rotate((0, 0, 0), (0, 0, 1), math.degrees(q["yaw"]))

    def _stabilizer(self, s, p, deployed: bool):
        """A swing-out leg [mm, vehicle frame]: a beam from the hull side to the pad, a vertical leg, the pad."""
        k = 1000.0
        half_w = p["hull_width"] / 2
        side = 1.0 if s.y > 0 else -1.0
        z_hinge = p["ground_clearance"] + 40.0
        r = s.pad_radius * k
        if deployed:
            y_pad = s.y * k
            z_pad = r / 4
        else:
            y_pad = side * (half_w + 60.0)
            z_pad = z_hinge - 40.0
        beam_len = abs(y_pad) - half_w
        beam = cq.Workplane("XY").box(80.0, beam_len, 60.0).translate((s.x * k, side * (half_w + beam_len / 2), z_hinge + 60.0))
        leg = cq.Workplane("XY").box(60.0, 60.0, z_hinge + 60.0 - z_pad).translate((s.x * k, y_pad, (z_hinge + 60.0 + z_pad) / 2))
        pad = cq.Workplane("XY").circle(r).extrude(r / 2).translate((s.x * k, y_pad, z_pad - r / 4))
        return beam.union(leg).union(pad)

    def build(self, p):
        part = p["part"]
        pl = self.platform()
        if part in ("arm", "upper_arm", "forearm"):
            arm = pl.mounts[0].arm
            if part == "upper_arm":
                return self._tube(arm.upper * 1000, tuple(v * 1000 for v in arm.upper_section), 0.55 * arm.upper_section[1] * 1000)
            if part == "forearm":
                return self._tube(arm.fore * 1000, tuple(v * 1000 for v in arm.fore_section), 0.55 * arm.fore_section[1] * 1000)
            return self._arm(arm, pa.pose(arm, p["arm_pose"]) | {"wrist_yaw": 0.0})
        if part != "robot":
            return PekariRover.build(self, {**p, "part": "rover" if part == "robot" else part})
        rover = PekariRover.build(self, {**p, "part": "rover"})
        roof = p["ground_clearance"] + p["hull_height"]
        for m in pl.mounts:
            q = pa.pose(m.arm, p["arm_pose"]) | {"wrist_yaw": 0.0}
            rover = rover.union(self._arm(m.arm, q).translate((m.x * 1000, m.y * 1000, roof + m.arm.pedestal[0] * 1000)))
        if p["stabilizers"] != "none":
            for s in pl.stabilizers:
                rover = rover.union(self._stabilizer(s, p, p["stabilizers"] == "deployed"))
        return rover


def _make(name, platform, chassis, doc):
    cls = type(name, (_FamilyCAD,), {
        "__doc__": doc, "PLATFORM": platform,
        "parameters": [Parameter("part", "robot", choices=PARTS, description="what to build")]
                      + chassis_parameters(chassis, name) + family_parameters()})
    return cls


from pekari_2m_robot import CHASSIS as _P2M_CHASSIS  # noqa: E402

Pekari2M = _make("Pekari2M", "pekari_2m_robot:PEKARI_2M", _P2M_CHASSIS,
                 "Pekari-2M: the Pekari chassis one size up with two light 5-DOF arms and the deck between them.")
