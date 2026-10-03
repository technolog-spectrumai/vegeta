"""Promoted from ``notebooks/designs/onager_manus.py`` as it was proven there; the notebook copy may move on."""
import math
from dataclasses import replace

import cadquery as cq
from vegeta.dedalus import Parameter

from .onager import OnagerSentinel


def _sentinel_params():
    """The Sentinel's parameters for the Manus: no sensor turret (the arms take the roof); the mast stays."""
    out = []
    for p in OnagerSentinel.parameters:
        if p.name == "part":
            continue
        if p.name == "turret_length":
            p = replace(p, default=0.0, description="no turret on the Manus: the arms take the roof")
        if p.name == "turret_height":
            p = replace(p, default=0.0, description="no turret: the mast stands on the roof")
        out.append(p)
    return out


class OnagerManus(OnagerSentinel):
    """Onager Manus: the Sentinel's wheel-leg chassis with two manipulator arms on the front of the roof, each
    ending in a pincer — two jaws on a common pivot driven by a ball-screw, with a hardened cutter notch near the
    pivot (it cuts wire) and serrated grip faces towards the tips (it lifts a log). Per arm: shoulder yaw (a
    pedestal on the roof), shoulder pitch, elbow pitch, wrist pitch, the jaws. x forward, y left, z up, ground at
    z = 0; arm angles: ``arm_shoulder_deg`` the upper arm above horizontal, ``arm_elbow_deg`` the forearm below
    horizontal, the pincer in line with the forearm.

    ``part`` adds the arm parts to the Sentinel's (each in its own frame: the joint bore at the origin, the link
    along +x, the bore axis along y): ``upper_arm``, ``forearm``, ``jaw`` (one finger, its pivot bore at the origin,
    the inner — cutting and gripping — edge on z = 0, the finger above it, the hooked tip below it)."""

    parameters = [Parameter("part", "robot", choices=("robot", "hull", "upper_leg", "lower_leg", "wheel", "upper_arm",
                                                      "forearm", "jaw"), description="what to build")] + _sentinel_params() + [
        Parameter("arm_x", 800.0, "mm", min=0, description="shoulder pedestals ahead of the hull centre"),
        Parameter("arm_y", 280.0, "mm", min=50, description="shoulder pedestals either side of the centre line"),
        Parameter("pedestal_height", 120.0, "mm", min=20, description="shoulder pitch axis above the roof"),
        Parameter("pedestal_diameter", 180.0, "mm", min=20),
        Parameter("upper_arm_length", 600.0, "mm", min=100, description="shoulder pitch to elbow"),
        Parameter("upper_arm_width", 60.0, "mm", min=10, description="rectangular tube, across (y)"),
        Parameter("upper_arm_depth", 80.0, "mm", min=10, description="rectangular tube, bending depth (z)"),
        Parameter("forearm_length", 550.0, "mm", min=100, description="elbow to wrist pitch"),
        Parameter("forearm_width", 50.0, "mm", min=10),
        Parameter("forearm_depth", 70.0, "mm", min=10),
        Parameter("arm_wall", 4.0, "mm", min=1, description="wall of the arm tubes (Al 6082-T6)"),
        Parameter("arm_boss_diameter", 110.0, "mm", min=20, description="joint module housings at the arm joints"),
        Parameter("arm_pin_diameter", 30.0, "mm", min=5, description="arm joint bores"),
        Parameter("palm_length", 100.0, "mm", min=20, description="wrist pitch to the jaw pivot"),
        Parameter("jaw_length", 220.0, "mm", min=50, description="jaw pivot to the tip"),
        Parameter("jaw_thickness", 14.0, "mm", min=3, description="finger plate thickness (y), tool steel"),
        Parameter("jaw_depth", 45.0, "mm", min=10, description="finger depth at the pivot (tapers to 45 % at the tip)"),
        Parameter("jaw_pin_diameter", 16.0, "mm", min=4),
        Parameter("notch_x", 40.0, "mm", min=10, description="cutter notch from the pivot"),
        Parameter("notch_depth", 6.0, "mm", min=1, description="depth of the V cutter notch in the inner edge"),
        Parameter("hook_length", 40.0, "mm", min=0, description="hooked tip: how far it turns in past the inner edge"),
        Parameter("hook_width", 25.0, "mm", min=5, description="hooked tip: its extent along the jaw"),
        Parameter("arm_shoulder_deg", 50.0, "deg", min=-90, max=120, description="upper arm above horizontal (ready pose)"),
        Parameter("arm_elbow_deg", 40.0, "deg", min=-90, max=150, description="forearm below horizontal (ready pose)"),
        Parameter("jaw_open_deg", 20.0, "deg", min=0, max=60, description="each jaw's opening from closed"),
    ]

    # ---- derived geometry shared by the notebook and the Chiron robot
    @staticmethod
    def shoulder_point(p):
        """Shoulder pitch axis of the left arm (x, y, z) [mm] in the world, standing; the right arm mirrors y."""
        roof = p["hull_bottom"] + p["hull_height"]
        return (p["arm_x"], p["arm_y"], roof + p["pedestal_height"])

    @staticmethod
    def reach(p):
        """Shoulder pitch axis to the jaw tips, arm straight [mm]."""
        return p["upper_arm_length"] + p["forearm_length"] + p["palm_length"] + p["jaw_length"]

    # ---- parts
    def _tube(self, L, w, d, wall, boss_d, pin_d):
        tube = cq.Workplane("XY").box(L, w, d).translate((L / 2, 0, 0))
        tube = tube.cut(cq.Workplane("XY").box(L - boss_d, w - 2 * wall, d - 2 * wall).translate((L / 2, 0, 0)))
        bosses = (cq.Workplane("XZ").circle(boss_d / 2).extrude(w / 2, both=True)
                  .union(cq.Workplane("XZ").center(L, 0).circle(boss_d / 2).extrude(w / 2, both=True)))
        bores = (cq.Workplane("XZ").circle(pin_d / 2).extrude(w, both=True)
                 .union(cq.Workplane("XZ").center(L, 0).circle(pin_d / 2).extrude(w, both=True)))
        return tube.union(bosses).cut(bores)

    def _upper_arm(self, p):
        return self._tube(p["upper_arm_length"], p["upper_arm_width"], p["upper_arm_depth"], p["arm_wall"],
                          p["arm_boss_diameter"], p["arm_pin_diameter"])

    def _forearm(self, p):
        return self._tube(p["forearm_length"], p["forearm_width"], p["forearm_depth"], p["arm_wall"],
                          p["arm_boss_diameter"] * 0.85, p["arm_pin_diameter"])

    def _jaw(self, p):
        """One finger: inner edge on z = 0 from the pivot to the tip, the back tapering; a V notch at notch_x; the
        tip hooked inwards (below z = 0) by ``hook_length`` — straight jaws closing on a round log from above wedge
        it down out of the grip; the hooks close under its middle."""
        L, t, d = p["jaw_length"], p["jaw_thickness"], p["jaw_depth"]
        r = d / 2 + 4.0
        profile = [(0.0, 0.0), (L, 0.0), (L, d * 0.45), (0.0, d)]
        finger = cq.Workplane("XZ").polyline(profile).close().extrude(t / 2, both=True)
        boss = cq.Workplane("XZ").center(0, d / 2).circle(r).extrude(t / 2, both=True)
        finger = finger.union(boss)
        nx, nd = p["notch_x"], p["notch_depth"]
        notch = cq.Workplane("XZ").polyline([(nx - nd, -1.0), (nx + nd, -1.0), (nx, nd)]).close().extrude(t, both=True)
        bore = cq.Workplane("XZ").center(0, d / 2).circle(p["jaw_pin_diameter"] / 2).extrude(t, both=True)
        if p["hook_length"] > 0:
            hw, hl = p["hook_width"], p["hook_length"]
            hook = cq.Workplane("XY").box(hw, t, hl + d * 0.45).translate((L - hw / 2, 0, (d * 0.45 - hl) / 2))
            finger = finger.union(hook)
        return finger.cut(notch).cut(bore)

    def _arm(self, p, side):
        """One arm in the ready pose, in the world (side +1 left, −1 right)."""
        sx, sy, sz = self.shoulder_point(p)
        sy *= side
        roof = p["hull_bottom"] + p["hull_height"]
        ped = cq.Workplane("XY").circle(p["pedestal_diameter"] / 2).extrude(p["pedestal_height"]).translate((sx, sy, roof))
        a, e = p["arm_shoulder_deg"], p["arm_elbow_deg"]
        Lu, Lf = p["upper_arm_length"], p["forearm_length"]
        upper = self._upper_arm(p).rotate((0, 0, 0), (0, 1, 0), -a).translate((sx, sy, sz))
        elbow = (sx + Lu * math.cos(math.radians(a)), sy, sz + Lu * math.sin(math.radians(a)))
        fore = self._forearm(p).rotate((0, 0, 0), (0, 1, 0), e).translate(elbow)
        wrist = (elbow[0] + Lf * math.cos(math.radians(e)), sy, elbow[2] - Lf * math.sin(math.radians(e)))
        palm = (cq.Workplane("XY").box(p["palm_length"], p["jaw_thickness"] * 3, p["jaw_depth"] * 1.6)
                .translate((p["palm_length"] / 2, 0, 0)).rotate((0, 0, 0), (0, 1, 0), e).translate(wrist))
        pivot = (wrist[0] + p["palm_length"] * math.cos(math.radians(e)), sy,
                 wrist[2] - p["palm_length"] * math.sin(math.radians(e)))
        jaws = None
        for k, (dy, flip) in enumerate(((-p["jaw_thickness"], 1), (p["jaw_thickness"], -1))):
            # the upper jaw (flip 1) sits above the pincer axis, the lower one mirrored below it
            j = self._jaw(p).translate((0, 0, -p["jaw_depth"] / 2))
            if flip < 0:
                j = j.mirror("XY")
            j = j.rotate((0, 0, 0), (0, 1, 0), -flip * p["jaw_open_deg"]).translate((0, dy, 0))
            j = j.rotate((0, 0, 0), (0, 1, 0), e).translate(pivot)
            jaws = j if jaws is None else jaws.union(j)
        return ped.union(upper).union(fore).union(palm).union(jaws)

    def build(self, p):
        part = p["part"]
        if part == "upper_arm":
            return self._upper_arm(p)
        if part == "forearm":
            return self._forearm(p)
        if part == "jaw":
            return self._jaw(p)
        if p["arm_x"] + p["pedestal_diameter"] / 2 > p["hull_length"] / 2:
            raise ValueError("the shoulder pedestals must sit on the roof (arm_x + pedestal radius ≤ hull_length / 2)")
        body = super().build(dict(p, part="hull" if part == "hull" else ("robot" if part == "robot" else part)))
        if part not in ("robot", "hull"):
            return body
        roof = p["hull_bottom"] + p["hull_height"]
        for side in (1, -1):
            if part == "hull":
                sx, sy, _ = self.shoulder_point(p)
                body = body.union(cq.Workplane("XY").circle(p["pedestal_diameter"] / 2).extrude(p["pedestal_height"])
                                  .translate((sx, side * sy, roof)))
            else:
                body = body.union(self._arm(p, side))
        return body
