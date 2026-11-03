"""A kinematic human hand: 20 finger joints and the wrist's pose, per-user segment lengths, fingertip kinematics.

Joints (``JOINTS``; angles in rad, flexion positive): the thumb's CMC flexion and abduction, MCP and IP flexion; for
each finger (index, middle, ring, little) MCP flexion and abduction, PIP and DIP flexion. Ranges (``RANGES``) are
typical active ranges of motion of an adult hand (clinical tables; individual users differ, so a calibration sets the
user's own).

Frames: the hand frame has its origin at the wrist joint, x along the middle finger's metacarpal (towards the
fingers), y towards the thumb for a right hand (the little finger for a left one), z out of the back of the hand.
Fingers flex towards -z.

Segment lengths come from the hand length (wrist crease to the middle fingertip) by ratios (``RATIOS``), after
Buchholz, Armstrong & Goldstein (1992), rounded: a stated approximation, enough for retargeting and glove sizing.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import _quat as Q

FINGERS = ("thumb", "index", "middle", "ring", "little")
JOINTS = ("thumb_cmc_flex", "thumb_cmc_abd", "thumb_mcp", "thumb_ip",
          "index_mcp", "index_abd", "index_pip", "index_dip",
          "middle_mcp", "middle_abd", "middle_pip", "middle_dip",
          "ring_mcp", "ring_abd", "ring_pip", "ring_dip",
          "little_mcp", "little_abd", "little_pip", "little_dip")
_D = math.radians
RANGES = {   # active range of motion [rad]: (extension or adduction, flexion or abduction)
    "thumb_cmc_flex": (_D(-15), _D(50)), "thumb_cmc_abd": (_D(0), _D(60)), "thumb_mcp": (_D(-10), _D(60)),
    "thumb_ip": (_D(-10), _D(80)),
    **{f"{f}_mcp": (_D(-20), _D(90)) for f in FINGERS[1:]},
    **{f"{f}_abd": (_D(-20), _D(20)) for f in FINGERS[1:]},
    **{f"{f}_pip": (_D(0), _D(110)) for f in FINGERS[1:]},
    **{f"{f}_dip": (_D(-5), _D(80)) for f in FINGERS[1:]},
}
#: Segment length / hand length: (metacarpal, proximal, middle, distal); the thumb has no middle phalanx (0).
RATIOS = {"thumb": (0.251, 0.196, 0.0, 0.158), "index": (0.374, 0.265, 0.143, 0.097),
          "middle": (0.373, 0.277, 0.170, 0.108), "ring": (0.336, 0.259, 0.165, 0.107),
          "little": (0.295, 0.206, 0.117, 0.093)}
#: Finger bases across the palm: lateral offset of each metacarpal's base at the wrist and its splay [hand breadths, rad].
_BASE = {"thumb": (0.30, _D(40)), "index": (0.28, _D(6)), "middle": (0.08, _D(0)), "ring": (-0.12, _D(-6)),
         "little": (-0.30, _D(-12))}


@dataclass(frozen=True)
class HandSize:
    """A user's hand: ``length`` (wrist crease to the middle fingertip) and ``breadth`` (across the MCP heads) [m],
    ``side`` 'R' or 'L'; ``segments`` per finger in metres (from ``RATIOS`` unless given)."""

    length: float = 0.19
    breadth: float = 0.085
    side: str = "R"
    segments: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.side not in ("R", "L"):
            raise ValueError("side must be 'R' or 'L'")
        if not self.segments:
            object.__setattr__(self, "segments", {f: tuple(r * self.length for r in RATIOS[f]) for f in FINGERS})

    @classmethod
    def percentile(cls, pct: int, side: str = "R", sex: str = "male"):
        """Common sizing points (adult hand length / breadth, approximate anthropometric tables): 5th, 50th, 95th."""
        table = {"male": {5: (0.173, 0.078), 50: (0.189, 0.087), 95: (0.205, 0.095)},
                 "female": {5: (0.159, 0.069), 50: (0.174, 0.076), 95: (0.189, 0.083)}}
        L, B = table[sex][pct]
        return cls(L, B, side)


@dataclass
class HandPose:
    """Joint angles (``JOINTS`` -> rad) and the wrist's orientation (quaternion, hand to world) and position [m]."""

    joints: dict = field(default_factory=lambda: {j: 0.0 for j in JOINTS})
    wrist_quat: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    wrist_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def clamped(self) -> "HandPose":
        """The pose with every joint inside its range of motion."""
        return HandPose({j: float(np.clip(a, *RANGES[j])) for j, a in self.joints.items()}, self.wrist_quat.copy(),
                        self.wrist_pos.copy())

    def vector(self) -> np.ndarray:
        return np.array([self.joints[j] for j in JOINTS])

    @classmethod
    def from_vector(cls, v, wrist_quat=None, wrist_pos=None) -> "HandPose":
        return cls(dict(zip(JOINTS, map(float, v))), np.array([1.0, 0, 0, 0]) if wrist_quat is None else np.asarray(wrist_quat, float),
                   np.zeros(3) if wrist_pos is None else np.asarray(wrist_pos, float))


#: Key poses for calibration and scripted motions.
def key_pose(name: str) -> HandPose:
    """'flat' (all joints at 0), 'fist', 'pinch' (thumb tip on the index tip), 'point' (index straight, others in),
    'spread' (fingers abducted)."""
    j = {k: 0.0 for k in JOINTS}
    if name == "flat":
        pass
    elif name == "fist":
        for f in FINGERS[1:]:
            j.update({f"{f}_mcp": _D(85), f"{f}_pip": _D(100), f"{f}_dip": _D(65)})
        j.update(thumb_cmc_flex=_D(40), thumb_cmc_abd=_D(20), thumb_mcp=_D(50), thumb_ip=_D(60))
    elif name == "pinch":
        # the thumb angles that bring its tip to the index tip (searched over the ranges: 5 mm apart, the pads touch)
        j.update(index_mcp=_D(45), index_pip=_D(55), index_dip=_D(25), thumb_cmc_flex=_D(35), thumb_cmc_abd=_D(40),
                 thumb_mcp=_D(0), thumb_ip=_D(0))
        for f in ("middle", "ring", "little"):
            j.update({f"{f}_mcp": _D(30), f"{f}_pip": _D(40), f"{f}_dip": _D(20)})
    elif name == "point":
        for f in ("middle", "ring", "little"):
            j.update({f"{f}_mcp": _D(85), f"{f}_pip": _D(100), f"{f}_dip": _D(65)})
        j.update(thumb_cmc_flex=_D(35), thumb_mcp=_D(45), thumb_ip=_D(50))
    elif name == "spread":
        j.update(index_abd=_D(15), ring_abd=_D(-10), little_abd=_D(-18), thumb_cmc_abd=_D(55))
    else:
        raise KeyError(name)
    return HandPose(j)


def finger_chains(pose: HandPose, size: HandSize) -> dict:
    """Each finger's joint centres [m, hand frame], from its base to its tip: the thumb (CMC, MCP, IP, tip), the
    fingers (metacarpal base, MCP, PIP, DIP, tip). What a glove's rings, sensor pods and links are placed on."""
    s = 1.0 if size.side == "R" else -1.0
    out = {}
    for f in FINGERS:
        mc, pp, mp, dp = size.segments[f]
        y0, splay = _BASE[f]
        base = np.array([0.0, s * y0 * size.breadth, 0.0])
        pts = [base.copy()]
        if f == "thumb":
            flex = [pose.joints["thumb_cmc_flex"], pose.joints["thumb_mcp"], pose.joints["thumb_ip"]]
            lengths = [mc, pp, dp]
        else:
            flex = [pose.joints[f"{f}_mcp"], pose.joints[f"{f}_pip"], pose.joints[f"{f}_dip"]]
            abd = splay + pose.joints[f"{f}_abd"]
            # the metacarpal is fixed in the palm; the chain flexes from the MCP head
            base = base + np.array([mc, s * math.tan(splay) * mc, 0.0])
            pts.append(base.copy())
            lengths = [pp, mp, dp]
        p, theta = base.copy(), 0.0
        if f == "thumb":
            # the thumb: flexion swings it across the palm towards the fingers (in the palm's plane), abduction lifts
            # it out of the palm's plane (palmar abduction, towards -z): opposition is the two together
            lift = pose.joints["thumb_cmc_abd"]
            for a, L in zip(flex, lengths):
                theta += a
                yaw = splay - theta
                p = p + L * np.array([math.cos(lift) * math.cos(yaw), s * math.cos(lift) * math.sin(yaw), -math.sin(lift)])
                pts.append(p.copy())
        else:
            heading = np.array([math.cos(abd), s * math.sin(abd), 0.0])
            for a, L in zip(flex, lengths):
                theta += a
                p = p + L * (math.cos(theta) * heading + np.array([0.0, 0.0, -math.sin(theta)]))
                pts.append(p.copy())
        out[f] = pts
    return out


def fingertips(pose: HandPose, size: HandSize) -> dict:
    """Fingertip positions [m] in the hand frame (wrist at the origin; see the module) for each finger."""
    return {f: pts[-1] for f, pts in finger_chains(pose, size).items()}


def pinch_aperture(pose: HandPose, size: HandSize) -> float:
    """Thumb tip to index tip distance [m]: what a pincer's jaw opening follows."""
    t = fingertips(pose, size)
    return float(np.linalg.norm(t["thumb"] - t["index"]))


def interpolate(a: HandPose, b: HandPose, u: float) -> HandPose:
    """A pose a fraction ``u`` (0..1) of the way from ``a`` to ``b`` (joints linear, the wrist by slerp)."""
    u = float(np.clip(u, 0.0, 1.0))
    j = {k: (1 - u) * a.joints[k] + u * b.joints[k] for k in JOINTS}
    qa, qb = a.wrist_quat, b.wrist_quat
    d = float(np.dot(qa, qb))
    if d < 0:
        qb, d = -qb, -d
    if d > 0.9995:
        q = Q.normalize((1 - u) * qa + u * qb)
    else:
        th = math.acos(d)
        q = (math.sin((1 - u) * th) * qa + math.sin(u * th) * qb) / math.sin(th)
    return HandPose(j, Q.normalize(q), (1 - u) * a.wrist_pos + u * b.wrist_pos)
