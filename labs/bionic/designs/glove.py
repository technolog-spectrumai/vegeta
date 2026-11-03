"""The data glove (``docs/bionic.md``, P3): a five-finger glove shaped like the wearer's hand, with the sensors of
``vegeta.bionic.glove.GloveSpec`` on it. Every part is placed from the bionic hand model (``vegeta.bionic.hand``: the
wearer's segment lengths from their hand length and breadth, any pose), so one design fits the 5th to the 95th
percentile hand, left or right.

Parts (``part``):

* ``fabric`` — the glove: a palm shell, five fingers as tapered phalanx tubes with rounded tips, the thumb's
  metacarpal over the thenar (stretch fabric; drawn as its outer surface);
* ``rings`` — printed split rings (TPU) round the proximal and middle phalanges and the thumb's proximal phalanx: they
  locate the sensor pods on the bone, not on the sliding fabric;
* ``sensors`` — one Hall sensor pod (PA12) over each sensed joint, dorsal; ``magnets`` — the Ø6 x 2 mm NdFeB disc
  for each, on the distal segment's ring next to the pod (``GloveSpec.default``: 14 joints);
* ``plate`` — the back-of-hand plate (PA12) with the IMU in its centre, the microcontroller board, the battery and the
  vibration motor (``haptics.VibrationMotor``) in pockets;
* ``cuff`` — the wrist cuff with its strap slot;
* ``cables`` — the flat cable channels from every pod back to the plate;
* ``glove`` — everything.

x along the middle finger, y towards the thumb (right hand), z out of the back of the hand; the wrist at the origin;
millimetres.
"""
from __future__ import annotations

import math

try:
    import cadquery as cq
except ImportError:
    cq = None
import numpy as np
from vegeta.dedalus import Design, Parameter

from vegeta.bionic.glove import GloveSpec
from vegeta.bionic.hand import FINGERS, HandPose, HandSize, finger_chains, interpolate, key_pose

#: Finger radius at the base of each phalanx [mm] for a 190 mm hand (scaled with the breadth): proximal, middle, distal.
FINGER_R = {"thumb": (11.0, 10.0, 9.0), "index": (9.6, 8.6, 7.6), "middle": (9.8, 8.8, 7.8),
            "ring": (9.2, 8.3, 7.4), "little": (8.0, 7.2, 6.4)}
#: The joint each sensed joint's pod sits over: (finger, chain index of the joint).
POD_AT = {"thumb_cmc_flex": ("thumb", 0), "thumb_cmc_abd": ("thumb", 0), "thumb_mcp": ("thumb", 1), "thumb_ip": ("thumb", 2),
          "index_mcp": ("index", 1), "index_abd": ("index", 1), "index_pip": ("index", 2),
          "middle_mcp": ("middle", 1), "middle_pip": ("middle", 2), "ring_mcp": ("ring", 1), "ring_pip": ("ring", 2),
          "little_mcp": ("little", 1), "little_abd": ("little", 1), "little_pip": ("little", 2)}
#: Bought parts [mm]: (length, width, height), mass [g].
BOARD = ((38.0, 26.0, 6.0), 6.0)          # ESP32-S3 class module on a carrier (chosen in P3, docs/bionic.md)
BATTERY = ((40.0, 28.0, 6.0), 18.0)       # 1S LiPo ~600 mAh
IMU = ((14.0, 14.0, 4.0), 1.5)            # 9-axis IMU breakout
MOTOR = ((10.0, 10.0, 3.0), 1.0)          # coin vibration motor (ERM)
HALL = ((9.0, 7.0, 4.0), 0.3)             # each pod: a ratiometric Hall sensor on a flex tail
MAGNET = (6.0, 2.0)                       # Ø x thickness


def pose_named(name: str) -> HandPose:
    """'flat', 'relaxed' (the hand at rest: slightly curled), or any ``key_pose`` name."""
    if name == "relaxed":
        return interpolate(key_pose("flat"), key_pose("fist"), 0.22)
    return key_pose(name)


class Glove(Design):
    parameters = [
        Parameter("part", "glove", choices=("glove", "fabric", "rings", "sensors", "magnets", "plate", "cuff", "cables"),
                  description="what to build"),
        Parameter("hand_length", 190.0, "mm", min=150, max=220, description="wrist crease to the middle fingertip"),
        Parameter("hand_breadth", 87.0, "mm", min=65, max=100, description="across the MCP heads"),
        Parameter("side", "R", choices=("R", "L")),
        Parameter("pose", "relaxed", choices=("flat", "relaxed", "fist", "pinch", "point", "spread")),
        Parameter("palm_thickness", 28.0, "mm", min=15, description="the hand's thickness at the palm (dorsum to palm)"),
        Parameter("fabric", 1.5, "mm", min=0.5, description="the glove's thickness over the skin"),
        Parameter("ring_width", 6.0, "mm", min=2),
        Parameter("ring_wall", 1.6, "mm", min=0.8),
        Parameter("plate_thickness", 2.4, "mm", min=1.0),
        Parameter("cuff_length", 45.0, "mm", min=15),
    ]

    # ------------------------------------------------------------------------------------------- the hand
    def _hand(self, p):
        size = HandSize(p["hand_length"] / 1000, p["hand_breadth"] / 1000, p["side"])
        chains = {f: [np.asarray(q) * 1000 for q in pts] for f, pts in finger_chains(pose_named(p["pose"]), size).items()}
        k = p["hand_breadth"] / 87.0
        radii = {f: tuple(r * k for r in FINGER_R[f]) for f in FINGERS}
        return size, chains, radii

    @staticmethod
    def _seg(a, b, ra, rb=None):
        """A tapered round segment from a (radius ra) to b (radius rb), with a sphere at b."""
        a, b = cq.Vector(*a), cq.Vector(*b)
        rb = ra if rb is None else rb
        d = b - a
        cone = cq.Solid.makeCone(ra, rb, d.Length, a, d)
        return cq.Workplane("XY").add(cone).union(cq.Workplane("XY").add(cq.Solid.makeSphere(rb, b)))

    def _dorsal(self, a, b, z_off):
        """A point over the segment a-b's middle, ``z_off`` above it (on the back of the finger)."""
        return (np.asarray(a) + np.asarray(b)) / 2 + np.array([0.0, 0.0, z_off])

    # ------------------------------------------------------------------------------------------- the parts
    def _fabric(self, p):
        size, ch, R = self._hand(p)
        t = p["palm_thickness"] / 2 + p["fabric"]
        sgn = 1 if p["side"] == "R" else -1
        # the palm: a rounded slab from the wrist to the MCP heads, wider at the knuckles
        mcp_x = float(np.mean([ch[f][1][0] for f in FINGERS[1:]]))
        bw = p["hand_breadth"] + 2 * p["fabric"]
        palm = (cq.Workplane("XY").polyline([(0, -0.40 * bw), (0, 0.40 * bw), (mcp_x * 0.98, 0.52 * bw), (mcp_x * 0.98, -0.52 * bw)])
                .close().extrude(t, both=True).edges("|Z").fillet(12.0).edges().fillet(min(9.0, t * 0.6)))
        palm = palm.translate((0, sgn * 0.03 * bw, 0))
        body = palm
        for f in FINGERS:
            pts = ch[f]
            r = [x + p["fabric"] for x in R[f]]
            if f == "thumb":
                # the thenar: the thumb's metacarpal inside the palm's edge, then its two phalanges
                body = body.union(self._seg(pts[0], pts[1], r[0] * 1.35, r[0]))
                body = body.union(self._seg(pts[1], pts[2], r[0], r[1]))
                body = body.union(self._seg(pts[2], pts[3], r[1], r[2] * 0.9))
            else:
                body = body.union(self._seg(pts[1], pts[2], r[0], r[1]))
                body = body.union(self._seg(pts[2], pts[3], r[1], r[2]))
                body = body.union(self._seg(pts[3], pts[4], r[2], r[2] * 0.85))
        return body

    def _ring(self, a, b, r_in, width, wall):
        """A ring round the segment a-b at its middle (axis along the segment)."""
        a, b = np.asarray(a), np.asarray(b)
        u = (b - a) / np.linalg.norm(b - a)
        c = (a + b) / 2 - u * width / 2
        o = cq.Solid.makeCylinder(r_in + wall, width, cq.Vector(*c), cq.Vector(*u))
        i = cq.Solid.makeCylinder(r_in, width, cq.Vector(*c), cq.Vector(*u))
        return cq.Workplane("XY").add(o.cut(i))

    def _rings(self, p):
        size, ch, R = self._hand(p)
        out = None
        for f in FINGERS:
            pts = ch[f]
            segs = [(pts[1], pts[2], R[f][0])] if f == "thumb" else [(pts[1], pts[2], R[f][0]), (pts[2], pts[3], R[f][1])]
            for a, b, r in segs:
                ring = self._ring(a, b, r + p["fabric"], p["ring_width"], p["ring_wall"])
                out = ring if out is None else out.union(ring)
        return out

    def _pod_point(self, p, joint, ch, R):
        f, k = POD_AT[joint]
        pts = ch[f]
        j = pts[k]
        r = R[f][0] + p["fabric"] + p["ring_wall"]
        if joint.endswith("_abd"):                                    # abduction pods sit on the knuckle's side
            side = np.cross(pts[k + 1] - pts[k], np.array([0, 0, 1.0]))
            side = side / np.linalg.norm(side) * (1 if joint.startswith("index") else -1) * (1 if p["side"] == "R" else -1)
            return j + side * (r + 2.0) + np.array([0, 0, 3.0])
        if f == "thumb" and k == 0:
            off = 6.0 if joint == "thumb_cmc_flex" else -6.0
            return j + np.array([off, 0.0, p["palm_thickness"] / 2 + p["fabric"] + 2.0])
        return j + np.array([0.0, 0.0, r + 2.0])

    def _sensors(self, p):
        size, ch, R = self._hand(p)
        out = None
        (lx, ly, lz), _ = HALL
        for j in GloveSpec.default(p["side"]).sensed:
            c = self._pod_point(p, j, ch, R)
            pod = cq.Workplane("XY").box(lx + 3, ly + 3, lz + 2).edges("|Z").fillet(2.0).translate(tuple(c))
            out = pod if out is None else out.union(pod)
        return out

    def _magnets(self, p):
        size, ch, R = self._hand(p)
        out = None
        d, t = MAGNET
        for j in GloveSpec.default(p["side"]).sensed:
            f, k = POD_AT[j]
            pts = ch[f]
            nxt = pts[min(k + 1, len(pts) - 1)]
            u = (nxt - pts[k]) / np.linalg.norm(nxt - pts[k])
            c = self._pod_point(p, j, ch, R) + u * 8.0                     # on the distal side of the joint
            m = cq.Workplane("XY").circle(d / 2).extrude(t).translate(tuple(c))
            out = m if out is None else out.union(m)
        return out

    def _plate(self, p):
        size, ch, R = self._hand(p)
        z0 = p["palm_thickness"] / 2 + p["fabric"]
        mcp_x = float(np.mean([ch[f][1][0] for f in FINGERS[1:]]))
        L, W = mcp_x * 0.72, p["hand_breadth"] * 0.62
        cx = mcp_x * 0.46
        plate = (cq.Workplane("XY").box(L, W, p["plate_thickness"]).edges("|Z").fillet(10.0)
                 .translate((cx, 0, z0 + p["plate_thickness"] / 2)))
        top = z0 + p["plate_thickness"]
        parts = plate
        for (dims, _), (dx, dy) in ((BOARD, (-0.18 * L, 0.18 * W)), (BATTERY, (-0.18 * L, -0.20 * W)),
                                    (IMU, (0.22 * L, 0.0)), (MOTOR, (0.30 * L, 0.30 * W))):
            lx, ly, lz = dims
            housing = (cq.Workplane("XY").box(lx + 3, ly + 3, lz + 1.5).edges("|Z").fillet(2.0)
                       .translate((cx + dx, dy, top + (lz + 1.5) / 2)))
            parts = parts.union(housing)
        # strap slots at the plate's sides
        for sy in (1, -1):
            parts = parts.cut(cq.Workplane("XY").box(L * 0.4, 4.0, 10.0).translate((cx - 0.1 * L, sy * (W / 2 - 4.0), z0)))
        return parts

    def _cuff(self, p):
        t = p["palm_thickness"] / 2 + p["fabric"]
        a, b = 0.36 * p["hand_breadth"] + 4.0, t + 4.0                    # the wrist's half width and half depth
        outer = cq.Workplane("YZ").ellipse(a + 2.5, b + 2.5).extrude(p["cuff_length"])
        inner = cq.Workplane("YZ").ellipse(a, b).extrude(p["cuff_length"])
        cuff = outer.cut(inner).translate((-p["cuff_length"] + 8.0, 0, 0))
        buckle = cq.Workplane("XY").box(18.0, 14.0, 4.0).translate((-p["cuff_length"] / 2 + 8.0, 0, b + 4.0))
        return cuff.union(buckle)

    def _cables(self, p):
        size, ch, R = self._hand(p)
        z0 = p["palm_thickness"] / 2 + p["fabric"] + p["plate_thickness"]
        mcp_x = float(np.mean([ch[f][1][0] for f in FINGERS[1:]]))
        hub = np.array([mcp_x * 0.46 + 0.36 * mcp_x * 0.72, 0.0, z0])
        out = None
        for j in GloveSpec.default(p["side"]).sensed:
            c = self._pod_point(p, j, ch, R)
            a, b = cq.Vector(*c), cq.Vector(*(hub + np.array([0, (c[1] - hub[1]) * 0.6, 0])))
            seg = cq.Workplane("XY").add(cq.Solid.makeCylinder(1.0, (b - a).Length, a, b - a))
            out = seg if out is None else out.union(seg)
        return out

    def build(self, p):
        if cq is None:
            raise ImportError("the glove's CAD needs CadQuery")
        parts = {"fabric": self._fabric, "rings": self._rings, "sensors": self._sensors, "magnets": self._magnets,
                 "plate": self._plate, "cuff": self._cuff, "cables": self._cables}
        if p["part"] in parts:
            return parts[p["part"]](p)
        out = None
        for name in ("fabric", "rings", "sensors", "magnets", "plate", "cuff", "cables"):
            s = parts[name](p)
            out = s if out is None else out.union(s)
        return out

    @staticmethod
    def bom(p=None) -> dict:
        """The bill of materials: counts and the bought parts' mass [g]."""
        n = len(GloveSpec.default().sensed)
        return {"Hall sensor pods": n, "NdFeB magnets Ø6x2": n, "TPU split rings": 9, "IMU (9-axis)": 1,
                "microcontroller board (ESP32-S3 class)": 1, "LiPo battery 1S ~600 mAh": 1, "coin vibration motor": 1,
                "back plate + housings (PA12)": 1, "wrist cuff (TPU) + strap": 1, "stretch fabric glove": 1,
                "bought mass g": round(n * HALL[1] + n * 0.4 + IMU[1] + BOARD[1] + BATTERY[1] + MOTOR[1], 1)}
