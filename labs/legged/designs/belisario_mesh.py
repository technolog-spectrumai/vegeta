"""Belisario's CAD on its MuJoCo bodies, for the sims' renderer (``sims/belisario_render.py``): the skeleton of
``space_units`` - the trunk's bones (the pelvis, the vertebrae, the ribcage round the chest bay, the shoulder girdle,
the neck, the head), the femurs, tibiae and foot bones, the arms, the palms, the fingers - a joint module at every
joint, the electronics' bays in the chest and the belly (what the ribcage is for), the limbs' casing plates; each
piece tessellated once and expressed in the frame of the chiron link that carries it. No linear actuators: the
joint modules do the work. ``load(casing)`` builds the lot (~40 s of CAD) or reads the cache under
``sims/output/mesh_cache``.

The CAD is built standing (``Humanoid.build``'s pose: the knees bent ``knee_bend_deg``, the elbows
``elbow_bend_deg``, the ground at z = 0, mm); the chiron frame's links are placed from the same numbers
(``humanoid_robot.talia``, ``space_unit_robots._spine`` / ``_hands``), so a piece goes into its link by the link's
build-pose transform. The one difference: Talia's hips sit half a pelvis height below the pelvis's centre, the
CAD's at it, so the pelvis's bones are rebuilt here with the hip modules at the frame's hips.

    import belisario_mesh as bm
    mesh = bm.load("plastic")          # .pieces [(body, triangles (n, 3, 3) m, rgba)], .struts [(body, local, body, local, r_barrel, r_rod)]
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import cadquery as cq
import numpy as np

import shape_kit as sk
import space_unit_specs as su
import space_units as sun
from fantasy_humanoids import hand_frame
from humanoid import Humanoid

__all__ = ["Mesh", "build", "load", "BONE", "MODULE", "BAY", "PLATE", "DARK"]

BONE = (0.88, 0.86, 0.80, 1.0)
MODULE = (0.16, 0.17, 0.19, 1.0)
DARK = (0.22, 0.22, 0.24, 1.0)
BAY = (0.33, 0.35, 0.39, 1.0)
PLATE = (0.70, 0.60, 0.40, 1.0)
STRUT = (0.28, 0.29, 0.31, 1.0)
VERSION = 12
CACHE = Path(__file__).resolve().parents[1] / "sims" / "output" / "mesh_cache"
mm = 1e-3


@dataclass
class Mesh:
    pieces: list = field(default_factory=list)      #: [(body, triangles (n, 3, 3) [m, in the body's frame], rgba)]
    struts: list = field(default_factory=list)      #: [(body_a, local_a [m], body_b, local_b [m], r_barrel [m], r_rod [m])]

    def bodies(self):
        return sorted({p[0] for p in self.pieces} | {s[0] for s in self.struts} | {s[2] for s in self.struts})

    def triangles(self):
        return int(sum(len(p[1]) for p in self.pieces))


# ---- tessellation and frames
def _tris(shape, tol):
    out = []
    for s in shape.solids().vals():
        v, t = s.tessellate(tol, 0.5)
        v = np.array([(q.x, q.y, q.z) for q in v])
        out.append(v[np.array(t)])
    return np.concatenate(out) if out else np.zeros((0, 3, 3))


def _ry(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def frames(p):
    """{link: (origin [mm], R)} of the chiron links in the CAD's build pose: a world point w is R.T @ (w - origin)
    in the link."""
    hh = Humanoid.hip_height(p)
    Hp, Ht = p["pelvis_height"], p["torso_height"]
    z0 = hh + Hp / 2
    split = 0.42 * Ht
    a, e = p["knee_bend_deg"], p["elbow_bend_deg"]
    Lt, Ls, L1 = p["thigh_length"], p["shin_length"], p["upper_arm_length"]
    sa, ca = math.sin(math.radians(a)), math.cos(math.radians(a))
    I = np.eye(3)
    out = {"pelvis": (np.array([0.0, 0.0, hh]), I), "torso": (np.array([0.0, 0.0, z0]), I), "thorax": (np.array([0.0, 0.0, z0 + split]), I)}
    d = p["arm_diameter"]
    for side, sy in (("L", 1.0), ("R", -1.0)):
        y = sy * p["hip_spacing"] / 2
        out[f"{side}_thigh"] = (np.array([0.0, y, hh]), _ry(-a))
        out[f"{side}_shin"] = (np.array([Lt * sa, y, hh - Lt * ca]), _ry(a))
        out[f"{side}_foot_link"] = (np.array([0.0, y, p["ankle_height"]]), I)
        c, _, _ = hand_frame(p, sy)
        cy = c[1]
        zs = z0 + Ht - d * 0.6
        out[f"{side}_upper_arm"] = (np.array([0.0, cy, zs]), I)
        out[f"{side}_forearm"] = (np.array([0.0, cy, zs - L1]), _ry(-e))
        out[f"{side}_hand"] = (np.asarray(c, dtype=float), _ry(-e))
    return out


def _local(tris_world, frame):
    o, R = frame
    return (tris_world - o) @ R * mm


def _module(c, p, axis="y", scale=1.0):
    return sun.joint_module(tuple(c), p, axis, scale * 1.03)


# ---- the pieces
def _pelvis(p, hip_z):
    """The pelvis's bones (``skeleton_trunk``'s, with the hip modules at ``hip_z``) and its modules."""
    hh = Humanoid.hip_height(p)
    Wp, Dp, Hp = p["pelvis_width"], p["pelvis_depth"], p["pelvis_height"]
    xs0, hy, wing_h = sun.spine_x(p, 0.0), p["hip_spacing"] / 2, p["iliac_wing_height"]
    bones = sk.rounded_box(Dp * 0.4, Wp * 0.22, Hp * 0.85, (xs0 + 10, 0, hh + Hp * 0.07), fillet=14)       # the sacrum: narrow, between the wings
    modules = None
    for sy in (-1, 1):
        wz = hip_z + wing_h * 0.6                                           # the wing rises from the hip to a crest below the lowest ribs
        wing = (sk.blade((0, 0, 0), Dp * 0.62, wing_h, 16.0, axis="y").rotate((0, 0, 0), (1, 0, 0), -sy * 22)
                .rotate((0, 0, 0), (0, 0, 1), sy * 18).translate((xs0 + Dp * 0.1, sy * (Wp / 2 - 30), wz)))
        bones = bones.union(wing)
        bones = bones.union(sk.rod((xs0 + 10, 0, hh + Hp * 0.5), (xs0 + Dp * 0.1, sy * (Wp / 2 - 30), wz), 20))
        bones = bones.union(sk.rod((xs0 + 10, sy * Wp * 0.15, hh + Hp * 0.3), (0, sy * hy, hip_z), 22))      # the ilium down to the hip
        bones = bones.union(sk.rod((0, sy * hy, hip_z), (Dp * 0.32, sy * 30, hip_z + Hp * 0.1), 18))         # to the pubis
        m = _module((0, sy * hy, hip_z), p, "y")
        modules = m if modules is None else modules.union(m)
    bones = bones.union(sk.rounded_box(28, 90, 36, (Dp * 0.32, 0, hip_z + Hp * 0.1), fillet=8))              # the pubic bar
    return bones, modules


def _vertebrae(p):
    """[(z, solid)] of the vertebral bodies with their discs and processes, in the build pose."""
    hh = Humanoid.hip_height(p)
    z0, Wt, Dt, Ht, zs = sun._torso_frame(p)
    N, dv = int(p["spine_segments"]), p["vertebra_diameter"]
    seg = Ht / N
    out = []
    for i in range(N):
        u = (i + 0.5) / N
        x, z = sun.spine_x(p, u), z0 - 12 + (i + 0.5) * seg
        d_i = dv * (1.12 - 0.3 * u)
        body = sk.disc((x, 0, z), d_i / 2, seg * 0.66, axis="z")
        disc = sk.disc((x, 0, z + seg * 0.5), d_i * 0.42, seg * 0.4, axis="z")
        spinous = sk.rounded_box(dv * 0.5, 18, seg * 0.5, (x - d_i / 2 - dv * 0.2, 0, z - seg * 0.1), fillet=5)
        transverse = sk.rod((x - d_i * 0.25, -d_i * 0.75, z), (x - d_i * 0.25, d_i * 0.75, z), 9)
        out.append((z, body.union(disc).union(spinous).union(transverse)))
    return out


def _arm_pieces(p):
    """(upper arm bone, shoulder disc, elbow module, forearm bone, wrist module) in their links' frames."""
    d, L1, L2 = p["arm_diameter"], p["upper_arm_length"], p["forearm_length"]
    upper = cq.Workplane("XY").circle(d / 2).extrude(-L1).union(cq.Workplane("XY").sphere(d * 0.48).translate((0, 0, -L1)))
    shoulder = cq.Workplane("XZ").circle(d * 0.65).extrude(d * 0.45, both=True)
    elbow = _module((0, 0, 0), p, "y", scale=d * 1.1 / p["joint_module_diameter"])            # at the forearm's origin
    fore = cq.Workplane("XY").circle(d * 0.42).extrude(-L2)
    wrist = _module((0, 0, -L2), p, "y", scale=d * 0.75 / p["joint_module_diameter"])
    return upper, shoulder, elbow, fore, wrist


def _hand_pieces(p, side):
    """[(link, solid)] of a hand in its links' frames (``space_unit_specs.hand_layout``): the palm with the thenar's
    bulk and the wrist's disc, every phalanx a capsule with a knuckle ball at its root, the fingertips padded."""
    sy = 1.0 if side == "L" else -1.0
    lay = su.hand_layout(p, sy)
    pt, pw, pl = lay["palm"]
    d = p["arm_diameter"]
    palm = sk.rounded_box(pt, pw, pl, (0, 0, -pl / 2 - 4), fillet=12).union(sk.disc((0, 0, 0), d * 0.3, 24, axis="z"))
    palm = palm.union(sk.rounded_box(pt * 0.8, 26, pl * 0.55, (pt * 0.35, lay["thumb"]["base"][1] * 0.75, -pl * 0.45), fillet=8))
    out = [(f"{side}_hand", palm)]
    for k, f in enumerate(lay["fingers"]):
        r = f["radius"]
        for j, L in enumerate(f["lengths"]):
            rj = r * (1.0, 0.95, 0.9)[j]
            seg = sk.capsule((0, 0, 0), (0, 0, -L), rj).union(sk.ball((0, 0, 0), rj * 1.12))
            if j == 2:
                seg = seg.union(sk.ball((rj * 0.25, 0, -L + rj * 0.3), rj * 0.95))                  # the fingertip's pad
            out.append((f"{side}_finger_{k}_{j}", seg))
    th = lay["thumb"]
    Lm, Lp, Ld = th["lengths"]
    m, ax, rt = np.asarray(th["dir"]), np.asarray(th["curl_axis"]), th["radius"]
    out.append((f"{side}_thumb_0", sk.capsule((0, 0, 0), tuple(m * Lm), rt).union(sk.ball((0, 0, 0), rt * 1.1))))
    d2 = _rot3(ax, th["tip_bend"]) @ m
    ip = m * Lp
    out.append((f"{side}_thumb_1", sk.capsule((0, 0, 0), tuple(ip), rt * 0.95).union(sk.ball((0, 0, 0), rt * 1.05))
                .union(sk.capsule(tuple(ip), tuple(ip + d2 * Ld), rt * 0.85)).union(sk.ball(tuple(ip + d2 * Ld * 0.9), rt * 0.8))))
    return out


def _rot3(axis, a):
    x, y, z = axis
    K = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + math.sin(a) * K + (1.0 - math.cos(a)) * K @ K


def _half_shell(c, r, L, side=1.0, front=True):
    """A 4 mm casing plate: half a cylindrical shell about z at ``c`` (radius ``r``, length ``L``) on its +x (front)
    or ``side``·y face."""
    shell = sk.disc(c, r, L, "z").cut(sk.disc(c, r - 4.0, L + 2.0, "z"))
    keep = cq.Workplane("XY").box(2 * r + 2, 2 * r + 2, L + 2).translate((c[0] + r + 1, c[1], c[2]) if front else (c[0], c[1] + side * (r + 1), c[2]))
    return shell.intersect(keep)


def _casing_pieces(p, side):
    """The limb's plates: two bands on the upper arm, the forearm, the thigh and the shin (the front faces, and the
    outer faces of the thigh and the shin), in their links' frames."""
    d, La, Lf, L1, L2 = p["arm_diameter"], p["upper_arm_length"], p["forearm_length"], p["thigh_length"], p["shin_length"]
    sy = 1.0 if side == "L" else -1.0
    out = []
    for link, r, L, x in ((f"{side}_upper_arm", d / 2 + 6, La, 0.0), (f"{side}_forearm", d * 0.42 + 6, Lf, 0.0),
                          (f"{side}_thigh", p["femur_diameter"] * 0.9, L1, 10.0), (f"{side}_shin", p["tibia_diameter"] * 0.85, L2, 0.0)):
        for zc in (-0.31 * L, -0.69 * L):
            plate = _half_shell((x, 0.0, zc), r, 0.3 * L, front=True)
            if "thigh" in link or "shin" in link:
                plate = plate.union(_half_shell((x, 0.0, zc), r, 0.3 * L, side=sy, front=False))
            out.append((link, plate))
    return out


def build(casing: str = "plastic", tol: float = 4.0, P: dict | None = None) -> Mesh:
    """Every piece from the CAD (``casing``: 'none' | 'plastic' | 'lorica' - the plates for the latter two)."""
    P = dict(su.MARINE if P is None else P)
    p = sun.SpaceMarine().resolve()
    p.update({k: v for k, v in P.items() if k in p})
    F = frames(p)
    hh, Hp, Ht = Humanoid.hip_height(p), p["pelvis_height"], p["torso_height"]
    z0, Wt, Dt, _, zs = sun._torso_frame(p)
    split = 0.42 * Ht
    mesh = Mesh()

    def add(body, shape, rgba, t=tol):
        tris = _tris(shape, t)
        if len(tris):
            mesh.pieces.append((body, _local(tris, F[body]), rgba))

    def add_local(body, shape, rgba, t=tol):
        tris = _tris(shape, t)
        if len(tris):
            mesh.pieces.append((body, tris * mm, rgba))

    # the trunk
    bones, hips = _pelvis(p, hh - Hp / 2)
    add("pelvis", bones, BONE)
    add("pelvis", hips, MODULE)
    for z, v in _vertebrae(p):
        add("torso" if z < z0 + split else "thorax", v, BONE)
    trunk = sun.skeleton_trunk(p)
    for name in ("ribcage", "neck"):
        add("thorax", trunk[name], BONE)
    add("thorax", trunk["head"], PLATE)
    add_local("torso", _module((0, 0, 0), p, "y", 1.1), MODULE)                                     # the lumbar module
    add_local("torso", sk.disc((0, 0, -Hp * 0.25), p["joint_module_diameter"] * 0.6, 40, "z"), MODULE)   # the waist yaw's stage
    add_local("thorax", _module((0, 0, 0), p, "y", 1.1), MODULE)                                    # the thoracic module
    add_local("thorax", sk.rounded_box(2 * Dt * 0.17, 2 * Wt * 0.2, 2 * Ht * 0.13, (-Dt * 0.05, 0, Ht * 0.6 - split), fillet=12), BAY)
    add_local("torso", sk.rounded_box(2 * p["pelvis_depth"] * 0.21, 2 * p["pelvis_width"] * 0.25, 2 * Ht * 0.1, (20, 0, Ht * 0.17), fillet=12), BAY)
    # the limbs
    knee, ankle, roll = (_module((0, 0, -p["thigh_length"]), p, "y", 0.9), _module((0, 0, -p["shin_length"]), p, "y", 0.8),
                         _module((0, 0, -p["ankle_height"] * 0.45), p, "x", 0.5))
    femur, tibia, foot = sun.femur(p).cut(knee), sun.tibia(p).cut(ankle), sun.foot_bones(p).cut(roll)   # the modules drawn dark on their own
    upper, shoulder, elbow, fore, wrist = _arm_pieces(p)
    girdle = trunk["girdle"]
    for sy in (1.0, -1.0):
        cy = sy * (Wt / 2 + 10 + p["arm_diameter"] * 0.5)
        girdle = girdle.cut(sun.joint_module((0, cy, zs), p, "y", scale=p["arm_diameter"] * 1.3 / p["joint_module_diameter"] * 1.03))
    for side, sy in (("L", 1.0), ("R", -1.0)):
        add_local(f"{side}_thigh", femur, BONE)
        add_local(f"{side}_thigh", knee, MODULE)
        add_local(f"{side}_shin", tibia, BONE)
        add_local(f"{side}_shin", ankle, MODULE)
        add_local(f"{side}_foot_link", foot, BONE)
        add_local(f"{side}_foot_link", sk.rounded_box(p["foot_length"], p["foot_width"], p["foot_thickness"] + 30.0,
                                                      (p["foot_length"] * 0.2, 0, -p["ankle_height"] + (p["foot_thickness"] + 30.0) / 2), fillet=14), PLATE)   # a solid foot over the bones
        add_local(f"{side}_foot_link", roll, MODULE)
        add_local(f"{side}_upper_arm", upper, BONE)
        add_local(f"{side}_upper_arm", shoulder, MODULE)
        add_local(f"{side}_forearm", elbow, MODULE)
        add_local(f"{side}_forearm", fore, BONE)
        add_local(f"{side}_forearm", wrist, MODULE)
        for link, piece in _hand_pieces(p, side):
            add_local(link, piece, DARK, t=2.0)
        cy = sy * (Wt / 2 + 10 + p["arm_diameter"] * 0.5)
        add_local("thorax", _module((0, cy, zs - z0 - split), p, "y", p["arm_diameter"] * 1.3 / p["joint_module_diameter"]), MODULE)
        if casing != "none":
            for link, plate in _casing_pieces(p, side):
                add_local(link, plate, PLATE)
    add("thorax", girdle, BONE)
    if P.get("tail_segments"):                                           # the prehensile tail: a caudal vertebra and a module per link, the grip pad at the tip
        n, seg = int(P["tail_segments"]), P["tail_length"] / int(P["tail_segments"])
        for k in range(n):
            f = k / max(n - 1, 1)
            d = P["tail_base_diameter"] * (1 - f) + P["tail_tip_diameter"] * f
            add_local(f"tail_{k}", sk.rod((-d * 0.45, 0, 0), (-seg + d * 0.45, 0, 0), d * 0.32), BONE)
            add_local(f"tail_{k}", sk.disc((0, 0, 0), d * 0.55, d * 0.8, "y"), MODULE)
            if k == n - 1:
                add_local(f"tail_{k}", sk.rounded_box(d * 1.6, d * 1.4, d * 1.1, (-seg, 0, -d * 0.2), fillet=max(1.0, d * 0.3)), DARK)
    return mesh


# ---- the cache
def _key(casing, tol, P):
    P = su.MARINE if P is None else P
    h = hashlib.sha1(json.dumps({"P": {k: P[k] for k in sorted(P)}, "casing": casing, "tol": tol, "v": VERSION}, sort_keys=True, default=str).encode()).hexdigest()[:12]
    return f"belisario_{casing}_{h}"


def save(mesh: Mesh, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {f"p{i}": tris for i, (_, tris, _) in enumerate(mesh.pieces)}
    meta = {"pieces": [(b, list(rgba)) for b, _, rgba in mesh.pieces],
            "struts": [(a, la.tolist(), b, lb.tolist(), rb, rr) for a, la, b, lb, rb, rr in mesh.struts]}
    np.savez_compressed(path, meta=json.dumps(meta), **arrays)


def read(path: Path) -> Mesh:
    z = np.load(path)
    meta = json.loads(str(z["meta"]))
    mesh = Mesh()
    for i, (body, rgba) in enumerate(meta["pieces"]):
        mesh.pieces.append((body, z[f"p{i}"], tuple(rgba)))
    for a, la, b, lb, rb, rr in meta["struts"]:
        mesh.struts.append((a, np.asarray(la), b, np.asarray(lb), rb, rr))
    return mesh


def load(casing: str = "plastic", tol: float = 4.0, P: dict | None = None, cache_dir: Path | None = None) -> Mesh:
    """The mesh from the cache, built and cached when missing."""
    path = (Path(cache_dir) if cache_dir else CACHE) / (_key(casing, tol, P) + ".npz")
    if path.exists():
        return read(path)
    mesh = build(casing, tol, P)
    save(mesh, path)
    return mesh


if __name__ == "__main__":
    import sys
    import time
    t = time.time()
    m = load(sys.argv[1] if len(sys.argv) > 1 else "plastic")
    print(f"{m.triangles()} triangles on {len(m.bodies())} bodies, {len(m.struts)} struts, {time.time() - t:.1f} s")
