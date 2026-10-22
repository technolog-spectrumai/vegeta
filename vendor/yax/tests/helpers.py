"""Comparing machines: every link's mass, centre of mass and inertia, every joint's kind, anchor, axis and limits, in
the frame of the first root link (formats that cannot place the root in the world — URDF — still compare)."""
import numpy as np

from yax import hub


def _rel(m):
    root = next(lk for lk in m.links if m.parent_joint(lk.name) is None or m.parent_joint(lk.name).parent is None)
    jr = m.parent_joint(root.name)
    rp, rq = (root.pos, root.quat)
    Ri = hub.R(rq).T
    out = {}
    for lk in m.links:
        p, q = hub.to_frame(lk.pos, lk.quat, rp, rq)
        Rl = hub.R(q)
        com = np.asarray(p) + Rl @ np.asarray(lk.com)
        I = Rl @ np.asarray(lk.inertia if lk.inertia is not None else np.zeros((3, 3))) @ Rl.T
        out[lk.name] = (lk.mass or 0.0, com, I)
    joints = {}
    for j in m.joints:
        if j.parent is None and j.kind == "fixed":
            continue
        joints[j.name] = (j.kind, Ri @ (np.asarray(j.pos) - rp), Ri @ np.asarray(j.axis), j.lower, j.upper)
    return out, joints


def assert_same(a, b, *, tol=1e-9, kinds=None, skip_joints=()):
    la, ja = _rel(a)
    lb, jb = _rel(b)
    for name, (ma, ca, Ia) in la.items():
        if name not in lb:
            assert ma == 0.0, f"link {name} missing"
            continue
        mb, cb, Ib = lb[name]
        assert abs(ma - mb) <= tol * max(1, ma), (name, ma, mb)
        np.testing.assert_allclose(cb, ca, atol=tol, err_msg=f"{name} com")
        np.testing.assert_allclose(Ib, Ia, atol=tol * max(1, np.abs(Ia).max()), err_msg=f"{name} inertia")
    for name, (ka, pa, xa, loa, hia) in ja.items():
        if name in skip_joints:
            continue
        kb, pb, xb, lob, hib = jb[name]
        assert (kinds or {}).get(ka, ka) == kb, (name, ka, kb)
        np.testing.assert_allclose(pb, pa, atol=tol, err_msg=f"joint {name} anchor")
        if ka not in ("fixed", "spherical"):
            np.testing.assert_allclose(xb, xa, atol=tol, err_msg=f"joint {name} axis")
        if loa is not None:
            assert abs(loa - lob) <= tol and abs(hia - hib) <= tol, (name, loa, lob, hia, hib)
