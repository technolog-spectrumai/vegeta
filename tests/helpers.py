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


def rich():
    """A two-link machine holding something of every system tag (the tag and YX2 tests)."""
    import numpy as np
    base = hub.Link("base", mass=2.0, inertia=np.diag([0.1, 0.1, 0.1]), shapes=[hub.Shape("hull", "box", (0.3, 0.2, 0.1))],
                    point_masses=[("battery", 0.3, (0.0, 0.0, 0.05))],
                    data={"material": "al6061", "strength": {"allowable_pa": 2.4e8, "safety_factor": 1.5, "source": "notebook"},
                          "aero": {"S_m2": 0.1}, "sensors": [{"name": "cam", "type": "camera", "pos": [0.1, 0, 0]}],
                          "sdf": {"sensor": ["<sensor name='cam' type='camera'/>"], "plugin": ["<plugin name='x'/>"]}})
    arm = hub.Link("arm", pos=(0.2, 0.0, 0.0), mass=0.5, inertia=np.diag([0.01, 0.01, 0.01]), shapes=[hub.Shape("rod", "capsule", (0.01, 0.2))])
    j = hub.Joint("shoulder", "revolute", "base", "arm", pos=(0.2, 0.0, 0.0), axis=(0.0, 1.0, 0.0), lower=-1.0, upper=1.0,
                  drive={"stiffness": 10.0, "damping": 1.0, "max_force": 5.0},
                  data={"damping": 0.1, "motor": {"stall_nm": 2.0, "no_load_rad_s": 50.0, "source": "datasheet"}, "connection": {"type": "pin"}})
    return hub.Machine("rich", links=[base, arm], joints=[j], producer={"tool": "test", "revision": "abc"},
                       data={"materials": {"al6061": {"density_kg_m3": 2700.0, "E_pa": 69e9, "source": "handbook"}},
                             "connections": [{"name": "b1", "links": ["base", "arm"], "type": "bolt", "spec": {"size": "M3"}}],
                             "aerodynamics": {"reference": {"S_m2": 0.1}, "source": "cfd"},
                             "propulsion": {"rotors": [{"name": "r0", "pos": [0, 0, 0], "axis": [0, 0, 1], "spin": 1, "propeller": "p", "motor": "mo"}],
                                            "propellers": {"p": {"diameter_m": 0.2, "coefficients": {"J": [0, 1], "CT": [0.1, 0.0], "CP": [0.05, 0.01]}}},
                                            "motors": {"mo": {"power_w": 100.0}}}})
