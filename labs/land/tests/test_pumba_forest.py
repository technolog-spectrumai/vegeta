"""Pumba in the forest (designs/pumba_forest.py, sims/pumba_forest_movie.py): the scene keeps the trail clear,
the grab's jaws close with a positive angle, and — the full MuJoCo mission (slow) — Pumba crosses the branches,
lifts the log off the trail and drives on.

Run: python tools/lab.py test land -- -k forest           (the mission: add -m slow)
"""
import math

import numpy as np
import pytest

import pekari_arms as pa
import pumba_forest as pfo


def test_trees_and_shrubs_stay_off_the_trail():
    sc = pfo.Scene()
    pts = [(0.0, 0.0)] + list(sc.waypoints)
    assert len(sc.trees) == sc.n_trees
    for x, y, r, _ in sc.trees:
        assert pfo._dist_to_polyline((x, y), pts) >= sc.clear + r
    assert 8.0 < sc.log_mass < 12.0                       # a 10 cm, 1.6 m green spruce log


def test_pursuit_point_lies_ahead_on_the_trail():
    pts = [(0.0, 0.0), (10.0, 0.0)]
    assert pfo._pursuit_point((2.0, 0.5), pts, 1.5) == pytest.approx((3.5, 0.0))
    assert pfo._pursuit_point((9.5, 0.0), pts, 1.5) == pytest.approx((10.0, 0.0))


def test_grab_closes_on_positive_angles():
    """The jaws' hinge axes: + turns each tip towards the other (the convention in pekari_arms)."""
    pytest.importorskip("mujoco")
    base, _, _ = pa.links(pa.PUMBA_MAIN_ARM, "m_", (0.0, 0.0, 0.0))

    def find(link, name):
        if link.name == name:
            return link
        for c in link.children:
            f = find(c, name)
            if f is not None:
                return f
    a, b = find(base, "m_grip_a_link"), find(base, "m_grip_b_link")
    # finger a sits on +z of the tool, b on -z; a rotation +q about the axis must move a's tip (+x) towards -z
    for link, z_side in ((a, 1.0), (b, -1.0)):
        ax = np.asarray(link.joints[0].axis, dtype=float)
        tip_motion = np.cross(ax, np.array([1.0, 0.0, 0.0]))          # d(tip)/dq for a tip on +x
        assert tip_motion[2] * z_side < 0


@pytest.mark.slow
def test_mission_clears_the_log_and_drives_on():
    pytest.importorskip("mujoco")
    lab, sc = pfo.make_lab()
    ep = pfo.run(lab, sc)
    s = pfo.summary(ep, sc)
    assert s["log_lift_m"] > 0.3 and s["log_off_trail"]
    assert s["distance_m"] > 19.0
    assert any(e[1] == "grasp" and "held" in e[2] for e in lab.events)
