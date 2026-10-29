"""The design-stage motions (``*_motions.py``): each machine has its motions, every IK-solved keyframe is within
5 mm of its target, the joints stay inside their ranges, stance feet stay on the ground, and the block is what
the export carries (``data:motions``, tag ``motion``)."""
import importlib

import numpy as np
import pytest

pytest.importorskip("mujoco")

#: motions module -> the motions it must have
MOTIONS = {"humanoid_motions": {"walk", "wave"}}


@pytest.fixture(scope="module", params=sorted(MOTIONS))
def machine(request):
    mod = importlib.import_module(request.param)
    kin, ms = mod.motions()
    return request.param, mod, kin, ms


def test_names(machine):
    name, _, _, ms = machine
    assert {m.name for m in ms} == MOTIONS[name]


def test_checks(machine):
    _, _, kin, ms = machine
    for m in ms:
        c = m.check
        assert c.get("ik_error_max_m", 0.0) < 0.005, (m.name, c)
        assert c["limit_margin_min_rad"] >= 0.0, (m.name, c)
        assert c.get("clipped_max_rad", 0.0) < 1e-6, (m.name, c)
        assert c.get("stance_height_spread_m", 0.0) < 0.005, (m.name, c)
        assert np.all(np.isfinite(m.Q)) and m.Q.shape == (len(m.t), len(m.joints))
        assert set(m.joints) <= set(kin.names)


def test_block(machine):
    _, mod, _, ms = machine
    b = mod.parts(None)
    assert b["tags"] == {"data:motions*": "motion"}
    data = b["motions"]
    assert data["format"] == "vegeta-motions/1" and set(data["motions"]) == {m.name for m in ms}
    for d in data["motions"].values():
        assert len(d["q"]) == len(d["t"]) and all(len(r) == len(d["joints"]) for r in d["q"])
