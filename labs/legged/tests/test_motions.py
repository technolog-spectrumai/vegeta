"""The design-stage motions (``*_motions.py``): each machine has its motions, every IK-solved keyframe is within
5 mm of its target, the joints stay inside their ranges, stance feet stay on the ground, and the block is what
the export carries (``data:motions``, tag ``motion``)."""
import importlib

import numpy as np
import pytest

pytest.importorskip("mujoco")

#: "module" or "module:animal" (its ``motions_<animal>`` and ``parts_<animal>``) -> the motions it must have
MOTIONS = {"humanoid_motions": {"walk", "wave"},
           "zoo_motions:dog": {"walk", "trot", "bound"}, "zoo_motions:bear": {"walk", "trot"},
           "zoo_motions:deer": {"walk", "trot", "bound"},
           "zoo_motions:elephant": {"walk", "trunk_swing", "trunk_lift"},
           "zoo_motions:snake": {"slither", "sidewind"}, "zoo_motions:bat": {"fold", "unfold", "flap"},
           "zoo_motions:octopus": {"crawl", "pulse", "reach"},
           "myriapod_motions:cleopatra": {"walk", "turn"}, "myriapod_motions:apheloria": {"walk", "curl", "uncurl"}}


class _Mod:
    """A motions module seen through one machine's ``motions``/``parts``."""

    def __init__(self, spec):
        name, _, animal = spec.partition(":")
        mod = importlib.import_module(name)
        self.motions = getattr(mod, f"motions_{animal}") if animal else mod.motions
        self.parts = getattr(mod, f"parts_{animal}") if animal else mod.parts


@pytest.fixture(scope="module", params=sorted(MOTIONS))
def machine(request):
    mod = _Mod(request.param)
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


def test_gaits_travel_at_their_froude_speed():
    import motion_kit as mk
    import zoo_motions as zm
    import zoo_specs as zs

    _, ms = zm.motions_dog()
    walk = {m.name: m for m in ms}["walk"]
    v = walk.base["velocity_m_s"][0]
    assert v == pytest.approx(mk.froude_speed(mk.GAITS["walk"]["froude"], zm.leg_length(zs.DOG)), rel=1e-3)
    assert walk.feet["FL"]["duty"] == mk.GAITS["walk"]["duty"]


def test_myriapod_gait_is_the_controllers():
    import myriapod_motions as mym
    import myropod_controller as mc

    kin, ms = mym.motions_cleopatra()
    walk, turn = ms
    p = mc.ControllerParams()
    assert walk.base["velocity_m_s"][0] == pytest.approx(p.v_target)
    assert walk.period == pytest.approx(p.stride / p.v_target)
    assert {f["duty"] for f in walk.feet.values()} == {p.duty}
    assert turn.base["velocity_m_s"][2] < 0                     # the longer left stride turns it right
