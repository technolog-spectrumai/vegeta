"""The spiders (docs/spiders.md): ARACHNE's budget, robot model, statics and gaits; the actuation study's verdicts;
SALTY's catapult; the simulation runner's YAML and one short MuJoCo climb."""
import math

import pytest

import arachne_specs as sp
import spider_actuation as sa


def test_budget_and_geometry():
    assert sum(sp.mass_budget().values()) == pytest.approx(156.5, abs=0.5)
    pads = [sp.leg_points(f)["pad"] for f in sp.LEGS]
    assert max(p[0] for p in pads) - min(p[0] for p in pads) == pytest.approx(181.4, abs=2)
    assert sp.servo_stall_nm() == pytest.approx(0.8 * 9.81 / 100 * 3.8 / 4.8)


def test_robot_model_mass_and_joints():
    pytest.importorskip("mujoco")
    import arachne_robot as ar
    r = ar.arachne()
    assert r.total_mass() * 1000 == pytest.approx(sp.gross_mass_g(), abs=0.5)
    assert r.summary()["n_actuated"] == 24 and len(r.feet) == 8


def test_statics_rules():
    pytest.importorskip("mujoco")
    import arachne_robot as ar
    import arachne_statics as st
    import motion_kit as mk
    kin = mk.Kin(ar.arachne())
    six = [f for f in sp.LEGS if f not in ("L1", "R3")]
    glass = st.check(kin, six, surface="glass")
    assert glass["margin"] >= 2.0
    F = sum(glass["forces"].values())
    assert F == pytest.approx(-st.gravity_body(kin.robot.total_mass()), abs=1e-6)   # equilibrium
    assert st.check(kin, six, surface="glass", pivot_mm=1.9)["margin"] < glass["margin"]   # the low pivot helps
    assert st.check(kin, six, surface="bark", module="spine", heading_deg=180)["margin"] == 0.0  # spines: one way


@pytest.mark.parametrize("gait,surface,need", [("climb", "glass", 6), ("climb_fast", "glass", 4),
                                               ("climb_careful", "varnished wood", 7)])
def test_gaits_keep_feet_and_margin(gait, surface, need):
    pytest.importorskip("mujoco")
    import arachne_gaits as ag
    import arachne_robot as ar
    import motion_kit as mk
    m = ag.wall_gait(mk.Kin(ar.arachne()), gait, surface=surface)
    assert m.check["attached_min"] >= need
    assert m.check["adhesion_margin_min"] >= 2.0
    assert m.check["ik_error_max_m"] < 1e-4
    assert m.check["ankle_tilt_max_deg"] <= sp.LEG["ankle_tilt_free_deg"]


def test_actuation_verdicts():
    assert sa.recommend("ARACHNE-2")[0] == "tendon"
    best, second, t = sa.recommend("TARANTULA")
    assert best == "tendon" and second == "spider"
    assert t["spider"]["walk_w"] < t["tendon"]["walk_w"] < t["valved"]["walk_w"]
    assert t["spider"]["seals"] > 0 == t["tendon"]["seals"]
    assert not t["servo"]["ok"]


def test_salty_catapult():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sims"))
    import salty_jump as sj
    r = sj.simulate()
    assert r["took_off"] and r["distance_m"] == pytest.approx(1.05, abs=0.05)
    assert 0.8 < r["efficiency"] < 0.95 and r["pressure_end_bar"] < 16.0          # the gas expanded
    assert sj.simulate(line_id_mm=1.5)["distance_m"] < 0.6 * r["distance_m"]
    j = sa.jump(0.25, 1.0, 0.05)
    assert j["v_m_s"] == pytest.approx(math.sqrt(9.81), rel=1e-6)


def test_runner_yaml_round_trip(tmp_path):
    yaml = pytest.importorskip("yaml")
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sims"))
    import spider_sims as ss
    doc = {"a": 1.5, "b": [1, {"c": "x: y", "d": None}], "e": {"f": True, "g": []}, "h": "yes", "i": float("nan")}
    back = yaml.safe_load(ss.to_yaml(doc))
    assert back["b"][1] == {"c": "x: y", "d": None} and back["h"] == "yes" and back["e"]["f"] is True
    idx = ss.run(["salty-jump", "spider-actuation"], 2, tmp_path, quick=True)
    assert idx["simulations"]["salty-jump"]["errors"] == 0
    assert yaml.safe_load((tmp_path / "salty-jump.yaml").read_text())["summary"]["reach_1m"] >= 1


def test_climb_sim_short():
    pytest.importorskip("mujoco")
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sims"))
    import arachne_climb as ac
    r = ac.run("climb", "glass", cycles=1)
    assert not r["fell"] and r["climbed_mm"] > 10 and r["detached_steps"] == 0
