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


def test_spider_legs_stand_and_salty_hips_line_up():
    import spider_specs as ss
    for s in ss.SPIDERS.values():
        c = ss.leg_chain(s)
        assert c["toe"][1] == pytest.approx(s["pad"][2] + 1.0)                  # the toe just over the pad
        assert c["knee"][1] > c["lift"][1] > 0                                 # knees high
        (hx, hy), _, sy = ss.hip_points(s, "L1")
        assert hy > ss.rim_y(s, hx) and sy == 1                                 # the hip just outside the rim
    s = ss.SALTY
    assert ss.leg_chain(s, s["jump_leg"], s["jump_angles"])["z_lift"] == pytest.approx(ss.leg_chain(s)["z_lift"], abs=0.05)


@pytest.mark.parametrize("cls", ["Arachne2", "Tarantula", "Salty"])
def test_spider_cad_parts_and_insides_fit(cls):
    pytest.importorskip("cadquery")
    import shape_kit as sk
    import spiders

    d = getattr(spiders, cls)()
    s = d.spec
    for part in ("foot", "leg", "body", "internals"):
        assert d.generate(part=part).measure()["valid"]
    shell = sk.union_all(d.shells(s)).val()
    for w in d.motor_bank(s) + [d.battery(s), d.board(s)]:
        v = w.val()
        assert v.cut(shell).Volume() < 0.01 * v.Volume()                        # inside the body


def test_hydrostatic_only_pays_where_tendons_rub():
    o = sa.tendon_routing("TARANTULA")
    h, r = o["hydrostatic"], o["routes"]
    assert o["break_even_eta"] == pytest.approx(0.88)
    pulleys, sleeve = r["pulleys on the joint axes"], r["sleeve, μ 0.2, 150°"]
    assert pulleys["mass_g"] < h["mass_g"] and pulleys["walk_w"] < h["walk_w"]       # pulleys: tendons win
    assert sleeve["mass_g"] > h["mass_g"] and sleeve["walk_w"] > h["walk_w"]         # rubbing sleeves: hydrostatic
    for small in ("ARACHNE-2", "SALTY"):
        assert sa.tendon_routing(small)["hydrostatic"]["share"] > 0.5              # too heavy on the small ones


def test_tarantula_jaws():
    c = sa.cutter()
    assert c["max_gap_mm"] == pytest.approx(9.8, abs=0.3) and 0 < c["edges_cross_deg"] < 10
    cuts = c["cuts"]
    assert cuts["copper wire Ø1.5 mm"]["cuts"] and cuts["mild steel wire Ø1.0 mm"]["cuts"]
    assert cuts["cable tie 4.8 × 1.3 mm (nylon 6/6)"]["cuts_at_stall"]
    assert not cuts["galvanised fence wire Ø2.5 mm"]["cuts_at_stall"]


def test_tarantula_jaws_cad_clear():
    pytest.importorskip("cadquery")
    import spiders

    d = spiders.Tarantula()
    s = d.spec
    assert d.generate(part="jaws", jaw=30.0).measure()["valid"]
    for deg in (0.0, 30.0):
        ch = d.chelicerae(s, deg)
        jl, jr = (w.val() for w in ch["jaws"])
        assert jl.intersect(jr).Volume() < 0.5                                   # the jaws pass (bypass)
        assert jl.intersect(ch["housing"][0].val()).Volume() < 0.5
        assert jl.intersect(ch["pins"][0].val()).Volume() < 0.5                  # clear of the other jaw's pin


def test_tarantula_attachment_verdict():
    import tarantula_attachment as ta
    st_ = ta.study("wave", "crouched")
    assert st_["glass"]["best"] == "pad" and st_["glass"]["pad"] >= 90
    assert st_["brick"]["spine"] >= 90 and st_["brick"]["pad"] is None                # no adhesive contact on brick
    assert st_["bare steel"]["magnet"] == 180 and st_["painted wall"]["hybrid"] < 45
    assert 14 <= st_["glass"]["friction"] <= 20                                        # tan β ≤ μ/2
    assert 1.8 < ta.monte_carlo("brick", "spine")["p5"] < 2.0                          # 160 spines: just short at p5
    n0 = ta.MODULES["spine"]["spines"]
    try:
        ta.MODULES["spine"]["spines"] = 192                                             # a sixth row: back over 2
        assert ta.monte_carlo("brick", "spine")["p5"] >= 2.0
    finally:
        ta.MODULES["spine"]["spines"] = n0
    assert ta.POSTURES["crouched"] == pytest.approx(41.6, abs=0.1)                     # the abdomen's belly 5 mm off
    assert ta.max_inclination("brick", "spine", "wave", "standing") < 90               # it must crouch


def test_tarantula_robot_and_gaits():
    pytest.importorskip("mujoco")
    import motion_kit as mk
    import tarantula_gaits as tg
    import tarantula_robot as tr

    r = tr.tarantula()
    assert r.total_mass() == pytest.approx(1.5, abs=1e-6) and r.summary()["n_actuated"] == 24 and len(r.feet) == 8
    assert tr.drive().stall_torque == pytest.approx(1.62)                  # 1000:1 gearmotor × 1.5 pulley (§5)
    kin = mk.Kin(r)
    for name, drop in (("wave4", 0.0), ("tetrapod", tg.POSTURES["crouched"]), ("turn", 0.0)):
        m = tg.gait(name, kin=kin, body_drop_mm=drop)
        assert m.check["ik_error_max_m"] < 1e-4 and m.check["limit_margin_min_rad"] > 0
        assert m.check["static_margin_min_m"] > 0.03                          # statically stable throughout
    up_stand = tg.gait("wave4", kin=kin).check["tip_deg_min"]["up"]
    up_crouch = tg.gait("wave4", kin=kin, body_drop_mm=tg.POSTURES["crouched"]).check["tip_deg_min"]["up"]
    assert up_crouch > up_stand                                              # crouching resists tipping


def test_tarantula_runner_family():
    import spider_sims as ss_

    fam = ss_.FAMILIES["tarantula"]
    assert {"tarantula-walk", "tarantula-slope", "tarantula-climb", "tarantula-attach"} <= set(fam)
    for n in fam:
        assert len(ss_.SIMS[n][0](True)) >= 1 and len(ss_.SIMS[n][0](False)) >= len(ss_.SIMS[n][0](True))


def test_tarantula_spine_foot():
    pytest.importorskip("cadquery")
    import spider_specs as ss_
    import spiders

    s = ss_.TARANTULA
    sc = s["spine_comb"]
    fp = spiders.foot_parts(s)
    assert len(fp["spines"]) == sc["rows"] and sc["rows"] * sc["per_row"] == 160          # §10's comb
    assert fp["foot"].val().isValid() and all(r.val().isValid() for r in fp["spines"])
    for r in fp["spines"]:
        bb = r.val().BoundingBox()
        assert -sc["reach"] - 0.2 < bb.zmin < 0.0                                    # tips just below the sole
        assert r.val().intersect(fp["named"]["sole"].val()).Volume() < 1e-3          # clear of the rubber sole
    assert spiders.foot_parts(spiders.ss.ARACHNE2)["spines"] == []                    # the others keep claw + pad
    d = spiders.Tarantula()
    assert d.generate(part="leg").measure()["n_solids"] > sc["rows"]
    assert d.generate(part="foot").measure()["valid"]


def test_tarantula_dragline_sizing():
    w = sa.tarantula_winch()
    assert w["line_capacity_m"] >= 10.0 and w["line_safety"] > 20
    assert w["motor_load_fraction"]["all the weight"] < 0.3 and w["reel_in_mm_s"]["all the weight"] > 40
    assert w["can_recock"] and 8 < w["harpoon_m_s"] < 14 and w["anchor_safety"][0] > 5


def test_tarantula_cords_ride_over_and_under_the_knee():
    import spider_specs as ss_
    import spiders

    c = ss_.leg_chain(ss_.TARANTULA)
    r = spiders.tendon_routes(ss_.TARANTULA)
    knee_z = c["knee"][1]
    wrap_a = next(p for tag, p in r["flexor"]["pts"] if tag == "wrap")
    wrap_b = next(p for tag, p in r["coupling"]["pts"] if tag == "wrap")
    assert wrap_a[2] > knee_z > wrap_b[2]                                         # flexor over, coupling under
    assert [tag for tag, _ in r["flexor"]["pts"]] == ["hip", "hip", "femur", "femur", "wrap", "tibia"]
    assert "cords" not in ss_.SALTY and "cords" not in ss_.ARACHNE2               # the others keep their legs


def test_tarantula_dragline_pod_cad():
    pytest.importorskip("cadquery")
    import shape_kit as sk
    import spiders

    d = spiders.Tarantula()
    s = d.spec
    assert d.generate(part="winch").measure()["valid"]
    pod = [v.val() for vs in d.dragline_pod(s).values() for v in vs]
    legs = [v for w in d.legs(s) for v in w.vals()]
    assert sum(a.intersect(b).Volume() for a in pod for b in legs) < 1.0           # clear of the legs
    deck = sk.union_all(d.deck(s)).val()
    assert sum(a.intersect(deck).Volume() for a in pod) < 1.0                      # and of the payload deck

