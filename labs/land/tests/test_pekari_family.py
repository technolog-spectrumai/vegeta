"""The Pekari family — Pekari-2M, Pumba and Calydon (docs/pekari_pumba.md): the shared arm description, the platforms' mass,
CG, stability, lift capacity, drive and power, the CAD and the Chiron (MuJoCo) models.

Run: python tools/lab.py test land -- -k pekari_family
"""
import math

import pytest

from ogma import actuators as act
import pekari_arms as pa
import pekari_family as pf
import pekari_rover_robot as prr
import pekari_2m_robot as p2
import pumba_robot as pu
import calydon_robot as cr

PLATFORMS = [p2.PEKARI_2M, pu.PUMBA, cr.CALYDON]


# ----------------------------------------------------------------------------------------------- the arms
@pytest.mark.parametrize("arm", [pa.PEKARI_ARM, pa.PUMBA_MAIN_ARM, pa.CALYDON_MAIN_ARM, pa.CALYDON_AUX_ARM], ids=lambda a: a.name)
def test_ik_fk_round_trip_and_ranges(arm):
    R = pa.reach(arm)
    for r, z, elev in ((0.5 * R, -0.2, -math.pi / 2), (0.7 * R, 0.1, 0.0), (0.4 * R, 0.3, -math.pi / 4)):
        q = pa.ik(arm, (r * 0.8, r * 0.6, z), elev)
        assert q is not None
        p = pa.fk(arm, q)
        assert p["tcp"] == pytest.approx((r * 0.8, r * 0.6, z), abs=1e-9)
        assert p["elevation"] == pytest.approx(elev, abs=1e-9)
    assert pa.ik(arm, (R * 1.05, 0.0, 0.0), 0.0) is None


@pytest.mark.parametrize("arm", [pa.PEKARI_ARM, pa.PUMBA_MAIN_ARM, pa.CALYDON_MAIN_ARM, pa.CALYDON_AUX_ARM], ids=lambda a: a.name)
def test_rated_payload_is_within_the_drives(arm):
    """The stated rating holds at its reach (stall / 1.5), the curve falls with reach, and at the rating every
    pitch joint is inside its module's peak torque over the safety factor."""
    assert pa.rated_load(arm, arm.rated_reach) >= arm.rated_payload_kg
    curve = [kg for _, kg in pa.load_curve(arm, 9)]
    assert all(a >= b - 1e-9 for a, b in zip(curve, curve[1:]))
    assert pa.rated_load(arm, pa.reach(arm) + 0.01) == 0.0
    q = pa._outstretched(arm, arm.rated_reach)
    tau = pa.gravity_torques(arm, q, arm.rated_payload_kg)
    for j, t in tau.items():
        assert t <= act.get(arm.drives[j]).stall_Nm / pa.SF_PEAK + 1e-6, j


def test_gravity_torque_hand_value():
    """The light arm outstretched with no load: the shoulder holds every outboard mass at its lever."""
    arm = pa.PEKARI_ARM
    q = pa.pose_level(arm)
    pts = pa.mass_points(arm, q)
    expect = sum(kg * pa.G * x for n, kg, x, _, _ in pts if n not in ("yaw drive", "shoulder drive"))
    assert pa.gravity_torques(arm, q)["shoulder"] == pytest.approx(expect, rel=1e-9)


def test_dof_and_the_main_arms_have_six():
    assert pa.arm_dof(pa.PEKARI_ARM) == 5
    for arm in (pa.PUMBA_MAIN_ARM, pa.CALYDON_MAIN_ARM):
        assert pa.arm_dof(arm) == 6 and "wrist_yaw" in arm.drives


# ----------------------------------------------------------------------------------------------- the platforms
def test_pekari_rover_is_untouched():
    """Pekari-2M builds on the Pekari Rover's functions; the rover's own numbers do not change."""
    assert prr.DESIGN["track_width"] == 80.0 and prr.PAYLOAD_KG == 5.0
    assert prr.cg()["loaded"]["mass_kg"] == pytest.approx(sum(prr.mass_budget().values()) + 5.0)


@pytest.mark.parametrize("pl", PLATFORMS, ids=lambda p: p.name)
def test_chassis_is_a_valid_pekari_design(pl):
    assert set(pl.chassis) == set(prr.DESIGN)
    g = pf.geometry(pl)
    assert g["links"] > 0 and 0 <= g["take_up"] < g["pitch"]
    assert pl.n_bogies * 2 == pl.chassis["road_wheels"]


def test_platform_classes_and_relative_scale():
    s2, sp = pf.summary(p2.PEKARI_2M), pf.summary(pu.PUMBA)
    # Pekari-2M: a ~55 kg confined-space platform, 0.58 m wide, below 0.5 m folded
    assert 45 < s2["mass_empty_kg"] < 70
    assert s2["overall_m"][1] <= 0.60 and s2["overall_m"][2] < 0.5
    # Pumba: WILK-class envelope (1.70 x 1.17 x 0.70 m reported), 100-200 kg on the deck
    w = pu.WILK_REFERENCE
    assert sp["overall_m"][0] <= w["length_m"] and sp["overall_m"][1] <= w["width_m"]
    assert pu.PUMBA.payload_range_kg == (100.0, 200.0)
    assert sp["v_max_m_s"] == pytest.approx(w["v_max_m_s"], rel=0.12)
    # the heavy derivative: wider, longer, heavier; Calydon heavier still, the same envelope class
    sc = pf.summary(cr.CALYDON)
    assert sp["overall_m"][1] / s2["overall_m"][1] > 1.8 and sp["mass_empty_kg"] / s2["mass_empty_kg"] > 3.5
    assert sc["overall_m"][0] <= w["length_m"] and sc["overall_m"][1] <= w["width_m"]
    assert sc["mass_empty_kg"] > 1.6 * sp["mass_empty_kg"]


def test_pumba_chassis_is_wilk_class():
    """Pumba without its arms and stabilizers weighs no more than WILK's reported 180 kg empty; with them it stays
    well under Calydon. The band track replaced the links and pins in the budget."""
    assert pu.chassis_mass() <= pu.WILK_REFERENCE["empty_kg"]
    assert pf.total_mass(pu.PUMBA) < 250.0
    rows = pf.mass_budget(pu.PUMBA)
    assert not any(k.startswith("track pins") for k in rows)
    band = next(v for k, v in rows.items() if k.startswith("track links: rubber band"))
    assert band == pytest.approx(2 * pf.geometry(pu.PUMBA)["belt_length"] * pu.BAND_TRACK_KG_M)
    assert pu.chassis_mass(cr.CALYDON) > 1.5 * pu.chassis_mass()


@pytest.mark.parametrize("pl", PLATFORMS, ids=lambda p: p.name)
def test_cg_balances_folded_and_is_low(pl):
    c = pf.cg(pl)
    g = pf.geometry(pl)
    assert abs(c["x"]) < 0.02                                       # the battery balances the folded platform
    assert c["z"] < 0.5 * pf.capabilities(pl)["height_folded_m"]       # low: under half the folded height
    st = pf.stability(pl, payload_kg=pl.payload_range_kg[1])
    assert st["stable"] and st["tip_deg"] > 30.0


def test_stability_margin_geometry():
    """The margin is the distance from the CG to the nearest polygon edge: the folded, symmetric Pekari-2M sits
    nearer its front or rear edge than its sides (its contact patch is longer than it is wide)."""
    pl = p2.PEKARI_2M
    st = pf.stability(pl)
    g = pf.geometry(pl)
    c = st["cg"]
    expect = min(g["contact_front"] - c["x"], c["x"] - g["contact_rear"], g["half_width"] - abs(c["y"]))
    assert st["margin_m"] == pytest.approx(expect, abs=1e-9)


@pytest.mark.parametrize("pl,r", [(pu.PUMBA, 1.0), (cr.CALYDON, 1.2)], ids=["Pumba", "Calydon"])
def test_stabilizers_raise_the_tipping_load(pl, r):
    for rr in (r, pa.reach(pl.mount("main").arm) * 0.9):
        free = pf.tipping_load(pl, "main", rr, math.pi / 2)
        stab = pf.tipping_load(pl, "main", rr, math.pi / 2, stabilizers_deployed=True)
        assert stab > 1.3 * free
    cap = pf.lift_capacity(pl, "main", r)
    assert cap["capacity_kg"] >= pl.mount("main").arm.rated_payload_kg
    assert cap["limit"] == "arm"                                     # the arm, not the tipping line, limits it


def test_lift_capacity_beyond_reach_is_zero():
    assert pf.lift_capacity(cr.CALYDON, "aux", 1.6)["capacity_kg"] == 0.0
    assert pf.lift_capacity(pu.PUMBA, "aux", 1.2)["capacity_kg"] == 0.0


@pytest.mark.parametrize("pl", PLATFORMS, ids=lambda p: p.name)
def test_drive_climbs_20_degrees_loaded(pl):
    d = pf.drive_check(pl, grade_deg=20.0, v_ms=0.3)
    assert d["ok_peak"] and d["ok_traction"]
    flat = pf.drive_check(pl, grade_deg=0.0, v_ms=pl.v_design * 0.8)
    assert flat["ok_rated"]


@pytest.mark.parametrize("pl", PLATFORMS, ids=lambda p: p.name)
def test_power_and_endurance(pl):
    pw = pf.power(pl)
    assert pw["average_W"] == pytest.approx(0.5 * pw["traction_W"] + 0.3 * pw["arms_W"] + pl.electronics_W)
    assert 2.0 < pf.endurance(pl) < 10.0


def test_capabilities_drive_the_fleet():
    c2, cp = pf.capabilities(p2.PEKARI_2M), pf.capabilities(pu.PUMBA)
    assert {"heavy_lift", "heavy_transport", "dual_arm"} <= pf.capabilities(cr.CALYDON)["tags"]
    assert {"confined", "dual_arm", "manipulate", "thermal"} <= c2["tags"]
    assert {"heavy_lift", "heavy_transport", "dual_arm"} <= cp["tags"] and "confined" not in cp["tags"]
    assert c2["payload_kg"] == 40.0 and cp["payload_kg"] == 200.0


def test_shared_interfaces():
    """One electrical bus, one arm convention, one joint naming for both."""
    assert p2.PEKARI_2M.bus_V == pu.PUMBA.bus_V == cr.CALYDON.bus_V == 48.0
    assert pu.PUMBA.mount("aux").arm is pa.PEKARI_ARM                # one arm module across the family
    for pl in PLATFORMS:
        for m in pl.mounts:
            names = pa.joint_names(m.arm, f"{m.name}_")
            assert names[0] == f"{m.name}_yaw" and names[-1] == f"{m.name}_grip"


# ----------------------------------------------------------------------------------------------- CAD
def test_cad_defaults_match_the_platforms():
    pytest.importorskip("cadquery")
    import calydon
    import pekari_2m
    import pumba

    for cls, pl in ((pekari_2m.Pekari2M, p2.PEKARI_2M), (pumba.Pumba, pu.PUMBA), (calydon.Calydon, cr.CALYDON)):
        p = cls().resolve()
        assert {k: p[k] for k in pl.chassis} == pl.chassis


def test_stored_cad_numbers_match_a_rebuild():
    pytest.importorskip("cadquery")
    for pl in PLATFORMS:
        fresh = pf.cad_numbers(pl, recompute=True)
        for k, v in pl.cad.items():
            if isinstance(v, tuple):
                assert fresh[k] == pytest.approx(v, abs=0.5)
            else:
                assert fresh[k] == pytest.approx(v, rel=0.01)


def test_cad_envelopes():
    pytest.importorskip("cadquery")
    import calydon
    import pekari_2m
    import pumba

    d = pekari_2m.Pekari2M().generate().measure()["dimensions"]
    assert d[1] == pytest.approx(580.0, abs=1.0) and d[2] < 450.0
    d = pumba.Pumba().generate().measure()["dimensions"]
    assert d[0] < 1700.0 and d[1] == pytest.approx(1080.0, abs=1.0)
    d = pumba.Pumba().generate(stabilizers="deployed", arm_pose="deployed").measure()["dimensions"]
    assert d[1] == pytest.approx(2 * 720.0 + 160.0, abs=2.0)          # pads at y = ±0.72 m, 80 mm radius
    d = calydon.Calydon().generate().measure()["dimensions"]
    assert d[0] < 1800.0 and d[1] == pytest.approx(1150.0, abs=1.0)
    d = calydon.Calydon().generate(stabilizers="deployed", arm_pose="deployed").measure()["dimensions"]
    assert d[1] == pytest.approx(2 * 800.0 + 180.0, abs=2.0)          # pads at y = ±0.80 m, 90 mm radius


# ----------------------------------------------------------------------------------------------- Chiron (MuJoCo)
@pytest.mark.parametrize("pl", PLATFORMS, ids=lambda p: p.name)
def test_chiron_model_mass_matches_the_budget(pl):
    pytest.importorskip("mujoco")
    r = pf.robot(pl)
    assert r.total_mass() == pytest.approx(pf.total_mass(pl, pl.payload_kg), rel=1e-6)
    for m in pl.mounts:
        assert all(j in r.nominal_qpos for j in (f"{m.name}_yaw", f"{m.name}_grip_a", f"{m.name}_wrist_roll"))


@pytest.mark.parametrize("pl,pose,loads,stab", [
    (p2.PEKARI_2M, "folded", {}, False),
    (p2.PEKARI_2M, "deployed", {"L": 5.0, "R": 5.0}, False),
    (pu.PUMBA, "deployed", {"main": 30.0, "aux": 5.0}, False),
    (pu.PUMBA, "deployed", {"main": 30.0, "aux": 5.0}, True),
    (cr.CALYDON, "deployed", {"main": 60.0, "aux": 8.0}, False),
    (cr.CALYDON, "deployed", {"main": 60.0, "aux": 8.0}, True),
], ids=["2M-folded", "2M-deployed-5kg", "pumba-deployed-rated", "pumba-stabilized", "calydon-deployed-rated",
        "calydon-stabilized"])
def test_stands_and_holds_the_rated_loads(pl, pose, loads, stab):
    pytest.importorskip("mujoco")
    res = pf.stand_test(pl, pose, loads=loads, stabilizers_deployed=stab, duration=1.0)
    assert res["tilt_deg"] < 0.5 and not res["belly"]
    assert res["arm_err_deg"] < 2.5                                   # the arms hold the rated loads (no feed-forward)
    if stab:
        assert 0.3 < res["track_share"] < 0.95                       # the pads take part of the weight
    else:
        assert res["track_share"] == pytest.approx(1.0, abs=0.02)


@pytest.mark.parametrize("pl", PLATFORMS, ids=lambda p: p.name)
def test_drives_straight_with_the_pekari_controller(pl):
    pytest.importorskip("mujoco")
    from vegeta import chiron as ch
    import pekari_controller as pc

    lab = pf.lab(pl, ch.Flat())
    ep = pc.run(lab, legs=[pc.Leg("straight", 3.0, v=0.8)], duration=5.0)
    ts = pc.timeseries(ep)
    assert ts.v.max() == pytest.approx(0.8, rel=0.1) and abs(ts.yaw_deg.iloc[-1]) < 3.0


@pytest.mark.parametrize("pl", PLATFORMS, ids=lambda p: p.name)
def test_bom_prices_every_drive(pl):
    rows = pf.bom(pl)
    n_arm = sum(len(m.arm.drives) for m in pl.mounts)
    drives = sum(q for item, q, *_ in rows if item.startswith("drive: "))
    assert drives == 2 + n_arm + len(pl.stabilizers)
    lo, hi = pf.bom_total(pl)
    assert 0 < lo < hi
