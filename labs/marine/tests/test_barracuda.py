"""Barracuda, the interceptor: the mass closes, it meets the brief's speeds, range and loiter, its speed falls with
the sea state for the stated reason, the manoeuvres and the sensors are physical, the carrier ramp can haul it, the
CAD builds."""
import pytest

from barracuda import DRAUGHT_CAD_MM, ENGINE_STEPS, REQUIREMENTS, Barracuda
import sensing


@pytest.fixture(scope="module")
def s():
    return Barracuda.sizing()


def test_meets_the_brief(s):
    assert all(s["meets"].values()), s["meets"]
    assert s["sprint_kn_full_load"] >= REQUIREMENTS["sprint_kn"] and s["cruise_kn_at_hs1"] >= REQUIREMENTS["cruise_kn"]
    assert s["range_nm_at_cruise"] >= REQUIREMENTS["range_nm"] and s["loiter_h"] >= REQUIREMENTS["loiter_h"]
    assert 8.0 <= s["length_overall_m"] <= 9.0


def test_mass_closes_and_engines_are_the_smallest_that_do(s):
    p = Barracuda.p()
    m = s["mass_full_kg"]
    assert 2500 < m < 6000 and s["engine_kW"] in ENGINE_STEPS
    need = Barracuda.shaft_kW(p, m, p["sprint_kn"], 0.0, Barracuda.hotel_W())
    assert 2 * s["engine_kW"] * 0.95 >= need
    smaller = [e for e in ENGINE_STEPS if e < s["engine_kW"]]
    assert not smaller or 2 * smaller[-1] * 0.95 < need
    assert abs(s["draught_m"] * 1000 - DRAUGHT_CAD_MM) < 60          # the CAD's waterline is the sizing's


def test_savitsky_is_sane():
    p = Barracuda.p()
    r = Barracuda.savitsky(p, 3400, 35 * 0.5144)
    assert r["valid"] and 1.5 < r["trim_deg"] < 6 and 0.05 < r["drag_N"] / (3400 * 9.81) < 0.2


def test_speed_falls_with_the_sea(s):
    t = s["speed_vs_hs"]
    for a, b in zip(t, t[1:]):
        assert b["sustained_kn"] <= a["sustained_kn"] and b["sprint_kn"] <= a["sprint_kn"] and b["power_kn"] <= a["power_kn"]
        assert b["sustained_kn"] <= b["sprint_kn"] and b["at_crewed_1_5g_kn"] <= b["sustained_kn"]
    assert t[-1]["hs_m"] == 2.5 and t[-1]["sustained_limited_by"] == "slamming"


def test_manoeuvres_and_blocking(s):
    m = s["manoeuvres"]
    assert 3 < m["accel_0_30kn_s"] < 20 and 2 < m["crash_stop_lengths"] < 15
    d = [v["diameter_m"] for v in m["turning_circle"].values()]
    assert d == sorted(d) and d[0] >= 2 * s["length_overall_m"]
    assert s["blocking"]["safe_closing_kn"] > 2


def test_sensors_and_effectors(s):
    dri = s["sensors"]["eoir"]["dri_skiff_km"]
    for k, v in dri.items():
        assert v["detection"] >= v["recognition"] >= v["identification"] > 0, k
    assert dri["night_fog"]["detection"] < dri["night_haze"]["detection"] <= dri["night_clear"]["detection"]
    assert dri["day_clear"]["detection"] <= sensing.horizon_km(Barracuda.p()["mast_h"], 1.2) + 0.01
    r = s["sensors"]["radar"]["skiff_km_by_hs"]
    assert r[0.5] > r[2.5] > 0
    e = s["effectors"]
    h = e["hailer"]["intelligible_m"]
    assert h["skiff_at_speed"] < h["skiff_idling"] < h["quiet_deck"]
    assert 30 < e["water_cannon"]["throw_m"] < 100 and e["water_cannon"]["sea_water"]
    assert e["strobe"]["laser"] is False and e["searchlight"]["range_m"] > 500


def test_carrier_ramp_can_take_it(s):
    c = s["carrier"]
    assert c["max_boat_kg"] > s["mass_full_kg"] and 1.0 <= c["hs_limit_m"] <= 3.0
    assert c["recovery_s"] > c["launch_s"] > 0 and c["ramp_width_m"] > s["beam_overall_m"]


def test_cad_builds():
    pytest.importorskip("cadquery")
    d, p = Barracuda(), Barracuda.p()
    assert d.build(p).val().isValid()
    bb = d.build(p).val().BoundingBox()
    assert 8000 < bb.xmax - bb.xmin < 9500
