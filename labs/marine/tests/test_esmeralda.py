"""Esmeralda, Zarco's guard variant: Zarco's hull, propulsion and flight deck unchanged; the harvest gear gone; the
boat fits its bay and the ramp can haul it; it floats, trims and stays stiff; the gensets carry the peak; the hangar
holds the guard drones; the effectors and the turret are physical; the CAD builds."""
import pytest

import zarco
from esmeralda import EFFECTORS, FLARE_LAUNCHER, REMOVED, ROBOTS, SONAR_OPTION, Esmeralda


@pytest.fixture(scope="module")
def s():
    return Esmeralda.sizing()


def test_zarco_kept_where_it_should_be(s):
    z, p = zarco.Zarco.sizing(), Esmeralda.p()
    for k in ("length_m", "beam_m", "displacement_t", "service_kn", "service_kW", "max_kn"):
        assert s[k] == z[k], k
    for k in ("pads", "pad_cycle_s", "catapult_cycle_s", "net_cycle_s", "deck_hs_limit", "gensets_kw", "pods_kw", "battery_kwh"):
        assert p[k] == zarco.Zarco.p()[k], k


def test_harvest_gear_removed(s):
    for k in REMOVED:
        assert k not in s["weights"]
    assert not {"davit", "basket_gantry", "packing_cell"} & set(ROBOTS) and "boat_ramp" in ROBOTS
    assert s["changed_vs_zarco"]["davits"] == [4, 0]
    assert not any("processing" in k or "reels" in k for k in s["price"])


def test_boat_fits_and_is_hauled(s):
    b = s["bay"]
    assert b["boat_fits"] and b["ramp_fits_tunnel"] and b["clear_height_over_cradle_m"] >= b["boat_needs_m"]
    assert b["boat_mast_up_top_m"] > b["flight_deck_underside_m"]           # so the mast must fold
    r = s["boat"]["ramp"]
    assert r["max_boat_kg"] > s["boat"]["mass_kg"] and r["boat_transom_keel_z_m"] > 0     # stowed clear of the water
    assert s["boat"]["hs_limit_m"] >= 1.5


def test_floats_trims_and_is_stiff(s):
    t = s["trim"]
    assert t["draught_m"] < Esmeralda.p()["draught"] / 1000 and abs(t["trim_deg"]) < 1.0 and t["gm_t_m"] > 5
    assert s["zarco_trim"]["trim_deg"] == 0.0 and abs(s["level_by_moving_fuel_m"]) < 5


def test_power_and_hangar(s):
    assert s["gensets_margin_at_peak"] > 0.3 and s["endurance_days"] > 10
    h = s["hangar"]
    assert h["gannet_fits"] and h["fulmar_fits"] and s["chargers"] >= 4


def test_effectors_and_sensors(s):
    e = s["effectors"]
    assert e["hailer"]["intelligible_m"]["skiff_at_speed"] > 200 and e["searchlights"]["range_m"] > 2000
    assert 50 < e["water_cannon"]["throw_m"] < 150 and e["water_cannon"]["sea_water"]
    assert e["flare_launcher"] == FLARE_LAUNCHER and FLARE_LAUNCHER["fitted"] is False and SONAR_OPTION["fitted"] is False
    t = s["sensors"]["turret"]["dri_skiff_km"]
    assert t["night_clear"]["identification"] > t["night_haze"]["identification"] > t["night_fog"]["detection"]
    r = s["sensors"]["radars"]["X"]["skiff_km_by_hs"]
    assert r[0.5] > r[2.5]
    assert EFFECTORS["water_cannon"]["flow_l_min"] > 1500                      # more than Barracuda's


def test_cad_builds():
    pytest.importorskip("cadquery")
    assert Esmeralda().build(Esmeralda.p()).val().isValid()
