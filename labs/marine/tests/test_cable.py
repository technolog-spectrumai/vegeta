"""Ketos Cable's machines: the jetting trencher's soil-dependent speed, the cable vessel's carousel, and the export
of Nephro (the cable AUV), the trencher, the vessel, a cable segment and a repair joint."""
import json

import pytest

from subsea_cable import CABLE, SOILS, CableVessel, JetTrencher


def test_trench_rate_falls_with_harder_soil_and_deeper_trench():
    r = JetTrencher.trench_rate
    assert r("silt", 1.5) >= r("sand", 1.5) > r("soft_clay", 1.5) > 0
    assert r("stiff_clay", 1.5) == 0.0 and r("rock", 0.5) == 0.0
    assert r("sand", 3.0) < r("sand", 2.0) < r("sand", 1.0) <= JetTrencher.p()["max_speed_m_h"]
    assert r("sand", 3.5) == 0.0                                    # deeper than the swords
    assert set(JetTrencher.sizing()["trench_m_h"]) == set(SOILS)


def test_trencher_sits_on_its_tracks_and_vessel_carries_a_long_cable():
    s = JetTrencher.sizing()
    assert s["on_tracks_N"] > 0 and s["power_W"] > 9e5 and 1.0 < s["v_max_m_s"] < 2.0
    v = CableVessel.sizing()
    assert 60 < v["carousel_km"] < 100 and CABLE["mbr_m"] > 10 * CABLE["diameter_m"]


def test_cable_export(tmp_path):
    pytest.importorskip("cadquery")
    import ketos_export
    ketos_export.export(str(tmp_path), {"nephro", "trencher", "cable_vessel", "cable", "cable_joint"})
    n = json.load(open(tmp_path / "nephro.card.json"))
    assert n["kind"] == "rov" and n["tether"] is None and n["battery_Wh"] == 8000 and len(n["thrusters"]) == 6
    assert n["tracker"]["dob_max_m"] == 3.0 and 60 <= n["range_km"] <= 120
    t = json.load(open(tmp_path / "trencher.card.json"))
    assert t["kind"] == "trencher" and t["power"] == "umbilical" and t["net_buoyancy_kg"] < 0 and len(t["thrusters"]) == 8
    assert t["trench_m_h"]["sand"]["1.5"] > t["trench_m_h"]["soft_clay"]["1.5"] > 0
    v = json.load(open(tmp_path / "cable_vessel.card.json"))
    assert v["kind"] == "boat" and v["chute"][0] < -60 and v["lars"]["overboard"][1] < -14
    c = json.load(open(tmp_path / "cable.card.json"))
    assert c["kind"] == "cable" and c["diameter_m"] == 0.24 and c["segment_m"] == 10.0
    j = json.load(open(tmp_path / "cable_joint.card.json"))
    assert j["kind"] == "joint" and j["joint_h"] > 24
    for card in (n, t, v, c, j):
        for part in card["parts"]:
            m = json.load(open(tmp_path / (part["model"] + ".json")))
            assert len(m["vertices"]) % 3 == 0 and max(m["triangles"]) < len(m["vertices"]) // 3
