"""The Ketos export: every farm machine gives meshes in metres and a physics card Ketos can read."""
import json
import os

import pytest

cq = pytest.importorskip("cadquery")

import ketos_export  # noqa: E402
from feed_barge import FeedBarge  # noqa: E402
from salmon_pen import SalmonPen  # noqa: E402


def test_pen_and_barge_sizing():
    s = SalmonPen.sizing()
    assert 25 < s["radius_m"] < 26 and 5000 < s["net_area_m2"] < 7000
    b = FeedBarge.sizing()
    assert b["dock_positions"] == 3 and b["displacement_t"] > 500


def test_export_writes_cards_and_meshes(tmp_path):
    ketos_export.export(str(tmp_path), {"shrimp", "pen"})
    card = json.load(open(tmp_path / "shrimp.card.json"))
    assert card["schema"] == "ketos.card" and card["kind"] == "rov" and len(card["thrusters"]) == 6
    assert card["tether"]["length_m"] == 50.0
    for part in card["parts"]:
        m = json.load(open(tmp_path / (part["model"] + ".json")))
        assert len(m["vertices"]) % 3 == 0 and max(m["triangles"]) < len(m["vertices"]) // 3
        assert max(m["size"]) < 1.0                       # metres: a 36 cm craft
    pen = json.load(open(tmp_path / "pen_collar.json"))
    assert 50 < pen["size"][0] < 55                     # the collar's outer diameter


def _meshes(tmp_path, card):
    out = []
    for part in card["parts"]:
        m = json.load(open(tmp_path / (part["model"] + ".json")))
        n = len(m["vertices"]) // 3
        assert len(m["vertices"]) % 3 == 0 and len(m["triangles"]) % 3 == 0
        assert min(m["triangles"]) >= 0 and max(m["triangles"]) < n
        out.append(m)
    return out


def test_scampi_and_echensis_export(tmp_path):
    ketos_export.export(str(tmp_path), {"scampi", "echensis"})
    sc = json.load(open(tmp_path / "scampi.card.json"))
    cy_keys = {"schema", "version", "name", "kind", "parts", "size", "mass_kg", "displacement_L", "net_buoyancy_kg", "cd",
               "thrusters", "battery_Wh", "hotel_W", "max_depth_m", "inspect_speed_m_s", "camera_swath_m", "tools", "tether",
               "resident", "price_eur"}
    assert set(sc) == cy_keys and sc["kind"] == "rov" and sc["tether"] is None and sc["resident"] is True
    assert len(sc["thrusters"]) == 8 and 60 <= sc["mass_kg"] <= 120 and sc["size"] == [1.5, 0.8, 0.56]
    zs = [z for m in _meshes(tmp_path, sc) for z in m["vertices"][2::3]]
    assert abs(max(zs) + min(zs)) < 0.01                 # the pose at its middle height
    ec = json.load(open(tmp_path / "echensis.card.json"))
    assert ec["kind"] == "crawler" and ec["probes"] == ["ut", "cp"] and 0 < ec["capture_efficiency"] <= 1
    assert ec["adhesion_N"] > 5000 and ec["tether"]["length_m"] == 60.0 and 50 < ec["mass_kg"] < 150
    ms = _meshes(tmp_path, ec)
    assert max(max(m["size"]) for m in ms) < 2.0 and min(z for m in ms for z in m["vertices"][2::3]) >= 0   # on the hull


def test_ship_hull_and_wellboat_export(tmp_path):
    ketos_export.export(str(tmp_path), {"ship_hull", "wellboat"})
    sh = json.load(open(tmp_path / "ship_hull.card.json"))
    assert sh["kind"] == "hull" and sh["wetted_area_m2"] == pytest.approx(1.7 * 120 * 7.5 + 0.78 * 120 * 20)
    (m,) = _meshes(tmp_path, sh)
    assert len(m["triangles"]) // 3 <= 6000 and m["size"] == pytest.approx([120.0, 20.0, 11.0], abs=0.05)
    zs = m["vertices"][2::3]
    assert min(zs) == pytest.approx(-7.5, abs=0.05) and max(zs) == pytest.approx(3.5, abs=0.05)   # z = 0 the waterline
    wb = json.load(open(tmp_path / "wellboat.card.json"))
    assert wb["kind"] == "boat" and 2 <= len(wb["parts"]) <= 4 and wb["max_speed_m_s"] == 6.0 and wb["wells_m3"] > 2000
    ms = _meshes(tmp_path, wb)
    xs = [x for m in ms for x in m["vertices"][0::3]]
    assert max(xs) - min(xs) == pytest.approx(75.0, abs=0.1) and abs(max(xs) + min(xs)) < 0.1


def test_merchant_hull_block_coefficient():
    from merchant_hull import MerchantHull
    assert MerchantHull.sizing()["cb_measured"] == pytest.approx(0.78, abs=0.03)


def test_oil_field_export(tmp_path):
    """Ketos Oil's machines: the work-class ROV, the jacket (its structure as data), the DP vessel, the TMS, the manifold."""
    from oil_platform import Jacket
    g = Jacket.members()
    kinds = {m["kind"] for m in g["members"]}
    assert {"leg", "brace", "horizontal", "conductor", "riser", "caisson"} <= kinds
    assert g["levels"][-1] == -110.0 and len([m for m in g["members"] if m["kind"] == "leg"]) == 16
    assert all(a["p"][2] < 0 for a in g["anodes"]) and len(g["anodes"]) > 40
    ketos_export.export(str(tmp_path), {"homarus", "jacket", "dp_vessel", "tms", "manifold"})
    h = json.load(open(tmp_path / "homarus.card.json"))
    assert h["kind"] == "rov" and h["power"] == "umbilical" and len(h["thrusters"]) == 7 and 3000 < h["mass_kg"] < 5000
    assert h["tether"]["from"] == "tms" and h["tether"]["length_m"] == 250.0
    zs = [z for m in _meshes(tmp_path, h) for z in m["vertices"][2::3]]
    assert abs(max(zs) + min(zs)) < 0.01
    j = json.load(open(tmp_path / "jacket.card.json"))
    js = _meshes(tmp_path, j)
    assert min(z for m in js for z in m["vertices"][2::3]) <= -109.9 and j["members"] == g["members"]
    v = json.load(open(tmp_path / "dp_vessel.card.json"))
    assert v["lars"]["max_hs_m"] == 3.0 and v["lars"]["overboard"][1] < -10 and 80 < _meshes(tmp_path, v)[0]["size"][0] < 95
    m = json.load(open(tmp_path / "manifold.card.json"))
    assert len(m["valves"]) == 6 and len(m["hot_stabs"]) == 2
    assert json.load(open(tmp_path / "tms.card.json"))["tether"]["length_m"] == 250.0
