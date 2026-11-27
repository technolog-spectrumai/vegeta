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
               "resident", "price_eur", "effectors"}
    assert set(sc) == cy_keys and sc["kind"] == "rov" and sc["tether"] is None and sc["resident"] is True
    assert len(sc["thrusters"]) == 8 and 60 <= sc["mass_kg"] <= 121 and sc["size"] == [1.5, 0.8, 0.56]
    ef = sc["effectors"]                                 # two grippers (one per arm) and the quick-change cutter
    assert {ef["gripper_l"]["arm"], ef["gripper_r"]["arm"]} == {"hold", "work"} and ef["cutter"]["arm"] == "work"
    assert sc["tools"]["gripper"]["count"] == 2 and sc["tools"]["cutter"]["count"] == 1
    assert ef["cutter"]["max_cut_mm"]["soft_line"] >= 20 and ef["cutter"]["cut_force_kN"] == 20.0
    for k in ("model", "jaw_model"):
        assert (tmp_path / (ef["gripper_l"][k] + ".json")).exists()
    for k in ("model", "blade_model"):
        assert (tmp_path / (ef["cutter"][k] + ".json")).exists()
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


def test_oil_stage3_export(tmp_path):
    ketos_export.export(str(tmp_path), {"squilla", "resident_dock", "riser_clamp", "debris"})
    sq = json.load(open(tmp_path / "squilla.card.json"))
    assert sq["kind"] == "rov" and sq["resident"] is True and sq["tether"] is None and sq["battery_Wh"] == 4500
    assert len(sq["thrusters"]) == 10 and 100 < sq["mass_kg"] < 200
    _meshes(tmp_path, sq)
    dock = json.load(open(tmp_path / "resident_dock.card.json"))
    assert dock["kind"] == "dock" and dock["charge_W"] > 0 and dock["depth_m"] == 80.0
    assert dock["leg"] in dock["legs_that_fit_at_depth"] and dock["mount"]["collars"] == 2 and dock["docking"]["dock_s"] > 0
    deb = json.load(open(tmp_path / "debris.card.json"))
    assert set(deb["kinds"]) == {"drum", "tube", "coil", "grating"} and len(_meshes(tmp_path, deb)) == 4
    _meshes(tmp_path, json.load(open(tmp_path / "riser_clamp.card.json")))


def test_walrus_clams_export(tmp_path):
    ketos_export.export(str(tmp_path), {"walrus", "clam_basket", "clam_boat"})
    w = json.load(open(tmp_path / "walrus.card.json"))
    assert len(w["thrusters"]) == 8 and w["thruster_layout"] == "C"
    assert w["kind"] == "crawler" and w["power"] == "tether" and w["battery_Wh"] is None and w["pickers"] == 2
    assert w["basket"]["kg"] == 20.0 and w["tether"]["length_m"] == 150.0 and 30 < w["mass_kg"] < 40
    assert w["rate_per_h"] == 1200 and w["price_eur"] < 30600
    ms = _meshes(tmp_path, w)
    assert len(ms) == 8 and max(max(m["size"]) for m in ms) < 1.2 and min(z for m in ms for z in m["vertices"][2::3]) >= -0.05   # the paddles reach 2 cm into the sand
    b = json.load(open(tmp_path / "clam_basket.card.json"))
    assert b["capacity_kg"] == 20.0 and _meshes(tmp_path, b)
    boat = json.load(open(tmp_path / "clam_boat.card.json"))
    assert boat["carries"] == ["walrus"] and boat["parts"][0]["model"] == "workboat_hull"


def test_carrier_and_drones_export(tmp_path):
    ketos_export.export(str(tmp_path), {"zarco", "gannet", "fulmar"})
    c = json.load(open(tmp_path / "zarco.card.json"))
    assert c["kind"] == "boat" and c["autonomous"] and c["crew"] == 0 and len(c["flight_deck"]["pads"]) == 4
    assert c["haul_min_per_basket"] < 3.0 and c["packing"]["crate_kg"] == 5.0 and "colregs" in c["autonomy"]
    assert c["unmanned"] and {"swap_robot", "crate_loader", "basket_gantry"} <= set(c["robots"])
    assert c["davit"]["compensated"] and c["davit_hs_limit_m"] == c["davit"]["hs_limit_m"] > c["davit"]["hs_limit_uncompensated_m"]
    _meshes(tmp_path, c)
    for name, kind in (("gannet", "vtol"), ("fulmar", "fixed_wing")):
        a = json.load(open(tmp_path / f"{name}.card.json"))
        assert a["kind"] == "air" and a["air"] == kind and a["payload_kg"] == 5.0
        for k in ("mass_kg", "cruise_m_s", "range_km", "battery_Wh", "power_cruise_W", "wind_limit_m_s", "deck_hs_limit_m"):
            assert a[k] and a[k] > 0, k
        _meshes(tmp_path, a)
    g = json.load(open(tmp_path / "gannet.card.json"))
    assert g["power_hover_W"] > g["power_cruise_W"] and g["recovery"] == "pad"
    f = json.load(open(tmp_path / "fulmar.card.json"))
    assert f["power_hover_W"] is None and f["launch"] == "catapult" and f["wh_per_km"] < g["wh_per_km"]


def test_barracuda_export(tmp_path):
    ketos_export.export(str(tmp_path), {"barracuda"})
    c = json.load(open(tmp_path / "barracuda.card.json"))
    assert c["kind"] == "boat" and c["role"] == "interceptor" and c["unmanned"] and c["crew"] == 0
    assert c["max_kn"] >= 35 and c["range_nm"] >= 150 and c["loiter_h"] >= 10 and c["class_values"]
    assert [r["hs_m"] for r in c["speed_vs_hs"]] == [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    assert {"hailer", "searchlight", "strobe", "water_cannon"} <= set(c["effectors"])
    assert {"eoir", "radar", "ais", "microphones"} <= set(c["sensors"]) and c["launch_recovery"]["hs_limit_m"] > 1
    ms = _meshes(tmp_path, c)
    zs = [z for m in ms for z in m["vertices"][2::3]]
    xs = [x for m in ms for x in m["vertices"][0::3]]
    assert -0.8 < min(zs) < -0.5 and 4.0 < max(zs) < 5.5 and abs(max(xs) + min(xs)) < 0.01   # z = 0 the waterline


def test_esmeralda_export(tmp_path):
    ketos_export.export(str(tmp_path), {"esmeralda", "zarco"})
    e = json.load(open(tmp_path / "esmeralda.card.json"))
    z = json.load(open(tmp_path / "zarco.card.json"))
    assert e["kind"] == "boat" and e["variant_of"] == "zarco" and e["unmanned"] and e["crew"] == 0
    assert "barracuda" in e["carries"] and "walrus" not in e["carries"] and "davit" not in e["robots"]
    assert e["service_kn"] == z["service_kn"] and e["flight_deck"]["pads"] == z["flight_deck"]["pads"]
    assert e["flight_deck"]["hangar"] == {"gannet": 4, "fulmar": 2} and e["sea_state_limits"]["deck_hs_m"] == 2.0
    assert {"hailer", "searchlights", "water_cannon", "flare_launcher"} <= set(e["effectors"]) and "turret" in e["sensors"]
    assert set(e["changed_vs_zarco"]) >= {"mass_t", "draught_m", "trim_deg", "price_eur"}
    _meshes(tmp_path, e)


def test_drone_payloads_export(tmp_path):
    ketos_export.export(str(tmp_path), {"gannet", "fulmar"})
    for name in ("gannet", "fulmar"):
        c = json.load(open(tmp_path / f"{name}.card.json"))
        cargo, eoir = c["payloads"]["cargo"], c["payloads"]["eoir"]
        assert cargo["payload_kg"] == c["payload_kg"] == 5.0 and cargo["range_km"] == c["range_km"]     # the cargo numbers kept
        assert 0 < eoir["payload_kg"] < 5.0 and eoir["pod"]["power_W"] > 0
        assert set(eoir["dri_km"]) == {"skiff_6m", "fishing_20m"} and eoir["datalink"] and eoir["fov"]["ir"]["pixels_h"] == 640
        for m in eoir["parts"]:
            assert (tmp_path / f"{m}.json").exists()
    assert json.load(open(tmp_path / "fulmar.card.json"))["payloads"]["eoir"]["radar_option"]["fits_with_eoir"] is False


def test_small_craft_export(tmp_path):
    names = {"fishing_boat", "pilot_boat", "yacht", "skiff"}
    ketos_export.export(str(tmp_path), names)
    for n in names:
        c = json.load(open(tmp_path / f"{n}.card.json"))
        assert c["kind"] == "craft" and c["weapons"] is None and c["class_values"] and c["basis"]
        assert [r["hs_m"] for r in c["speed_vs_hs"]] == [0.0, 0.5, 1.0, 1.5, 2.0, 2.5] and {"beam", "bow", "stern"} <= set(c["rcs_m2"])
        ms = _meshes(tmp_path, c)
        xs = [x for m in ms for x in m["vertices"][0::3]]
        assert abs(max(xs) + min(xs)) < 0.01 and min(z for m in ms for z in m["vertices"][2::3]) < 0   # under the waterline
