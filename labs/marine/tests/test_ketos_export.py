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
