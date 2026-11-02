"""Echensis v0: the grip holds with margin on steel, the cleaning rate does a bulker in one port call, the capture
pump and the tether stay modest; the CAD builds where CadQuery is."""
import pytest

from echensis import REQUIREMENTS, Echensis


def test_grip_cleaning_capture_and_power():
    s = Echensis.sizing()
    assert all(g["margin"] > 0 for g in s["grip"].values())
    assert s["grip"]["vertical side, crawling up"]["margin"] > 1.0
    assert s["hulls"]["Supramax bulker 190 m"]["robots_for_window"] == 1
    assert s["hose_speed_m_s"] < 2.5 and s["pump_W"] < 2500
    assert s["power_W"]["total"] < 5000 and s["tether_current_A"] < 15


def test_cavitation_head_is_slower_and_needs_the_quay_pump():
    b, c = Echensis.sizing(), Echensis.sizing({"head_type": "cavitation"})
    assert c["clean_rate_m2_h"] < 0.5 * b["clean_rate_m2_h"] and c["quay_W"] > 5000 and b["quay_W"] == 0
    assert all(g["margin"] > 0 for g in c["grip"].values())


def test_magnets_carry_the_steel_cases():
    weak = Echensis.sizing({"magnet_N_per_m": 0.0})
    assert weak["grip"]["vertical side, crawling up"]["margin"] < 0     # suction alone cannot brush in 3 kn


def test_cad_builds():
    pytest.importorskip("cadquery")
    for head in ("brushes", "cavitation"):
        assert Echensis().build(Echensis.p({"head_type": head})).val().isValid()
