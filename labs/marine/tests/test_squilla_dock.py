"""Squilla's dock on the jacket: it fits leg A1 at -80 m with room, not the riser leg A2, not at the frames' levels;
the collars hold it with margin; Squilla charges in hours and patrols for longer; docking takes minutes and holds a
cross current; the CAD builds with the collars."""
import pytest

import squilla_dock as sd


def test_fits_at_80_but_not_on_the_riser_leg():
    r = sd.report()
    assert "A1" in r["fits_at_80_on"] and "A2" not in r["fits_at_80_on"]
    a1 = sd.clearance("A1", -80.0)
    assert a1["fits"] and a1["clearance_m"] > 1.0
    assert not sd.clearance("A1", -65.0)["fits"] and not sd.clearance("A1", -35.0)["fits"]       # the horizontal frames' levels
    assert sd.clearance("A2", -80.0)["nearest"].startswith("R-")


def test_clamp_and_services():
    m = sd.mounting(80.0)
    assert m["bolts_per_collar"] >= 4 and m["bolts_per_collar"] * 280 >= m["normal_force_per_collar_kN"] and m["no_welding"]
    s = sd.services()
    assert 1 < s["charger"]["charge_h"] < 4 and s["patrol"]["hours_on_a_charge"] > s["charger"]["charge_h"]
    d = sd.docking()
    assert d["dock_s"] < 600 and 0.2 < d["cross_current_limit_m_s"] < 1.0
    assert d["orbital_at_depth_m_s"]["-80m"] < d["orbital_at_depth_m_s"]["-30m"]


def test_cad_with_collars():
    pytest.importorskip("cadquery")
    from oil_platform import ResidentDock
    w = ResidentDock().build(ResidentDock.p())
    assert w.val().isValid() and w.val().BoundingBox().xlen > 4.5 * 1000
