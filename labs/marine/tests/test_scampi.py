"""Scampi v0: the sizing meets the brief (scampi.md); the CAD builds where CadQuery is."""
import pytest

from scampi import REQUIREMENTS, Scampi


def test_sizing_meets_the_brief():
    s = Scampi.sizing()
    assert 60 <= s["mass_in_air_kg"] <= 120
    assert all(h["buckling_margin"] > 0 and h["hoop_margin"] > 0 for h in s["housings"].values())
    assert s["foam_fits"] and s["net_buoyancy_kg"] > 0
    assert s["station_keeping_current_m_s"] >= REQUIREMENTS["current_m_s"]
    assert REQUIREMENTS["endurance_h"][0] <= s["dive_hours"] <= REQUIREMENTS["endurance_h"][1]
    assert s["battery_margin"] > 0.1


def test_housing_buckling_is_the_long_tube_formula():
    p = {q.name: q.default for q in Scampi.parameters}
    h = Scampi.housing_check(p, 600.0)
    t, D = p["housing_wall"], p["housing_od"]
    assert h["buckling_MPa"] == pytest.approx(2 * 68.9e3 / (1 - 0.33 ** 2) * (t / D) ** 3, rel=1e-9)


def test_cad_builds_and_the_tools_and_folded_arms_stay_above_the_soles():
    pytest.importorskip("cadquery")
    p = {q.name: q.default for q in Scampi.parameters}
    d = Scampi()
    assert d.build(p).val().isValid()
    assert d._tool_skid(p).val().BoundingBox().zmin > 0
    for reach, y, links in ((p["arm_reach"], -0.22 * p["width"], (0.38, 0.32, 0.18, 0.12)),
                            (p["hold_reach"], 0.22 * p["width"], (0.45, 0.40, 0.15))):
        assert d._arm(p, reach, y, links, "folded").val().BoundingBox().zmin > 0
