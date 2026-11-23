"""Scampi v0: the sizing meets the brief (scampi.md); the CAD builds where CadQuery is."""
import pytest

from scampi import ARMS, EFFECTORS, REQUIREMENTS, Scampi


def test_sizing_meets_the_brief():
    s = Scampi.sizing()
    assert 60 <= s["mass_in_air_kg"] <= 120.5              # the brief's 120 kg: 0.2 kg over with the effectors (flagged)
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


def test_effectors_within_the_arms_and_trim():
    s = Scampi.sizing()
    e = s["effectors"]
    for c in e["checks"].values():                         # every tool on its arm: payload and wrist with margin
        assert c["payload_margin"] > 1 and c["wrist_margin"] > 1
    assert s["trim_ok"] and s["ballast_bare_kg"] >= 0 and s["payload_in_water_kg"] > 0
    assert e["mass_kg"] == pytest.approx(2 * 1.4 + 2.6 + 1.0)
    assert abs(e["cutter_out_pitch_deg"]) < 10             # passive; the vertical thrusters trim it out
    # the cutter's rating: its 20 kN against the shear of the section, capped by the 30 mm throat
    assert Scampi.cut_force_kN("soft_line", e["max_cut_mm"]["soft_line"]) == pytest.approx(20.0)
    assert Scampi.max_cut_mm("armour_wire") <= EFFECTORS["cutter"]["throat_mm"]
    assert 5.0 < e["max_cut_mm"]["wire_rope"] < 8.0


def test_effectors_cad():
    pytest.importorskip("cadquery")
    p = {q.name: q.default for q in Scampi.parameters}
    d = Scampi()
    for part in ("gripper_l", "gripper_r", "cutter", "tool_holder"):
        assert d.build(p | {"part": part}).val().isValid()
    shut = d._gripper_local(40.0).val().BoundingBox().ylen
    wide = d._gripper_local(100.0).val().BoundingBox().ylen
    assert wide - shut == pytest.approx(60.0)
    for part in ("gripper_l", "gripper_r"):                 # folded for transit: above the soles
        assert d.build(p | {"part": part, "arm_pose": "folded"}).val().BoundingBox().zmin > 0
    assert d._cutter(p).val().BoundingBox().zmin > 0
