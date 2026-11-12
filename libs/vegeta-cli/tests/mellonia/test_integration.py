"""PrusaSlicer integration tests checked against geometry, not just the exit code."""
import math

import pytest

from vegeta.mellonia import Orientation, read_gcode, slice_stl
from vegeta.mellonia.examples import GENERIC_PLA_0_2MM as PLA

pytestmark = pytest.mark.requires_prusaslicer


def test_cube_height_and_layer_count(tmp_path, cube_stl):
    s = PLA.replace(print={"layer_height": 0.2, "first_layer_height": 0.3})
    r = slice_stl(cube_stl, s, Orientation(), tmp_path)
    assert r.ok, r.messages
    m = r.metrics
    # layers are quantised: first layer + (n - 1) regular layers, top within half a layer of 20 mm
    n = m["layer_count"]
    assert m["max_z"] == pytest.approx(0.3 + (n - 1) * 0.2, abs=1e-6)
    assert abs(m["max_z"] - 20.0) <= 0.1 + 1e-6
    assert n == 1 + math.ceil((20.0 - 0.3) / 0.2 - 1e-9)
    assert m["estimated_time_s"] > 0 and m["filament_used_mm"] > 0
    assert m["filament_used_g"] == pytest.approx(m["filament_used_cm3"] * 1.24, rel=0.02)
    for key in ("gcode", "effective_config", "printer_ini", "filament_ini", "print_ini", "log", "input_stl"):
        assert r.artifacts[key].is_file(), key
    assert r.execution[0].command[0] == "prusa-slicer" and "--export-gcode" in r.execution[0].command
    info = read_gcode(r.artifacts["gcode"])
    assert info.layer_extrusion_mm.sum() == pytest.approx(m["filament_used_mm"], rel=0.005)


def test_solid_cube_material_matches_volume(tmp_path, cube_stl):
    s = PLA.replace(print={"fill_density": "100%", "fill_pattern": "rectilinear", "skirts": 0})
    r = slice_stl(cube_stl, s, Orientation(), tmp_path)
    assert r.ok, r.messages
    assert r.metrics["filament_used_cm3"] == pytest.approx(8.0, rel=0.03)  # 20 mm cube = 8 cm^3


def test_engineer_orientation_changes_layers(tmp_path, tall_stl):
    upright = slice_stl(tall_stl, PLA, Orientation(), tmp_path / "up")
    lying = slice_stl(tall_stl, PLA, Orientation(rotate_x=90), tmp_path / "lying")
    assert upright.metrics["layer_count"] == 200 and upright.metrics["max_z"] == pytest.approx(40)
    assert lying.metrics["layer_count"] == 50 and lying.metrics["max_z"] == pytest.approx(10)
    assert lying.metrics["estimated_time_s"] != upright.metrics["estimated_time_s"]


def test_misspelled_setting_is_not_silently_ignored(tmp_path, cube_stl):
    r = slice_stl(cube_stl, PLA.replace(print={"layer_heigth": 0.3}), Orientation(), tmp_path)
    assert r.status == "failed" and "layer_heigth" in r.messages[0]
    assert r.artifacts["gcode"].is_file()  # output is kept for inspection


def test_invalid_combination_reports_prusaslicer_error(tmp_path, cube_stl):
    r = slice_stl(cube_stl, PLA.replace(print={"fill_density": "100%", "fill_pattern": "gyroid"}), Orientation(), tmp_path)
    assert r.status == "failed" and "100% density" in r.messages[0]
