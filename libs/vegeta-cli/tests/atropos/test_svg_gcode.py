import xml.etree.ElementTree as ET

import pytest

from vegeta.atropos import (Part, cut_job, estimate_time, nest, polyline, read_gcode, read_svg_paths, rect, write_gcode,
                            write_svg)

from .test_nest import OPS, SETTINGS, SHEET, ribs


def test_svg_round_trip(tmp_path):
    parts = ribs(8)
    n = nest(parts, SETTINGS)
    on = n.parts_on(0)
    on[0].scores.append(polyline([[10, 10], [30, 10]]))
    svg = write_svg(tmp_path / "s.svg", on, SHEET.width_mm, SHEET.height_mm, OPS, labels=True)
    root = ET.parse(svg).getroot()
    assert root.get("width") == "400mm" and root.get("viewBox") == "0 0 400 100"
    ids = [g.get("id") for g in root.findall("{http://www.w3.org/2000/svg}g")]
    assert ids == ["engrave", "score", "cut"]           # the operations' order
    d = read_svg_paths(svg)
    assert d["groups"]["cut"] == sum(1 + len(p.holes) for p in on) and d["groups"]["score"] == 1
    assert d["length_mm"]["cut"] == pytest.approx(sum(p.cut_length() for p in on), rel=1e-4)
    assert d["length_mm"]["score"] == pytest.approx(20.0, abs=1e-6)
    # y flipped back: the first part's bounding box survives
    back = [p for p in d["paths"] if p["part"] == on[0].name and p["closed"]]
    import numpy as np
    pts = np.vstack([p["points"] for p in back])
    x0, y0, x1, y1 = on[0].bbox()
    assert pts[:, 0].min() == pytest.approx(x0, abs=1e-3) and pts[:, 1].max() == pytest.approx(y1, abs=1e-3)
    assert any(el.text == on[0].name for el in root.iter("{http://www.w3.org/2000/svg}text"))


def test_gcode_lengths_powers_and_time(tmp_path):
    parts = ribs(6)
    n = nest(parts, SETTINGS)
    on = n.parts_on(0)
    job = write_gcode(tmp_path / "s.gcode", on, SETTINGS)
    text = (tmp_path / "s.gcode").read_text()
    assert text.startswith(";") and "G21 G90" in text and text.rstrip().endswith("M2")
    assert "M4 S1000" in text and "F600" in text
    cut_len = sum(p.cut_length() for p in on)
    assert job.cut_length_mm["cut"] == pytest.approx(2 * cut_len, rel=1e-9)              # two passes
    assert job.time_s["cut"] == pytest.approx(2 * cut_len / 600 * 60, rel=1e-9)
    assert job.time_s["travel"] == pytest.approx(job.travel_length_mm / 6000 * 60, rel=1e-9)
    assert estimate_time(on, SETTINGS) == pytest.approx(job.time_s)
    d = read_gcode(tmp_path / "s.gcode")
    assert d["g1_length_mm"] == pytest.approx(2 * cut_len, rel=1e-3)
    assert d["max_s"] == 1000 and d["feeds"] == [600.0, 6000.0]            # the cut feed and the rapid feed
    assert d["cut_time_s"] == pytest.approx(job.time_s["cut"], rel=1e-3)
    bx0, by0, bx1, by1 = d["bbox"]
    assert bx0 >= 0 and by0 >= 0 and bx1 <= SETTINGS.machine.bed_width_mm and by1 <= SETTINGS.machine.bed_height_mm
    assert (bx0, by0, bx1, by1) == pytest.approx(job.bbox, abs=1e-3)
    # holes come before the contour that holds them
    first_with_hole = next(p for p in on if p.holes)
    seg = text.split(f"; part {first_with_hole.name}")[1].split("; part")[0]
    moves = [ln for ln in seg.splitlines() if ln.startswith("G0")]
    assert len(moves) == 1 + len(first_with_hole.holes)
    with pytest.raises(ValueError, match="outside the bed"):
        write_gcode(tmp_path / "x.gcode", [Part("far", rect(500, 0, 10, 10))], SETTINGS)
    with pytest.raises(ValueError, match="no operation"):
        p = on[0]
        p.engraves.append(polyline([[0, 0], [1, 1]]))
        write_gcode(tmp_path / "y.gcode", [p], SETTINGS.replace(operations={"cut": OPS["cut"]}))


def test_cut_job_writes_everything(tmp_path):
    parts = ribs(12)
    res = cut_job(parts, SETTINGS, tmp_path / "job", name="ribs")
    assert res.ok, res.messages
    assert res.metrics["parts"] == len(parts) and res.metrics["sheets"] == res.nesting.n_sheets >= 1
    for i in range(res.metrics["sheets"]):
        assert res.artifacts[f"svg_{i + 1:02d}"].is_file() and res.artifacts[f"gcode_{i + 1:02d}"].is_file()
    assert res.artifacts["summary_json"].is_file() and res.artifacts["preview_png"].is_file()
    assert res.artifacts["parts_json"].is_file()
    # the kerf: the toolpaths are longer than the drawn contours by about pi * kerf per contour
    drawn = sum(p.cut_length() for p in parts)
    assert res.metrics["cut_length_mm"] / 2 > drawn
    assert res.metrics["cut_length_mm"] / 2 == pytest.approx(drawn, rel=0.02)
    assert res.metrics["parts_mass_g"] == pytest.approx(sum(p.area() for p in parts) * 2e-3 * 0.16)
    assert res.metrics["time_s"] > 0 and 0 < res.metrics["utilisation"] < 1
    import json
    summary = json.loads(res.artifacts["summary_json"].read_text())
    assert summary["metrics"]["sheets"] == res.metrics["sheets"] and summary["metadata"]["settings"]["sheet"]["thickness_mm"] == 2.0
    failed = cut_job([Part("huge", rect(0, 0, 500, 20))], SETTINGS, tmp_path / "bad")
    assert not failed.ok and "does not fit" in failed.messages[0]
