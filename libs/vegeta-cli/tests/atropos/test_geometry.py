import math

import numpy as np
import pytest

from vegeta.atropos import (Part, Path, arc_length, circle, clip_x, naca4, rect, section_area, section_outline,
                            thickness_at)


def test_rect_measures_and_offset():
    r = rect(10, 20, 100, 50)
    assert r.area() == pytest.approx(5000) and r.length() == pytest.approx(300)
    assert r.bbox() == (10, 20, 110, 70) and r.centroid() == pytest.approx((60, 45))
    k = 0.2
    grown = r.offset(k)
    # the bisector offset of a right angle is k*sqrt(2) along the diagonal: the exact grown rectangle
    assert grown.area() == pytest.approx(5000 + 300 * k + 4 * k * k, rel=1e-9)
    assert r.offset(-k).area() == pytest.approx(5000 - 300 * k + 4 * k * k, rel=1e-9)
    assert r.reversed().offset(k).area() == pytest.approx(-(5000 + 300 * k + 4 * k * k), rel=1e-9)   # a hole grows outward too


def test_circle_offset_is_a_radius_change():
    c = circle(5, -3, 10, n=256)
    assert c.area() == pytest.approx(math.pi * 100, rel=1e-3)
    assert c.offset(1).area() == pytest.approx(math.pi * 121, rel=1e-3)
    assert c.offset(-1).area() == pytest.approx(math.pi * 81, rel=1e-3)


def test_airfoil_outline_area_and_offset():
    up, lo = naca4(0.0, 0.0, 0.12, n=200, te_cut=1.0)
    # closed-form area of a NACA 00xx: 0.6851 t c^2 (integral of the thickness polynomial)
    assert section_area(up, lo, 1.0) == pytest.approx(0.6851 * 0.12, rel=2e-3)
    assert section_area(up, lo, 250.0) == pytest.approx(0.6851 * 0.12 * 250 ** 2, rel=2e-3)
    p = Path(section_outline(*naca4(0.03, 0.4, 0.12, n=60), 250.0))
    assert p.area() > 0                                        # counter-clockwise
    k = 0.1
    grown = p.offset(k)
    assert grown.area() == pytest.approx(p.area() + p.length() * k, rel=5e-3)
    assert p.offset(-k).area() == pytest.approx(p.area() - p.length() * k, rel=5e-3)


def test_arc_length_and_thickness():
    up, lo = naca4(0.0, 0.0, 0.12, n=200, te_cut=1.0)
    assert arc_length(up, 100.0, 0.0, 1.0) > 100.0                      # a curved surface is longer than its chord
    assert arc_length(up, 100.0, 0.0, 1.0) == pytest.approx(arc_length(lo, 100.0, 0.0, 1.0), rel=1e-9)
    assert arc_length(up, 100.0, 0.2, 0.3) == pytest.approx(10.0, rel=0.02)   # nearly flat there
    zu, zl = thickness_at(up, lo, 100.0, 0.3)
    assert zu == pytest.approx(6.0, rel=0.01) and zl == pytest.approx(-6.0, rel=0.01)   # 12 % thick at 30 %


def test_contains_distance_clip_and_transform():
    r = rect(0, 0, 10, 4)
    assert r.contains(5, 2) and not r.contains(11, 2) and r.contains(10, 2)
    assert r.distance(5, 2) == pytest.approx(2) and r.distance(12, 2) == pytest.approx(2)
    left = Path(clip_x(r.points, None, 4))
    assert left.area() == pytest.approx(16) and left.bbox() == (0, 0, 4, 4)
    mid = Path(clip_x(r.points, 3, 7))
    assert mid.area() == pytest.approx(16)
    assert len(clip_x(r.points, 20, None)) == 0
    t = r.transformed(90, 0, 0)
    assert t.bbox() == pytest.approx((-4, 0, 0, 10))
    assert t.area() == pytest.approx(40)


def test_part_measures_kerf_and_json_round_trip():
    outer = rect(0, 0, 100, 30)
    hole = circle(50, 15, 8, n=128)
    part = Part("p", outer, [hole], meta={"station": 1})
    assert part.holes[0].area() < 0                                    # holes run clockwise
    assert part.area() == pytest.approx(3000 - math.pi * 64, rel=1e-3)
    assert part.cut_length() == pytest.approx(260 + 2 * math.pi * 8, rel=1e-3)
    comp = part.kerf_compensated(0.2)
    assert abs(comp.outer.area()) > abs(part.outer.area())
    assert abs(comp.holes[0].area()) == pytest.approx(math.pi * 7.9 ** 2, rel=1e-3)   # the hole's toolpath shrinks
    back = Part.from_dict(part.to_dict())
    assert back.name == "p" and back.area() == pytest.approx(part.area()) and back.meta == {"station": 1}
    moved = part.transformed(0, 5, 5).at_origin()
    assert moved.bbox() == pytest.approx((0, 0, 100, 30))
    with pytest.raises(ValueError):
        Path(np.array([[0, 0], [1, 1]]), closed=True)
