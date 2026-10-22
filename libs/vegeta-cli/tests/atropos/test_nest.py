import math

import pytest

from vegeta.atropos import (CutSettings, Machine, Operation, Part, RibSpec, Sheet, TubeHole, circle, make_rib, naca4, nest,
                            rect)

MACHINE = Machine("test laser", 432.0, 406.0, travel_feed_mm_min=6000.0)
SHEET = Sheet("balsa 400 x 100 x 2", 400.0, 100.0, 2.0, 0.16, grain="x", margin_mm=4.0)
OPS = {"cut": Operation("cut", 100.0, 600.0, 2, 0.15, order=2), "score": Operation("score", 25.0, 1500.0, 1, 0.0, order=1),
       "engrave": Operation("engrave", 12.0, 3000.0, 1, 0.0, order=0)}
SETTINGS = CutSettings(SHEET, MACHINE, OPS, spacing_mm=3.0)

UP, LO = naca4(0.03, 0.4, 0.12, n=40)


def ribs(n=40):
    out = []
    for i in range(n):
        c = 300.0 - 100.0 * i / (n - 1)
        out += make_rib(RibSpec(f"R{i:02d}", c, UP, LO, tubes=[TubeHole(0.30, 12.0, 0.3)],
                                split_at_frac=0.72 if i % 3 == 0 else None, split_gap_mm=1.5))
    return out


def _overlap(a: Part, b: Part, gap: float) -> bool:
    ax0, ay0, ax1, ay1 = a.bbox()
    bx0, by0, bx1, by1 = b.bbox()
    return not (ax1 + gap <= bx0 + 1e-6 or bx1 + gap <= ax0 + 1e-6 or ay1 + gap <= by0 + 1e-6 or by1 + gap <= ay0 + 1e-6)


def test_nesting_is_valid_and_deterministic():
    parts = ribs()
    n = nest(parts, SETTINGS)
    assert len(n.placements) == len(parts)
    W, H = SHEET.width_mm, SHEET.height_mm
    for i in range(n.n_sheets):
        on = n.parts_on(i)
        for p in on:
            x0, y0, x1, y1 = p.bbox()
            assert x0 >= SHEET.margin_mm - 1e-6 and y0 >= SHEET.margin_mm - 1e-6
            assert x1 <= W - SHEET.margin_mm + 1e-6 and y1 <= H - SHEET.margin_mm + 1e-6
        for a in range(len(on)):
            for b in range(a + 1, len(on)):
                assert not _overlap(on[a], on[b], SETTINGS.spacing_mm), (on[a].name, on[b].name)
    usable = (W - 2 * SHEET.margin_mm) * (H - 2 * SHEET.margin_mm)
    assert n.n_sheets >= math.ceil(sum(p.area() for p in parts) / usable)
    assert all(0 < u < 1 for u in n.utilisation)
    again = nest(parts, SETTINGS)
    assert again.table() == n.table()
    assert sum(p.area() for i in range(n.n_sheets) for p in n.parts_on(i)) == pytest.approx(sum(p.area() for p in parts))


def test_orientation_grain_and_too_large():
    tall = Part("tall", rect(0, 0, 20, 80))
    n = nest([tall], SETTINGS)
    assert n.placements[0].angle_deg == 90.0 and n.placements[0].part.size() == pytest.approx((80, 20))
    along_y = Part("y-grain", rect(0, 0, 20, 80), meta={"grain": "y"})       # the part's y must follow the grain (sheet x)
    assert nest([along_y], SETTINGS).placements[0].angle_deg == 90.0
    along_x = Part("x-grain", rect(0, 0, 20, 80), meta={"grain": "x"})
    assert nest([along_x], SETTINGS).placements[0].angle_deg == 0.0
    no_rot = nest([tall], SETTINGS.replace(rotate=False))
    assert no_rot.placements[0].angle_deg == 0.0
    with pytest.raises(ValueError, match="does not fit"):
        nest([Part("huge", rect(0, 0, 500, 20))], SETTINGS)
    with pytest.raises(ValueError, match="does not fit"):
        nest([Part("wide", rect(0, 0, 95, 95))], SETTINGS.replace(rotate=False))
    # a hole does not change the footprint
    with_hole = Part("h", rect(0, 0, 60, 30), [circle(30, 15, 5)])
    assert nest([with_hole], SETTINGS).placements[0].part.size() == pytest.approx((60, 30))


def test_settings_validation():
    with pytest.raises(ValueError, match="'cut'"):
        CutSettings(SHEET, MACHINE, {"score": OPS["score"]})
    with pytest.raises(ValueError, match="does not fit the bed"):
        CutSettings(Sheet("big", 500, 100, 2, 0.16), MACHINE, OPS)
    with pytest.raises(ValueError):
        Operation("cut", 120.0, 600.0, 2, 0.15)
    with pytest.raises(ValueError):
        Operation("cut", 50.0, 600.0, 0, 0.15)
    with pytest.raises(ValueError):
        Sheet("thin", 400, 100, 0.0, 0.16)
    assert SHEET.mass_g == pytest.approx(400 * 100 * 2 * 1e-3 * 0.16)
    assert SETTINGS.replace(spacing_mm=1.0).spacing_mm == 1.0 and SETTINGS.spacing_mm == 3.0
