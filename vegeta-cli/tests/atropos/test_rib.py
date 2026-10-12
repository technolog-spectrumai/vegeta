import math

import pytest

from vegeta.atropos import (Lightening, Notch, Path, RibSpec, Sheeting, TubeHole, arc_length, make_rib, naca4,
                            section_area, section_outline)

UP, LO = naca4(0.03, 0.4, 0.12, n=60)
C = 250.0


def spec(**kw):
    return RibSpec(name="r", chord_mm=C, upper=UP, lower=LO, **kw)


def test_plain_rib_is_the_section():
    (rib,) = make_rib(spec())
    assert rib.name == "r" and rib.holes == []
    assert rib.area() == pytest.approx(section_area(UP, LO, C), rel=1e-9)
    assert rib.bbox()[0] == pytest.approx(0.0, abs=1.0) and rib.bbox()[2] == pytest.approx(0.97 * C, abs=1.0)   # a cambered section leans a little past x = 0 and 0.97 c
    assert rib.meta["chord_mm"] == C and rib.meta["role"] == "rib"


def test_sheeting_lets_the_skin_in():
    t = 1.5
    (rib,) = make_rib(spec(sheeting=[Sheeting(0.0, 0.32, t, "both")]))
    removed = section_area(UP, LO, C) - rib.area()
    expected = t * (arc_length(UP, C, 0.0, 0.32) + arc_length(LO, C, 0.0, 0.32))
    assert removed == pytest.approx(expected, rel=0.05)
    # the step at 32 % chord: each surface has the point on the section line and, next to it, the let-in one
    xs = rib.outer.points[:, 0]
    assert ((abs(xs - 0.32 * C) < 1e-6).sum()) == 2 and ((abs(xs - 0.32 * C) < 2.0).sum()) >= 4
    # the upper-only sheet removes about half
    (half,) = make_rib(spec(sheeting=[Sheeting(0.0, 0.32, t, "upper")]))
    assert section_area(UP, LO, C) - half.area() == pytest.approx(t * arc_length(UP, C, 0.0, 0.32), rel=0.08)
    # the trailing-edge sheet reaches the trailing edge with no step there
    (te,) = make_rib(spec(sheeting=[Sheeting(0.85, 1.0, t, "both")]))
    assert section_area(UP, LO, C) - te.area() == pytest.approx(t * (arc_length(UP, C, 0.85, 0.97) + arc_length(LO, C, 0.85, 0.97)), rel=0.08)


def test_tube_holes_on_the_mid_line_and_checked():
    (rib,) = make_rib(spec(tubes=[TubeHole(0.30, 20.0, 0.3), TubeHole(0.55, 18.0, 0.3, name="rear")]))
    assert len(rib.holes) == 2
    assert section_area(UP, LO, C) - rib.area() == pytest.approx(math.pi / 4 * (20.3 ** 2 + 18.3 ** 2), rel=2e-3)
    cx, cz = rib.holes[0].centroid()
    assert cx == pytest.approx(0.30 * C, abs=1e-6)
    outline = Path(section_outline(UP, LO, C))
    zs = [z for x, z in outline.points if abs(x - 0.30 * C) < 3]
    assert min(zs) < cz < max(zs)
    # an explicit centre height is used as given
    (rib2,) = make_rib(spec(tubes=[TubeHole(0.30, 10.0, 0.0, z_mm=5.0)]))
    assert rib2.holes[0].centroid()[1] == pytest.approx(5.0, abs=1e-6)
    with pytest.raises(ValueError, match="does not fit"):
        make_rib(spec(tubes=[TubeHole(0.30, 40.0, 0.0)]))                  # thicker than the rib
    with pytest.raises(ValueError, match="does not fit"):
        make_rib(spec(tubes=[TubeHole(0.30, 20.0, 0.0, min_wall_mm=6.0)]))   # not enough wall left


def test_split_at_the_hinge_with_a_bevel():
    fwd, aft = make_rib(spec(split_at_frac=0.70, split_gap_mm=2.0, bevel_deg=15.0,
                             tubes=[TubeHole(0.30, 20.0, 0.3)]))
    assert fwd.name == "r-fwd" and aft.name == "r-aft"
    assert fwd.bbox()[2] == pytest.approx(0.70 * C - 1.0, abs=1e-6)
    assert aft.bbox()[0] == pytest.approx(0.70 * C + 1.0, abs=1e-6)
    assert len(fwd.holes) == 1 and aft.holes == []
    # the bevel takes material off the aft piece's nose only
    _, aft_plain = make_rib(spec(split_at_frac=0.70, split_gap_mm=2.0))
    assert aft.area() < aft_plain.area()
    assert fwd.area() + aft_plain.area() < section_area(UP, LO, C)
    assert fwd.meta["role"] == "wing" and aft.meta["role"] == "surface"
    # the wedge: at the nose the aft piece is a point; a few mm aft it is open by 2 x tan(15) x dx
    x_nose = aft.bbox()[0]
    near = [z for x, z in aft.outer.points if abs(x - x_nose) < 1e-6]
    assert len(near) == 1


def test_edge_cuts_and_notches_and_lightening():
    (rib,) = make_rib(spec(le_cut_mm=8.0, te_cut_mm=15.0))
    assert rib.bbox()[0] == pytest.approx(8.0) and rib.bbox()[2] == pytest.approx(C - 15.0)
    (notched,) = make_rib(spec(notches=[Notch(0.45, "upper", 6.0, 2.0), Notch(0.45, "lower", 6.0, 2.0)]))
    assert section_area(UP, LO, C) - notched.area() == pytest.approx(2 * 6.0 * 2.0, rel=0.15)
    (light,) = make_rib(spec(tubes=[TubeHole(0.30, 20.0, 0.3)],
                             lightening=[Lightening(0.45, 30.0, 4.0), Lightening(0.70, 30.0, 4.0), Lightening(0.95, 30.0, 4.0)]))
    assert len(light.holes) == 3               # the one at 95 % is too small and dropped
    for h in light.holes[1:]:
        for x, z in h.points:
            assert light.outer.contains(x, z) and light.outer.distance(x, z) >= 4.0 - 1e-6
    with pytest.raises(ValueError):
        make_rib(spec(le_cut_mm=200.0, te_cut_mm=60.0))
