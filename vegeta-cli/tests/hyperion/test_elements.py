"""Materials against the catalogues; the ABCD elements against the lensmaker's equation, the determinant rule and
the thin-lens limit; trains and their serialisation."""
import math

import numpy as np
import pytest

from vegeta import hyperion as hy


def test_sellmeier_against_the_catalogues():
    assert hy.BK7.n(587.6e-9) == pytest.approx(1.5168, abs=2e-4)                # Schott N-BK7 n_d
    assert hy.FUSED_SILICA.n(632.8e-9) == pytest.approx(1.4570, abs=2e-4)       # Malitson
    assert hy.FUSED_SILICA.n(1064e-9) == pytest.approx(1.4496, abs=2e-4)
    assert hy.YAG.n(1064e-9) == pytest.approx(1.816, abs=3e-3)                   # YAG at the Nd line
    assert hy.SAPPHIRE.n(800e-9) == pytest.approx(1.760, abs=2e-3)
    assert hy.AIR_STP.n(633e-9) == pytest.approx(1.000277, abs=2e-6)
    assert hy.get("bk7") is hy.BK7 and hy.get("Nd:YAG") is hy.YAG and hy.get(hy.YAG) is hy.YAG
    assert hy.index_of(1.7, None) == 1.7 and hy.index_of("vacuum", None) == 1.0
    with pytest.raises(ValueError):
        hy.index_of("BK7", None)                                                 # a Material needs the wavelength
    with pytest.raises(ValueError):
        hy.get("unobtainium")
    assert hy.BK7.group_index(1064e-9) > hy.BK7.n(1064e-9)                      # normal dispersion
    assert hy.Material.from_dict(hy.YAG.to_dict()) == hy.YAG


def test_thick_lens_against_the_lensmaker():
    """An equiconvex n = 1.5 lens, R = ±100 mm, 20 mm thick: f from the lensmaker's equation with the thickness term,
    the principal planes h = −f (n−1) t / (n R₂) inside the lens, BFL = f − h₂ (Hecht)."""
    lens = hy.ThickLens(0.1, -0.1, 0.02, 1.5)
    cp = lens.cardinal_points()
    f_expected = 1.0 / (0.5 * (20.0 - 0.5 * 0.02 / (1.5 * 0.01)))
    assert cp["f"] == pytest.approx(f_expected, rel=1e-12) and cp["f_lensmaker"] == pytest.approx(f_expected, rel=1e-12)
    h = -f_expected * 0.5 * 0.02 / (1.5 * -0.1)
    assert cp["h1"] == pytest.approx(h, rel=1e-9) and cp["h2"] == pytest.approx(h, rel=1e-9)
    assert cp["bfl"] == pytest.approx(f_expected - h, rel=1e-9) and cp["ffl"] == pytest.approx(f_expected - h, rel=1e-9)
    assert np.linalg.det(lens.matrix()) == pytest.approx(1.0, abs=1e-12)        # air to air
    # the thin limit: thickness → 0 gives the thin lens matrix
    thin = hy.ThickLens(0.1, -0.1, 1e-9, 1.5).matrix()
    assert thin == pytest.approx(hy.ThinLens(0.1).matrix(), abs=1e-7)
    # a plano-convex BK7 lens at 633 nm: f = R / (n − 1), the flat side has no power
    pc = hy.ThickLens(0.05, math.inf, 0.005, "BK7")
    assert pc.cardinal_points(633e-9)["f"] == pytest.approx(0.05 / (hy.BK7.n(633e-9) - 1.0), rel=1e-12)
    assert pc.reverse().cardinal_points(633e-9)["f"] == pytest.approx(pc.cardinal_points(633e-9)["f"], rel=1e-12)


def test_determinants_and_media():
    """Every element in air has determinant 1; an interface's is n₁/n₂; a slab's parts join."""
    for e in (hy.FreeSpace(0.3), hy.ThinLens(0.2), hy.ThermalLens(0.5, 10.0), hy.FlatMirror(), hy.CurvedMirror(0.4, 20.0),
              hy.Slab(0.01, "Nd:YAG"), hy.BrewsterPlate(0.003, "BK7"), hy.ThickLens(0.1, -0.2, 0.01, "BK7")):
        for plane in ("t", "s"):
            assert np.linalg.det(e.matrix(1064e-9, plane)) == pytest.approx(1.0, abs=1e-12), e
    i = hy.Interface(1.0, 1.5, 0.1)
    assert np.linalg.det(i.matrix()) == pytest.approx(1.0 / 1.5)
    assert np.linalg.det(i.reverse().matrix()) == pytest.approx(1.5)
    # a flat interface crossed and crossed back is nothing; a curved one crossed twice has twice the power (a lens on the way back too)
    flat = hy.Interface(1.0, 1.5)
    assert (flat.reverse().matrix() @ flat.matrix()) == pytest.approx(np.eye(2), abs=1e-12)
    twice = i.reverse().matrix() @ i.matrix()
    assert twice[1, 0] == pytest.approx(2 * (1.0 - 1.5) / (0.1 * 1.0)) and twice[1, 1] == pytest.approx(1.0)
    slab = hy.Slab(0.01, 1.8)
    m = np.eye(2)
    for p in slab.parts():
        m = p.matrix() @ m
    assert m == pytest.approx(slab.matrix(), abs=1e-12)
    assert slab.matrix()[0, 1] == pytest.approx(0.01 / 1.8)
    assert slab.optical_length() == pytest.approx(0.018)
    with pytest.raises(ValueError):
        hy.Train([hy.FreeSpace(0.1), hy.FreeSpace(0.1, 1.5)]).check()           # media do not join
    hy.Train([hy.FreeSpace(0.1), hy.Interface(1.0, 1.5), hy.FreeSpace(0.1, 1.5), hy.Interface(1.5, 1.0)]).check()


def test_tilted_mirror_and_brewster_plate():
    m = hy.CurvedMirror(0.5, 30.0)
    assert m.focal_length("t") == pytest.approx(0.25 * math.cos(math.radians(30)))
    assert m.focal_length("s") == pytest.approx(0.25 / math.cos(math.radians(30)))
    assert hy.CurvedMirror(0.5).matrix() == pytest.approx(hy.ThinLens(0.25).matrix())
    bp = hy.BrewsterPlate(0.005, 1.5)
    s = math.sqrt(1.5 ** 2 + 1)
    assert bp.matrix(plane="t")[0, 1] == pytest.approx(0.005 * s / 1.5 ** 4)
    assert bp.matrix(plane="s")[0, 1] == pytest.approx(0.005 * s / 1.5 ** 2)
    assert bp.length == pytest.approx(0.005 * s / 1.5)
    assert bp.matrix(plane="t")[0, 1] < bp.matrix(plane="s")[0, 1] < bp.length    # Kogelnik 1972


def test_train_positions_product_and_json():
    t = hy.Train([hy.CurvedMirror(0.5, name="M1"), hy.FreeSpace(0.05, name="a"), hy.Slab(0.01, "Nd:YAG", name="rod"), hy.FreeSpace(0.14, name="b"),
                  hy.FlatMirror(0.95, name="OC")], name="svarog")
    assert t.positions() == pytest.approx([0.0, 0.0, 0.05, 0.06, 0.2, 0.2])
    assert t.length == pytest.approx(0.2)
    assert t.optical_length(1064e-9) == pytest.approx(0.19 + 0.01 * hy.YAG.n(1064e-9))
    m = t.matrix(1064e-9)
    assert m == pytest.approx(t.matrix(1064e-9, start=2) @ t.matrix(1064e-9, stop=2), abs=1e-12)
    assert hy.Train.from_dict(t.to_dict()).matrix(1064e-9) == pytest.approx(m, abs=1e-12)
    assert [r["element"] for r in t.table(1064e-9)] == ["M1", "a", "rod", "b", "OC"]
    rev = t.reverse()
    assert [e.name for e in rev] == ["OC", "b", "rod", "a", "M1"]
    assert hy.element_from_dict(hy.CurvedMirror(0.3, 10.0, 0.9, "x").to_dict()) == hy.CurvedMirror(0.3, 10.0, 0.9, "x")
