"""Strato's hull numbers and CAD (Stage V8.5): the analytic profile and the solid agree, impossible hulls are refused,
every part builds, and the family scales as L³ / L²."""
import math

import pytest

import strato
import strato_geometry as sg


def test_ellipsoid_limit():
    # n_nose = 2 with x_max = 0.5 is half an ellipsoid; the tail with n_tail = 2 is a paraboloid of revolution:
    # V = 2/3 π R² (L/2) + 8/15 π R² (L/2) (∫(1-u²)² du from 0 to 1 = 8/15)
    h = sg.Hull(10.0, 2.0, x_max=0.5)
    expected = math.pi * 1.0 * 5.0 * (2 / 3 + 8 / 15)
    assert h.volume == pytest.approx(expected, rel=1e-5)


def test_demonstrator_numbers():
    s = sg.DEMONSTRATOR.summary()
    assert s["fineness"] == pytest.approx(15 / 3.5)
    assert 0.5 < s["prismatic_coefficient"] < 0.667          # between a cone pair and an ellipsoid
    assert 0.35 * 15 < s["centre_of_buoyancy_m"] < 0.5 * 15


@pytest.mark.parametrize("kw", [dict(length=-1.0), dict(diameter=10.0), dict(x_max=0.9), dict(n_tail=1.0)])
def test_impossible_hulls_refused(kw):
    with pytest.raises(ValueError):
        sg.Hull(**kw)


def test_family_scales():
    fam = sg.family()
    v15, a15 = fam[15.0].hull.volume, fam[15.0].hull.surface_area
    for L, m in fam.items():
        k = L / 15.0
        assert m.hull.volume == pytest.approx(v15 * k ** 3, rel=1e-9)
        assert m.hull.surface_area == pytest.approx(a15 * k ** 2, rel=1e-9)


def test_cad_hull_matches_profile():
    g = strato.StratoAirship().generate(part="hull")
    h = sg.DEMONSTRATOR.hull
    assert g.volume / 1e9 == pytest.approx(h.volume, rel=2e-3)      # the tail closed at 4 % of R loses ~0.01 %
    assert g.surface_area / 1e6 == pytest.approx(h.surface_area, rel=2e-3)


def test_every_part_builds_and_is_valid():
    d = strato.StratoAirship()
    for part in strato.PARTS:
        m = d.generate(part=part).measure()
        assert m["valid"], part
    assert d.generate(part="airship").measure()["n_solids"] == 1      # one connected ship


def test_cad_refuses_bad_array():
    from vegeta.dedalus.design import BuildError
    with pytest.raises((ValueError, BuildError)):
        strato.StratoAirship().generate(solar_x0=0.7, solar_x1=0.2)
