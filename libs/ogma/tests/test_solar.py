"""og.solar: the NOAA algorithm's known values and irradiance sanity."""
from datetime import datetime, timezone

import numpy as np
import pytest

from ogma import solar as so

UTC = timezone.utc


def test_declination_at_solstices_and_equinox():
    assert so.declination(datetime(2026, 6, 21, 12, tzinfo=UTC)) == pytest.approx(23.44, abs=0.02)
    assert so.declination(datetime(2026, 12, 21, 12, tzinfo=UTC)) == pytest.approx(-23.44, abs=0.02)
    assert abs(so.declination(datetime(2026, 3, 20, 14, tzinfo=UTC))) < 0.05      # March equinox 2026: 20 Mar 14:46 UTC


def test_equation_of_time_extremes():
    # the classic extremes: about −14.2 min in mid February and +16.4 min in early November
    assert so.equation_of_time(datetime(2026, 2, 11, 12, tzinfo=UTC)) == pytest.approx(-14.2, abs=0.15)
    assert so.equation_of_time(datetime(2026, 11, 3, 12, tzinfo=UTC)) == pytest.approx(16.45, abs=0.15)


def test_noon_elevation():
    lat = 50.0
    t = datetime(2026, 6, 21, 12, 0, tzinfo=UTC) - __import__("datetime").timedelta(minutes=so.equation_of_time(datetime(2026, 6, 21, 12, tzinfo=UTC)))
    el, az = so.position(t, lat, 0.0)
    assert el == pytest.approx(90 - lat + 23.44, abs=0.05)
    assert az == pytest.approx(180.0, abs=0.5)


def test_distance_perihelion_aphelion():
    assert so.distance_au(datetime(2026, 1, 3, tzinfo=UTC)) == pytest.approx(0.9833, abs=2e-4)
    assert so.distance_au(datetime(2026, 7, 6, tzinfo=UTC)) == pytest.approx(1.0167, abs=2e-4)


def test_day_length():
    eq = datetime(2026, 3, 20, tzinfo=UTC)
    assert so.day_length(eq, 0.0) == pytest.approx(12.0, abs=0.1)
    # height lengthens the day: the horizon dips ~4.5° at 20 km
    assert so.horizon_dip(20000.0) == pytest.approx(4.54, abs=0.02)
    assert so.day_length(datetime(2026, 12, 21, tzinfo=UTC), 50.0, 0.0, 20000.0) > so.day_length(datetime(2026, 12, 21, tzinfo=UTC), 50.0)


def test_beam_above_the_clouds():
    t = datetime(2026, 6, 21, 12, tzinfo=UTC)
    top = so.beam_normal(t, 60.0, 0.0)
    assert top == pytest.approx(so.extraterrestrial(t))
    at20 = so.beam_normal(t, 60.0, 5529.3, 20000.0)
    sea = so.beam_normal(t, 60.0, 101325.0)
    assert sea < at20 < top and at20 / top > 0.9
    assert so.beam_normal(t, -10.0, 5529.3, 20000.0) == 0.0


def test_irradiance_on_facets():
    t = datetime(2026, 6, 21, 12, 0, tzinfo=UTC)
    el, az = so.position(t, 50.0, 0.0)
    s = so.sun_vector(el, az)
    E = so.irradiance_on([s, -s, [0, 0, 1]], t, 50.0, 0.0, 5529.3, 20000.0, albedo=0.0, diffuse_fraction=0.0)
    dni = so.beam_normal(t, el, 5529.3, 20000.0)
    assert E[0] == pytest.approx(dni) and E[1] == 0.0
    assert E[2] == pytest.approx(dni * np.sin(np.radians(el)))
