"""og.stratosphere against the U.S. Standard Atmosphere 1976 tables (NOAA-S/T 76-1562, geometric altitude)."""
import numpy as np
import pytest

from ogma import atmosphere as oa
from ogma import stratosphere as st

# z [m], T [K], p [Pa], rho [kg/m³] — Table I of the 1976 standard (5 significant figures as printed)
TABLE = [(0.0, 288.150, 101325.0, 1.2250),
         (11000.0, 216.774, 22700.0, 0.36480),
         (20000.0, 216.650, 5529.3, 0.088910),
         (25000.0, 221.552, 2549.2, 0.040084),
         (30000.0, 226.509, 1197.0, 0.018410),
         (50000.0, 270.650, 79.779, 0.0010269)]


@pytest.mark.parametrize("z,T,p,rho", TABLE)
def test_table(z, T, p, rho):
    s = st.state(z)
    assert s.temperature == pytest.approx(T, abs=2e-3)
    assert s.pressure == pytest.approx(p, rel=2e-4)
    assert s.density == pytest.approx(rho, rel=2e-4)


def test_layer_bases_geopotential():
    # Table 4: the base pressures follow from the hydrostatic equation
    assert st.pressure(st.geometric(11000.0)) == pytest.approx(22632.06, rel=1e-6)
    assert st.pressure(st.geometric(20000.0)) == pytest.approx(5474.889, rel=1e-6)
    assert st.pressure(st.geometric(32000.0)) == pytest.approx(868.0187, rel=1e-6)


def test_matches_og_atmosphere_in_the_troposphere():
    z = np.linspace(0.0, 10000.0, 21)
    tropo = [oa.isa(float(h)).density for h in z]
    # og.atmosphere uses geometric altitude in the geopotential formula: they differ by < 0.2 % up to 10 km
    assert np.allclose(st.density(z), tropo, rtol=2e-3)


def test_gravity_and_inverse():
    assert st.gravity(0.0) == pytest.approx(9.80665)
    assert st.gravity(20000.0) == pytest.approx(9.80665 * (6356766 / 6376766) ** 2)
    assert st.altitude_of_density(st.density(20000.0)) == pytest.approx(20000.0, abs=0.01)
    assert st.altitude_of_pressure(st.pressure(18500.0)) == pytest.approx(18500.0, abs=0.01)


def test_delta_t_keeps_pressure():
    assert st.pressure(20000.0) == st.state(20000.0, 15.0).pressure
    assert st.state(20000.0, 15.0).temperature == pytest.approx(216.65 + 15.0)


def test_out_of_range_refused():
    with pytest.raises(ValueError):
        st.state(90000.0)
