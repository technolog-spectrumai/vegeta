"""The thin-plate (line-source) welding forms against their closed forms: K₀, Rosenthal's 2-D field, Rykalin's t8/5,
Adams' cooling rate, the HAZ."""
import math

import numpy as np
import pytest

from vegeta.hyperion import thermal as th
from vegeta.hyperion import welding as wd


def test_bessel_k0_against_its_limits():
    x = np.array([0.01, 0.5, 1.0, 2.0, 5.0, 10.0])
    k0 = wd._bessel_k0(x)
    assert k0[0] == pytest.approx(-math.log(0.01 / 2) - 0.5772156649, rel=1e-3)                    # the logarithmic limit
    assert k0[-1] == pytest.approx(math.sqrt(math.pi / 20) * math.exp(-10) * (1 - 1 / 80), rel=2e-3)   # the asymptote
    assert k0[2] == pytest.approx(0.4210244382, rel=1e-6) and k0[3] == pytest.approx(0.1138938727, rel=1e-6)
    assert np.all(np.diff(k0) < 0)


def test_thin_plate_field_shape_and_pool():
    q, v, d = 400.0, 0.02, 2e-3
    T = wd.thin_plate_field("mild steel", q, v, d, [-0.01, -0.002, 0.0, 0.002], 0.0)
    assert T[2] == th.MILD_STEEL.T_fail and T[1] > T[3] > th.T_ROOM                                # capped at the source; hotter behind than ahead
    pool = wd.rosenthal_thin_plate("mild steel", q, v, d)
    assert pool["length_behind"] > pool["length_ahead"] > 0 and 0 < pool["half_width"] < pool["length_behind"]
    assert pool["solidification_time"] == pytest.approx(pool["length_behind"] / v)
    assert pool["peak_T"][0] == th.MILD_STEEL.T_fail and np.all(np.diff(pool["peak_T"]) <= 1e-9)
    # on the centreline behind the source the field is q/(2πkd) e^u K0(u): check one point against the formula
    xi = -0.004
    u = v * 0.004 / (2 * th.MILD_STEEL.diffusivity)
    expect = th.T_ROOM + q / (2 * math.pi * 40 * d) * math.exp(u) * float(wd._bessel_k0(u))
    assert float(pool["field"](xi, 0.0)) == pytest.approx(expect, rel=1e-9)
    wider = wd.rosenthal_thin_plate("mild steel", q, v, 1e-3)
    assert wider["half_width"] > pool["half_width"]                                                   # a thinner sheet spreads the same heat wider


def test_t85_and_cooling_rate_closed_forms():
    m = th.MILD_STEEL
    Q, d, T0 = 1.0e5, 2e-3, th.T_ROOM
    r = wd.cooling_time_t85(m, Q, d, T0)
    thin = (Q / d) ** 2 * (1 / (773.15 - T0) ** 2 - 1 / (1073.15 - T0) ** 2) / (4 * math.pi * m.k * m.rho * m.c)
    assert r["t85_thin_s"] == pytest.approx(thin) and r["t85_thick_s"] == pytest.approx(Q * (1 / (773.15 - T0) - 1 / (1073.15 - T0)) / (2 * math.pi * m.k))
    assert r["regime"] == "thin" and r["t85_s"] == r["t85_thin_s"]
    assert wd.cooling_time_t85(m, Q, 0.05, T0)["regime"] == "thick"
    c = wd.cooling_rate_at(m, Q, d, 900.0, T0)
    assert c["rate_thin_K_per_s"] == pytest.approx(2 * math.pi * m.k * m.rho * m.c * (d / Q) ** 2 * (900 - T0) ** 3)
    assert wd.cooling_rate_at(m, 2 * Q, d, 900.0, T0)["rate_thin_K_per_s"] < c["rate_thin_K_per_s"]   # more heat input cools slower


def test_haz_grows_with_heat_input():
    a = wd.haz_width("mild steel", 300.0, 0.02, 2e-3, 996.0)
    b = wd.haz_width("mild steel", 600.0, 0.02, 2e-3, 996.0)
    assert 0 < a["haz_width_each_side"] < b["haz_width_each_side"] and a["haz_half_width"] > a["melt_half_width"]
