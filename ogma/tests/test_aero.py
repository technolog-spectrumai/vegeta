"""og.aero reproduces the Vegeta code each piece came from (tests/golden/aero.json)."""
import json
import math
from pathlib import Path

import numpy as np
import pytest

from ogma import aero

G = json.loads((Path(__file__).parent / "golden" / "aero.json").read_text())
C = G["coeff"]
STATES = [(a, b, p, q, r, tuple(d), V) for a, b, p, q, r, d, V in G["states"]]


@pytest.mark.parametrize("key,crow,post", [("nisus", 0.0, True), ("nisus_linear", 0.0, False), ("plus_crow0", 0.0, True),
                                           ("plus_crow", 0.6, True), ("plus_crow_linear", 0.6, False)])
def test_stability_coefficients_are_nisus_and_nisus_plus(key, crow, post):
    for s, want in zip(STATES, G[key]):
        got = aero.stability_coefficients(C, *s, post_stall=post, crow=crow)
        assert list(map(float, got)) == want


def test_polar_is_kestrels():
    g = G["polar"]
    pol = aero.Polar(cl_alpha=4.6, cd0=0.035, k=0.07, cl_max=1.05, cd90=aero.viterna_cd90(6.5), alpha0=math.radians(-2.0))
    assert aero.viterna_cd90(6.5) == g["cd90"] and pol.alpha_stall == g["alpha_stall"]
    cl, cd, cm = pol.coefficients(np.array(g["alpha"]))
    np.testing.assert_array_equal(cl, g["cl"]); np.testing.assert_array_equal(cd, g["cd"]); np.testing.assert_array_equal(cm, g["cm"])


def test_airframe_is_merlins():
    g = G["airframe"]
    from ogma.units import G, G_VEGETA
    assert aero.Airframe("t", 2.4, 0.5, 8.0, 0.03).g == G
    af = aero.Airframe("t", 2.4, 0.5, 8.0, 0.03, g=G_VEGETA)          # Merlin's model as recorded (9.81)
    np.testing.assert_array_equal(af.drag(np.array(g["V"])), g["drag"])
    np.testing.assert_array_equal(af.drag(np.array(g["V"]), rho=1.1, n=2.0), g["drag_n2"])
    assert af.stall_speed() == g["stall"] and af.k == g["k"]


def test_vlm_is_peregrines():
    g = G["vlm"]
    quads = [np.array(q) for q in G["quads"]]
    r = aero.vlm(quads, alpha_deg=4.0, incidence_deg=[2.0, 2.0, -1.0, -1.0], S_ref=0.258, c_ref=0.215, x_ref=0.06,
                 n_span=[8, 8, 4, 4], n_chord=3, hidden=[0.0, 0.0, 0.0, 0.0])
    # a linear solve: the last bit depends on the BLAS, so 1e-12 rather than bit-equal
    assert (r["CL"], r["CDi"], r["Cm"]) == pytest.approx((g["CL"], g["CDi"], g["Cm"]), rel=1e-12)
    np.testing.assert_allclose(r["gamma"], g["gamma"], rtol=1e-12)


def test_vlm_flat_rectangle_near_helmbold():
    """A flat AR-6 rectangle: CLα within 8 % of Helmbold's 2πA / (2 + sqrt(A² + 4)) and a span efficiency a little
    below elliptic (Vegeta's own lattice test)."""
    b, c = 1.2, 0.2
    q = np.array([[0.0, -b / 2, 0.0], [0.0, b / 2, 0.0], [c, b / 2, 0.0], [c, -b / 2, 0.0]])
    r = aero.vlm([q], alpha_deg=5.0, S_ref=b * c, c_ref=c, n_span=24, n_chord=4)
    A = b / c
    assert r["CL"] / math.radians(5.0) == pytest.approx(2 * math.pi * A / (2 + math.sqrt(A * A + 4)), rel=0.08)
    assert 0.93 < r["CL"] ** 2 / (math.pi * A * r["CDi"]) < 1.03


def test_skin_friction():
    g = G["cf"]
    assert [aero.cf_mixed(x) for x in g["re"]] == g["mixed"]
    assert [aero.cf_turbulent(x, re_min=1e4) for x in g["re"]] == g["turbulent"]
    assert [aero.cf_turbulent(x, mach=0.6, re_min=1e4) for x in g["re"]] == g["mach06"]


def test_air_angles_and_wrench():
    v = np.array([12.0, -1.0, -1.5])                         # body x fwd, y left, z up: a little descent and sideslip
    V, alpha, beta, uvw = aero.air_angles(v)
    assert V == pytest.approx(np.linalg.norm(v)) and alpha == pytest.approx(math.atan2(1.5, 12.0))
    assert beta == pytest.approx(math.asin(1.0 / V))
    F, M = aero.aero_wrench((0.5, 0.04, 0.0, 0.0, 0.0, 0.0), V, alpha, uvw, 1.225, 0.4, 1.5, 0.27)
    qS = 0.5 * 1.225 * V * V * 0.4
    drag = -F @ v / V
    assert drag == pytest.approx(0.04 * qS) and np.linalg.norm(F) == pytest.approx(qS * math.hypot(0.5, 0.04))
    assert F[2] > 0 and np.allclose(M, 0.0)                  # lift up (z up) and no moment from pure lift and drag
    assert aero.air_angles(np.array([0.5, 0.0, 0.0]))[1:3] == (0.0, 0.0)
