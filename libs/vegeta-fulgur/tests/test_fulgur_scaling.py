"""P4: the scaled family, its calibrated estimate, the masses and the two flight questions."""
import math

import pytest

from vegeta.fulgur import Air
from vegeta.fulgur.scaling import Calibration, Family, at_margin, estimate, flight_masses, thruster_mass, winged

G = 9.81


def test_family_rings_alternate_and_scale():
    f = Family(0.2, pitch=0.014, dz=0.012)
    rings = f.rings()
    roles = [role for _, role, _ in rings]
    assert roles[0] == "ground" and roles[-1] == "ground" and all(a != b for a, b in zip(roles, roles[1:]))
    assert rings[0][0] == pytest.approx(0.1) and all(z == 0.012 for _, role, z in rings if role == "ground")
    assert Family(0.5).n_emitters > Family(0.2).n_emitters
    assert Family(0.5).exit_area == pytest.approx(math.pi * (0.25 ** 2 - 0.02 ** 2))


def test_calibration_fit_recovers_a_known_factor():
    from vegeta.fulgur.corona import Gap
    pts = []
    for pitch, dz in ((0.011, 0.008), (0.014, 0.012), (0.018, 0.016)):
        g = Gap(d=math.hypot(pitch, dz), r=0.1e-3, length=0.8, collection=1.0)
        V0 = 3.0 * g.onset_voltage()
        V = 2.0 * V0
        pts.append({"pitch": pitch, "dz": dz, "V": V, "I_2d": 0.7 * g.current_constant() * V * (V - V0), "onset_2d": V0,
                    "edge_r": 0.1e-3, "length": 0.8})
    cal = Calibration.fit(pts)
    assert cal.k_current == pytest.approx(0.7, rel=1e-6) and cal.k_onset == pytest.approx(3.0, rel=1e-6) and cal.spread < 1e-9


def test_estimate_follows_the_p3_laws():
    cal = Calibration(k_current=1.8, k_onset=4.0)
    f = Family(0.3, dz=0.012, stages=2)
    e = estimate(f, 22e3, cal)
    assert e["thrust"] == pytest.approx(e["I"] * 0.012 / Air().mobility())                 # T = I dz / mu
    assert e["thrust_per_watt"] == pytest.approx(0.012 / (Air().mobility() * 22e3))
    assert estimate(Family(0.3, dz=0.012, stages=1), 22e3, cal)["thrust"] == pytest.approx(e["thrust"] / 2)
    m = at_margin(f, cal, margin=1.2)
    assert m["margin"] == pytest.approx(1.2, rel=1e-3)
    assert at_margin(f, cal, Air(rh=0.95), 1.2)["thrust"] < m["thrust"]                   # wet: the surface limits


def test_masses_and_the_two_questions():
    cal = Calibration(k_current=1.8, k_onset=4.0)
    f = Family(0.5, dz=0.016, stages=3)
    e = at_margin(f, cal)
    assert thruster_mass(f, "flight")["total"] < thruster_mass(f, "replica")["total"]
    m = flight_masses(f, e, minutes=2.0)
    assert m["total"] == pytest.approx(m["thruster"] + m["supply"] + m["battery"])
    assert 0.0 < m["thrust_to_weight"] < 1.0                                               # the hover wall
    w = winged(f, e, m, ld=10.0)
    assert w["margin"] == pytest.approx(m["thrust_to_weight"] * 10.0)
