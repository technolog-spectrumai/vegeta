"""The geometry and thin-airfoil theory against the textbook (Abbott & von Doenhoff): no backend needed."""
import math

import numpy as np
import pytest

from vegeta import zephyr
from vegeta.zephyr import naca, thin_airfoil


def test_naca4_thickness_and_symmetry():
    s = naca("0012")
    xy = s.coordinates(200)
    assert xy[0].tolist() == pytest.approx([1.0, 0.0], abs=1e-9) and xy[-1].tolist() == pytest.approx([1.0, 0.0], abs=1e-9)   # closed TE
    i = np.argmin(xy[:, 0])
    assert xy[i].tolist() == pytest.approx([0.0, 0.0], abs=1e-9)
    up = xy[:i + 1]
    assert up[:, 1].max() == pytest.approx(0.06, abs=0.0005)                                   # 12 % thick
    assert abs(up[np.argmax(up[:, 1]), 0] - 0.30) < 0.02                                        # thickest at 30 % chord
    open_te = zephyr.thickness_distribution(1.0, 0.12, closed=False)
    assert open_te == pytest.approx(5 * 0.12 * 0.0021, abs=2e-5)                         # the classic open trailing edge: 0.21 % of the chord
    assert zephyr.thickness_distribution(1.0, 0.12) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("designation,alpha0,cm", [("2412", -2.077, -0.0528), ("4412", -4.15, -0.1056), ("0012", 0.0, 0.0),
                                                     ("23012", -1.09, -0.0133), ("6412", -6.2, -0.158)])
def test_thin_airfoil_against_the_textbook(designation, alpha0, cm):
    """α₀ and Cm_c/4 of the classic camber lines (thin-airfoil theory as tabulated by Abbott & von Doenhoff)."""
    ta = thin_airfoil(naca(designation))
    assert ta["alpha0_deg"] == pytest.approx(alpha0, abs=0.06)
    assert ta["cm_c4"] == pytest.approx(cm, abs=0.002)
    assert ta["cl_alpha_per_rad"] == pytest.approx(2 * math.pi)


def test_reflexed_five_digit_lines_have_no_moment():
    """The standard reflexed lines (230-, 240-, 250-series, third digit 1) were designed for Cm_c/4 = 0."""
    for P in (2, 3, 4, 5):
        ta = thin_airfoil(naca(f"2{P}112"))
        assert abs(ta["cm_c4"]) < 0.0025, (P, ta)
        assert -1.0 < ta["alpha0_deg"] < -0.5
    more = thin_airfoil(naca("23112", reflex_k21=0.02))
    assert more["cm_c4"] > 0.03 and more["alpha0_deg"] > thin_airfoil(naca("23112"))["alpha0_deg"]   # more reflex: nose-up moment


def test_flap_geometry_and_effectiveness():
    s = naca("2412")
    f = s.flapped(0.25, 15.0)
    xy = np.array(f.points)
    assert xy[0, 1] < -0.05 and xy[0, 0] < 1.0                                                  # the trailing edge went down and forward
    assert f.params["flap_deg"] == 15.0
    assert zephyr.flap_effectiveness(0.25) == pytest.approx(0.61, abs=0.02)                     # the textbook curve
    assert zephyr.flap_effectiveness(1.0) == pytest.approx(1.0) and zephyr.flap_effectiveness(1e-9) == pytest.approx(0.0, abs=1e-4)
    # thin-airfoil theory on the flapped outline's mean line agrees with the closed form dα₀/dδ = −τ at a small
    # deflection (a thick outline turned 15° is no longer the linear theory's kinked line: τ drops to ~0.5)
    # on the flat plate (0012) the mean line of the flapped thick outline is the kinked line of the theory; a cambered
    # outline's numeric mean line rounds the kink a little more
    flat = naca("0012")
    assert -thin_airfoil(flat.flapped(0.25, 5.0))["alpha0_deg"] / 5.0 == pytest.approx(zephyr.flap_effectiveness(0.25), rel=0.08)
    assert -thin_airfoil(flat.flapped(0.25, 15.0))["alpha0_deg"] / 15.0 < zephyr.flap_effectiveness(0.25)
    a00 = thin_airfoil(s)["alpha0_deg"]
    assert 0.45 < (a00 - thin_airfoil(s.flapped(0.25, 5.0))["alpha0_deg"]) / 5.0 < 0.65


def test_dat_round_trip(tmp_path):
    s = naca("23112")
    p = s.write_dat(tmp_path / "s.dat", n=80)
    r = zephyr.from_dat(p)
    assert r.name == "NACA 23112" and len(r.points) == 159
    assert np.allclose(np.array(r.points), s.coordinates(80), atol=1e-6)
    ta = thin_airfoil(r)
    assert ta["alpha0_deg"] == pytest.approx(thin_airfoil(s)["alpha0_deg"], abs=0.1)
    # Lednicer: counts line, upper from the LE, lower from the LE
    xy = s.coordinates(50)
    i = int(np.argmin(xy[:, 0]))
    up, lo = xy[:i + 1][::-1], xy[i:]
    text = "led\n" + f"{len(up)}. {len(lo)}.\n" + "\n".join(f"{x:.6f} {y:.6f}" for x, y in up) + "\n\n" + "\n".join(f"{x:.6f} {y:.6f}" for x, y in lo) + "\n"
    (tmp_path / "l.dat").write_text(text)
    r2 = zephyr.from_dat(tmp_path / "l.dat")
    assert np.allclose(np.array(r2.points), xy, atol=1e-6)


def test_bad_designations():
    for bad in ("12", "230112", "23212", "21112", "27012"):
        with pytest.raises(ValueError):
            naca(bad)
