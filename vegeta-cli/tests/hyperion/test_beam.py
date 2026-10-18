"""Gaussian beams against the closed forms: w(z), R(z), the Gaussian lens formula (Self 1983), M² through a lens, the
mode overlap, a fibre's beam."""
import math

import numpy as np
import pytest

from vegeta import hyperion as hy

LAM = 1064e-9


def test_closed_form_w_and_R():
    b = hy.GaussianBeam(LAM, 200e-6)
    zr = math.pi * 200e-6 ** 2 / LAM
    assert b.rayleigh_range == pytest.approx(zr)
    assert b.w(zr) == pytest.approx(200e-6 * math.sqrt(2))
    assert b.R(zr) == pytest.approx(2 * zr) and math.isinf(b.R(0.0))
    assert b.divergence == pytest.approx(LAM / (math.pi * 200e-6))
    z = np.linspace(-1, 1, 7)
    assert b.w(z) == pytest.approx(200e-6 * np.sqrt(1 + (z / zr) ** 2))
    # in a medium of index n the Rayleigh range grows n times, the spot is the same
    bn = hy.GaussianBeam(LAM, 200e-6, n=1.8)
    assert bn.rayleigh_range == pytest.approx(1.8 * zr)
    # the q parameter round trip
    q = b.q_at(0.3)
    assert hy.GaussianBeam.from_q(q, LAM).w0 == pytest.approx(200e-6) and hy.GaussianBeam.from_q(q, LAM).z0 == pytest.approx(-0.3)
    w, R = hy.spot_from(q, LAM)
    assert w == pytest.approx(b.w(0.3)) and R == pytest.approx(b.R(0.3))
    assert hy.q_from(w, R, LAM) == pytest.approx(q)
    assert hy.GaussianBeam.from_spot(w, R, LAM).w0 == pytest.approx(200e-6)


def test_free_space_trace_matches_w_of_z():
    b = hy.GaussianBeam(LAM, 100e-6, z0=0.1)                  # the waist 0.1 m ahead
    tr = b.through([hy.FreeSpace(0.5, name="air")], samples=200)
    assert tr.w == pytest.approx(b.w(tr.z), rel=1e-12)
    assert len(tr.waists) == 1 and tr.waists[0]["z"] == pytest.approx(0.1) and tr.waists[0]["w0"] == pytest.approx(100e-6)
    assert tr.out.w0 == pytest.approx(100e-6) and tr.out.z0 == pytest.approx(-0.4)


def test_gaussian_lens_formula():
    """A thin lens f at distance z from a waist w0 (Self): w0' = w0 f / √((z − f)² + z_R²), the new waist at
    z' = f + (z − f) f² / ((z − f)² + z_R²)."""
    w0, f, z = 200e-6, 0.1, 0.5
    b = hy.GaussianBeam(LAM, w0)
    tr = b.through([hy.FreeSpace(z), hy.ThinLens(f, name="lens"), hy.FreeSpace(0.3)])
    zr = b.rayleigh_range
    den = (z - f) ** 2 + zr ** 2
    w0p = w0 * f / math.sqrt(den)
    zp = f + (z - f) * f * f / den
    assert len(tr.waists) == 1
    assert tr.waists[0]["w0"] == pytest.approx(w0p, rel=1e-9)
    assert tr.waists[0]["z"] - z == pytest.approx(zp, rel=1e-9)
    assert tr.spot_at("lens")["w"] == pytest.approx(b.w(z))
    # the inverse problem
    sol = hy.lens_for_waist(hy.GaussianBeam(LAM, w0, z0=-z), w0p, distance=zp)
    assert sol["f"] == pytest.approx(f, rel=1e-9) and sol["waist_distance"] == pytest.approx(zp, rel=1e-9)
    with pytest.raises(ValueError):
        hy.lens_for_waist(hy.GaussianBeam(LAM, w0, z0=-z), 10 * b.w(z))       # larger than the spot on the lens
    # the geometric limit: a collimated beam (z_R ≫ f) focuses at f
    big = hy.GaussianBeam(LAM, 5e-3)
    tr2 = big.through([hy.ThinLens(0.1), hy.FreeSpace(0.2)])
    assert tr2.waists[0]["z"] == pytest.approx(0.1, rel=1e-3)
    assert tr2.waists[0]["w0"] == pytest.approx(LAM * 0.1 / (math.pi * 5e-3), rel=1e-3)


def test_m2_beam_focuses_m2_times_larger():
    """The embedded Gaussian: the same lens focuses a collimated M² beam to an M² times larger spot at the same place;
    an uncollimated one follows the lens formula with z_R/M²."""
    train = [hy.FreeSpace(0.2), hy.ThinLens(0.05, name="lens"), hy.FreeSpace(0.1)]
    g = hy.GaussianBeam(LAM, 5e-3).through(train)                              # z_R = 74 m: collimated
    p = hy.GaussianBeam(LAM, 5e-3, m2=4.0).through(train)
    assert p.waists[0]["w0"] / g.waists[0]["w0"] == pytest.approx(4.0, rel=1e-4)
    assert p.waists[0]["z"] == pytest.approx(g.waists[0]["z"], rel=1e-4)
    w0, f, z = 1e-3, 0.05, 0.2
    q = hy.GaussianBeam(LAM, w0, m2=4.0).through(train)
    zr = math.pi * w0 ** 2 / (4.0 * LAM)
    assert q.waists[0]["w0"] == pytest.approx(w0 * f / math.sqrt((z - f) ** 2 + zr ** 2), rel=1e-9)
    assert q.spot_at("lens")["w"] == pytest.approx(w0 * math.sqrt(1 + (z / zr) ** 2))
    assert hy.GaussianBeam(LAM, 1e-3, m2=4.0).divergence == pytest.approx(4 * hy.GaussianBeam(LAM, 1e-3).divergence)
    fb = hy.GaussianBeam.from_fibre(808e-9, 50e-6, 0.22)
    assert fb.divergence == pytest.approx(0.22) and fb.m2 == pytest.approx(math.pi * 50e-6 * 0.22 / 808e-9)
    assert hy.GaussianBeam.from_fibre(1064e-9, 3e-6, 0.1).m2 == 1.0            # single mode: clamped at 1


def test_slab_keeps_the_spot_continuous():
    """Through a slab the spot is continuous at both faces and the beam inside spreads n times more slowly."""
    b = hy.GaussianBeam(LAM, 150e-6, z0=0.0)
    tr = b.through([hy.Slab(0.02, 1.8, name="rod"), hy.FreeSpace(0.0)])
    s = {st["name"]: st for st in tr.stations}
    assert s["rod in"]["w"] == pytest.approx(150e-6)
    assert s["rod"]["w"] == pytest.approx(b.w(0.02 / 1.8))                     # 20 mm of n = 1.8 is 11.1 mm of air
    assert s["rod out"]["w"] == pytest.approx(s["rod"]["w"])
    assert tr.out.z0 == pytest.approx(-0.02 / 1.8)


def test_overlap():
    b = hy.GaussianBeam(LAM, 200e-6)
    assert b.overlap(b) == pytest.approx(1.0)
    assert b.overlap(hy.GaussianBeam(LAM, 400e-6)) == pytest.approx(4.0 / (0.5 + 2.0) ** 2)   # 0.64, the spot-size mismatch alone
    tilted = hy.GaussianBeam(LAM, 200e-6, z0=0.2)                                # same spot size at z = 0? no: different w and R
    assert 0 < b.overlap(tilted) < 1
    with pytest.raises(ValueError):
        b.overlap(hy.GaussianBeam(532e-9, 200e-6))
