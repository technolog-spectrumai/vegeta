"""Resonators against Kogelnik & Li's closed forms, the stability edges, a ring against its unfolded linear twin, the
spectrum, sweeps and the JSON round trip."""
import math

import numpy as np
import pytest

from vegeta import hyperion as hy

LAM = 1064e-9


@pytest.mark.parametrize("R1,R2,L", [(0.5, math.inf, 0.2), (0.5, 0.5, 0.3), (1.0, 0.3, 0.25), (-0.5, 0.2, 0.1), (math.inf, 0.4, 0.3)])
def test_two_mirror_cavity_against_kogelnik_li(R1, R2, L):
    r = hy.two_mirror(R1, R2, L, LAM)
    kl = hy.kogelnik_li(R1, R2, L, LAM)
    m = r.mode()
    assert m.stable == kl["stable"]
    assert r.g_parameters()["g1g2"] == pytest.approx(kl["g1g2"])
    assert m.m == pytest.approx(2 * kl["g1g2"] - 1)                              # (A+D)/2 = 2 g₁g₂ − 1
    assert m.w0 == pytest.approx(kl["w0"], rel=1e-9)
    assert m.z0 == pytest.approx(kl["z0"], rel=1e-9, abs=1e-12)
    assert m.spot("M1") == pytest.approx(kl["w1"], rel=1e-9)
    assert m.spot("M2") == pytest.approx(kl["w2"], rel=1e-9)
    # the mode reproduces itself: the beam at the reference plane after a full round trip
    q = m.beam.q
    assert hy.propagate_q(q, r.round_trip()) == pytest.approx(q, rel=1e-9)
    # the Gouy phase: one-way arccos(√(g₁g₂)) for g₁, g₂ > 0 (all the cases above), twice per round trip
    assert kl["g1"] > 0 and kl["g2"] > 0
    assert m.gouy_round_trip == pytest.approx(2 * math.acos(math.sqrt(kl["g1g2"])), abs=1e-9)


def test_stability_edges_and_the_confocal_limit():
    L = 0.2
    assert hy.two_mirror(L, L, L * (1 - 1e-6), LAM).stable()                     # just inside the confocal point
    assert hy.two_mirror(L, L, L, LAM).mode().stable is False                    # the confocal point itself: g₁g₂ = 0, m = −1, degenerate
    assert hy.two_mirror(L, L, 2 * L * (1 - 1e-9), LAM).stable()                 # just inside the concentric edge
    assert not hy.two_mirror(L, L, 2 * L * (1 + 1e-9), LAM).stable()             # past it
    assert not hy.two_mirror(math.inf, math.inf, L, LAM).stable()                # plane-plane: marginal (m = 1)
    assert not hy.two_mirror(-0.5, math.inf, L, LAM).stable()                    # convex-flat: unstable
    # near-confocal symmetric cavity: w0² → λL/2π, the mirror spot √2 w0, the round-trip Gouy phase → π (degenerate transverse modes)
    m = hy.two_mirror(L, L, L * (1 - 1e-6), LAM).mode()
    assert m.w0 == pytest.approx(math.sqrt(LAM * L / (2 * math.pi)), rel=1e-5)
    assert m.spot("M2") == pytest.approx(math.sqrt(2) * m.w0, rel=1e-5)
    assert m.gouy_round_trip == pytest.approx(math.pi, abs=1e-2)
    s = hy.two_mirror(L, L, L * (1 - 1e-6), LAM).spectrum()
    assert s["transverse_mode_spacing"] == pytest.approx(s["fsr"] / 2, rel=1e-2)


def test_ring_equals_its_unfolded_linear_twin():
    """A symmetric linear cavity M1–L–M2 unfolded: the ring [M2, L, M1, L] (flat M1 at 0°) has the same mode at M2."""
    lin = hy.two_mirror(0.5, 0.5, 0.3, LAM)
    ring = hy.Resonator(hy.Train([hy.CurvedMirror(0.5, name="M2"), hy.FreeSpace(0.3), hy.CurvedMirror(0.5, name="M1"), hy.FreeSpace(0.3)]), LAM, ring=True)
    ml, mr = lin.mode(), ring.mode()
    assert mr.stable and mr.w0 == pytest.approx(ml.w0, rel=1e-9)
    assert mr.spot("M2") == pytest.approx(ml.spot("M2"), rel=1e-9) and mr.spot("M1") == pytest.approx(ml.spot("M1"), rel=1e-9)
    assert ring.spectrum()["fsr"] == pytest.approx(lin.spectrum()["fsr"])         # the loop is twice the linear length


def test_astigmatic_folded_cavity():
    """A folding mirror at an angle: tangential and sagittal modes differ; at 0° they coincide."""
    def cav(angle):
        return hy.Resonator(hy.Train([hy.FlatMirror(name="M1"), hy.FreeSpace(0.2), hy.CurvedMirror(0.5, angle, name="fold"), hy.FreeSpace(0.2),
                                      hy.FlatMirror(name="M2")]), LAM)
    straight, folded = cav(0.0), cav(20.0)
    assert not straight.astigmatic() and folded.astigmatic()
    ms = straight.modes()
    assert ms["t"].w0 == pytest.approx(ms["s"].w0)
    mf = folded.modes()
    assert mf["t"].stable and mf["s"].stable and mf["t"].w0 != pytest.approx(mf["s"].w0, rel=1e-3)
    assert mf["t"].spot("fold") < mf["s"].spot("fold")                           # the tangential focal length is shorter


def test_spectrum():
    r = hy.two_mirror(0.5, math.inf, 0.2, LAM, reflectivity=(0.999, 0.95), loss=0.01)
    s = r.spectrum()
    c = 299792458.0
    assert s["fsr"] == pytest.approx(c / 0.4)
    assert s["round_trip_time"] == pytest.approx(0.4 / c)
    survival = 0.999 * 0.95 * 0.99
    assert s["survival"] == pytest.approx(survival)
    assert s["photon_lifetime"] == pytest.approx(0.4 / c / -math.log(survival))
    ramp = math.sqrt(survival)
    assert s["finesse"] == pytest.approx(math.pi * math.sqrt(ramp) / (1 - ramp))
    assert s["linewidth"] == pytest.approx(s["fsr"] / s["finesse"])
    assert s["Q"] == pytest.approx(2 * math.pi * c / LAM * s["photon_lifetime"])
    # a slab inside lengthens the optical path
    rs = hy.Resonator(hy.Train([hy.FlatMirror(name="M1"), hy.FreeSpace(0.1), hy.Slab(0.01, "Nd:YAG", name="rod"), hy.FreeSpace(0.09), hy.CurvedMirror(0.5, name="M2")]), LAM)
    assert rs.optical_length() == pytest.approx(0.19 + 0.01 * hy.YAG.n(LAM))
    assert rs.mode().spot("rod in") == pytest.approx(rs.mode().spot("rod"), rel=0.05)    # a short rod: nearly the same spot at both faces


def test_thermal_lens_sweep_has_a_stability_window():
    """A flat–flat cavity with a lens inside: unstable cold (m = 1), stable for 0 < 1/f < 4/L (two spaces L/2) except
    the degenerate point 1/f = 2/L where the lens images one mirror onto the other (round trip −I), unstable past 4/L;
    the half-trace is 2(1 − PL/2)² − 1."""
    L = 0.2

    def build(power):
        f = math.inf if power == 0 else 1.0 / power
        return hy.Resonator(hy.Train([hy.FlatMirror(name="M1"), hy.FreeSpace(L / 2), hy.ThermalLens(f, name="tl"), hy.FreeSpace(L / 2), hy.FlatMirror(name="M2")]), LAM)

    sw = hy.Resonator.sweep(build, np.linspace(0, 30, 61), elements=["M1", "M2"])
    P = sw["values"]
    assert sw["m"] == pytest.approx(2 * (1 - P * L / 2) ** 2 - 1)
    edge, degenerate = 4.0 / L, 2.0 / L                                           # 20 and 10 dioptres
    expected = (P > 0) & (P < edge) & (P != degenerate)
    assert sw["stable"].tolist() == expected.tolist()
    assert np.isnan(sw["w0"][0]) and np.all(np.isfinite(sw["spots"]["M1"][expected])) and np.all(np.isnan(sw["w0"][~expected]))
    assert sw["spots"]["M1"] == pytest.approx(sw["spots"]["M2"], nan_ok=True)      # symmetric
    assert np.nanargmax(sw["w0"]) == 1 and sw["w0"][39] < sw["w0"][1] / 5         # the mode shrinks towards the outer edge


def test_json_round_trip_and_summary(tmp_path):
    r = hy.Resonator(hy.Train([hy.CurvedMirror(0.5, name="M1"), hy.FreeSpace(0.05), hy.Slab(0.01, "Nd:YAG", name="rod"), hy.FreeSpace(0.14),
                               hy.FlatMirror(0.95, name="OC")]), LAM, loss=0.01, name="svarog")
    p = hy.save_design(tmp_path / "d.json", {"cavity": r, "beam": hy.GaussianBeam(LAM, 1e-4), "f": math.inf, "x": [1, 2]})
    d = hy.load_design(p)
    assert isinstance(d["cavity"], hy.Resonator) and d["cavity"].mode().w0 == pytest.approx(r.mode().w0, rel=1e-12)
    assert isinstance(d["beam"], hy.GaussianBeam) and d["f"] == math.inf and d["x"] == [1, 2]
    s = r.summary()
    assert s["stable_t"] and s["stable_s"] and set(s["spots_t"]) == {"M1", "FreeSpace", "rod in", "rod", "rod out", "OC"}
    with pytest.raises(ValueError):
        hy.Resonator(hy.Train([hy.FreeSpace(0.1), hy.FlatMirror()]), LAM)          # must start with a mirror
