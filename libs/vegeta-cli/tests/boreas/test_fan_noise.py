"""Ducted-fan rotor-stator interaction noise: Tyler–Sofrin modes and cut-on, rotor wakes, Sears, the ring of vane dipoles."""
import math

import numpy as np
import pytest

from vegeta import boreas
from vegeta.boreas import fan_noise as fn

# the EDF of notebook 25: 12 blades, ~90 mm rotor, vanes ~1 chord behind it
B, RPM, R_TIP, R_VANE = 12, 25000.0, 0.045, 0.036
OMEGA = RPM * 2 * math.pi / 60
C = boreas.AIR.speed_of_sound


def tones(vanes, gust=(0.0, 5.0, 3.0, 1.5, 0.5), **kw):
    args = dict(vane_radius_m=R_VANE, vane_chord_m=0.015, vane_span_m=0.025, vane_inflow_m_s=55.0, gust_m_s=gust,
                distance=1.0, medium=boreas.AIR, angle_deg=45.0, stagger_deg=20.0)
    args.update(kw)
    return fn.interaction_tones(B, vanes, RPM, **args)


def brute_force(vanes, row, R, stagger_deg, angle_deg, azimuth_deg, distance):
    """|p| / sqrt2 of V compact dipoles summed one by one: vane v at phi_v carries the force L (-sin xi, cos xi) in
    (x, phi) with the phase -m B phi_v, observed far away in the direction x_hat."""
    m, L = row["harmonic"], row["vane_force_N"]
    w = m * B * OMEGA
    k = w / C
    th, ph, xi = (math.radians(a) for a in (angle_deg, azimuth_deg, stagger_deg))
    xh = np.array([math.cos(th), math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph)])
    P = 0j
    for v in range(vanes):
        pv = 2 * math.pi * v / vanes
        y = np.array([0.0, R * math.cos(pv), R * math.sin(pv)])
        F = L * (-math.sin(xi) * np.array([1.0, 0.0, 0.0]) + math.cos(xi) * np.array([0.0, -math.sin(pv), math.cos(pv)]))
        P += xh @ F * np.exp(1j * k * (xh @ y)) * np.exp(-1j * m * B * pv)
    return w / (4 * math.pi * C * distance) * abs(P) / math.sqrt(2)


def first_modes(blades, vanes, harmonics=3, **kw):
    """The first row (lowest |n|) of every harmonic of tyler_sofrin_modes, as (n, s)."""
    modes = fn.tyler_sofrin_modes(blades, vanes, harmonics, **kw)
    return [next((d["n"], d["s"]) for d in modes if d["harmonic"] == m) for m in range(1, harmonics + 1)]


def test_tyler_sofrin_mode_selection():
    modes = fn.tyler_sofrin_modes(B, 7, harmonics=3, s_range=4)
    assert all(d["n"] == d["harmonic"] * B - d["s"] * 7 for d in modes)
    for m, s0 in ((1, 2), (2, 3), (3, 5)):                           # s0: the multiple of 7 nearest 12 m
        own = [d for d in modes if d["harmonic"] == m]
        # s = -4..4 and s0 - 4..s0 + 4, each once
        assert sorted(d["s"] for d in own) == sorted(set(range(-4, 5)) | set(range(s0 - 4, s0 + 5)))
        assert [abs(d["n"]) for d in own] == sorted(abs(d["n"]) for d in own)
    # the lowest mode comes first even when m B / V > s_range: 12 - 2 x 7, 24 - 3 x 7, 36 - 5 x 7
    assert first_modes(B, 7) == [(-2, 2), (3, 3), (1, 5)]
    assert first_modes(B, 5) == [(2, 2), (-1, 5), (1, 7)]
    assert first_modes(B, 3) == [(0, 4), (0, 8), (0, 12)]
    assert first_modes(B, 1) == [(0, 12), (0, 24), (0, 36)]               # one vane: the plane wave n = mB - mB
    assert first_modes(B, 7, s_range=0) == [(-2, 2), (3, 3), (1, 5)]      # only the lowest mode
    # = the lowest mode of vane_count_study (ties to the smaller s), for every vane count
    for V in range(1, 40):
        study = fn.vane_count_study(B, [V], 0.4, harmonics=4)[0]["modes"]
        assert first_modes(B, V, 4) == [(d["n"], d["s"]) for d in study], V
    # as many vanes as blades: a plane wave (n = 0) at every harmonic
    for m in (1, 2, 3):
        assert [d for d in fn.tyler_sofrin_modes(B, B, 3) if d["harmonic"] == m][0]["n"] == 0
    with pytest.raises(ValueError):
        fn.tyler_sofrin_modes(0, 7)
    with pytest.raises(ValueError):
        fn.tyler_sofrin_modes(B, 7.5)


def test_cut_on_ratio():
    assert fn.cut_on_ratio(1, B, -2, 0.4) == pytest.approx(2.4)
    assert fn.cut_on_ratio(2, B, 5, 0.4, radius_ratio=0.5) == pytest.approx(0.96)
    assert fn.cut_on_ratio(1, B, 0, 0.4) == math.inf
    assert fn.cut_on_ratio(1, B, B, 0.4) == pytest.approx(0.4)          # the rotor-locked mode n = mB: cut on only above M = 1
    with pytest.raises(ValueError):
        fn.cut_on_ratio(1, B, 2, -0.1)


def test_vane_count_study_is_the_tyler_sofrin_rule():
    rows = {r["vanes"]: r for r in fn.vane_count_study(B, range(1, 61), 0.4, harmonics=2)}
    seven = rows[7]["modes"][0]
    assert (seven["n"], seven["s"], seven["cut_on"]) == (-2, 2, True) and seven["cut_on_ratio"] == pytest.approx(2.4)
    assert not rows[7]["bpf_cut_off"] and rows[7]["cut_on_harmonics"] == [1, 2]
    assert rows[25]["bpf_cut_off"] and rows[25]["modes"][0]["n"] == 12 and rows[25]["modes"][0]["cut_on_ratio"] == pytest.approx(0.4)
    assert rows[12]["modes"][0]["n"] == 0 and rows[12]["modes"][0]["cut_on"]
    for V, r in rows.items():
        if V > B * 1.4:                                    # V > B (1 + M_tip): 1 x BPF always cut off
            assert r["bpf_cut_off"], V
        if V < 2 * B * 0.4:                                # V < 2 B M_tip: never
            assert not r["bpf_cut_off"], V
        for d in r["modes"]:                               # the lowest |n| is the first of tyler_sofrin_modes
            ts = [x for x in fn.tyler_sofrin_modes(B, V, d["harmonic"], s_range=2 * B + 2) if x["harmonic"] == d["harmonic"]]
            assert (d["n"], d["s"]) == (ts[0]["n"], ts[0]["s"])
    assert min(V for V, r in rows.items() if r["bpf_cut_off"]) == 17
    assert min(V for V, r in rows.items() if not r["cut_on_harmonics"]) == 34


def test_sears():
    assert fn.sears(0.0) == 1.0
    k = np.linspace(0.0, 20.0, 41)
    s = fn.sears(k)
    assert isinstance(s, np.ndarray) and np.all(np.diff(s) < 0)
    assert fn.sears(200.0) * math.sqrt(2 * math.pi * 200.0) == pytest.approx(1.0, rel=2e-3)
    with pytest.raises(ValueError):
        fn.sears(-0.1)


def test_rotor_wake_harmonics():
    chord, cd, spacing = 0.015, 0.03, 0.015
    a = fn.rotor_wake_harmonics(B, chord, cd, spacing, R_VANE, harmonics=6)
    xc = spacing / chord
    u_c = 1.21 * math.sqrt(cd) / (xc + 0.3)                                    # half Silverstein's total-head loss 2.42 ...
    b = 0.34 * chord * math.sqrt(cd * (xc + 0.15))                             # ... half his edge half width 0.68 ...
    pitch = 2 * math.pi * R_VANE / B
    mean = u_c * b * math.sqrt(math.pi / math.log(2)) / pitch                 # the wake's area over the pitch
    assert len(a) == 7 and a[0] == pytest.approx(mean, rel=1e-9)
    # the wake's momentum deficit is the section drag's, cd c / 2, within the near-wake fit: 1.4x at x = c, 1.1x at 2 c
    for x, ratio in ((1.0, 1.445), (2.0, 1.117), (4.0, 0.830)):
        area = fn.rotor_wake_harmonics(B, chord, cd, x * chord, R_VANE)[0] * pitch
        assert area / (cd * chord / 2) == pytest.approx(ratio, abs=2e-3)
    m = np.arange(1, 7)
    assert a[1:] == pytest.approx(2 * mean * np.exp(-(math.pi * m * b / pitch) ** 2 / math.log(2)), rel=1e-6)
    assert np.all(np.diff(a[1:]) < 0)
    # further downstream the wake is wider and shallower: every harmonic weaker, the higher ones most
    far = fn.rotor_wake_harmonics(B, chord, cd, 3 * spacing, R_VANE, harmonics=6)
    ratio = far[1:] / a[1:]
    assert np.all(ratio < 1) and np.all(np.diff(ratio) < 0)
    # a wake travelling at 60 deg from the axis: twice the path, twice as wide along the circumference
    sl = fn.rotor_wake_harmonics(B, chord, cd, spacing, R_VANE, harmonics=6, wake_angle_deg=60.0)
    u2 = 1.21 * math.sqrt(cd) / (2 * xc + 0.3)
    b2 = 2 * 0.34 * chord * math.sqrt(cd * (2 * xc + 0.15))
    assert sl[0] == pytest.approx(u2 * b2 * math.sqrt(math.pi / math.log(2)) / pitch, rel=1e-9) and sl[3] < a[3]
    assert np.all(fn.rotor_wake_harmonics(B, chord, 0.0, spacing, R_VANE) == 0.0)
    with pytest.raises(ValueError):
        fn.rotor_wake_harmonics(B, chord, cd, spacing, R_VANE, wake_angle_deg=90.0)


def test_single_vane_on_the_axis_is_a_compact_dipole():
    g = (0.0, 4.0, 2.0)
    # stagger 90 deg: the lift is all axial, F_a = L
    for angle in (0.0, 30.0, 120.0):
        for row in tones(1, gust=g, harmonics=2, vane_radius_m=0.0, stagger_deg=90.0, angle_deg=angle, azimuth_deg=40.0):
            w = row["frequency_hz"] * 2 * math.pi
            assert row["p_rms_Pa"] == pytest.approx(w * row["vane_force_N"] * abs(math.cos(math.radians(angle))) / (4 * math.pi * C) / math.sqrt(2), rel=1e-9)
            assert row["dominant_n"] == 0
    row = tones(1, gust=g, harmonics=1, vane_radius_m=0.0, stagger_deg=90.0, angle_deg=0.0)[0]
    U, chord, span = 55.0, 0.015, 0.025
    kred = B * OMEGA * chord / (2 * U)
    assert row["reduced_frequency"] == pytest.approx(kred) and row["sears"] == pytest.approx(1 / math.sqrt(1 + 2 * math.pi * kred))
    assert row["vane_force_N"] == pytest.approx(math.pi * boreas.AIR.density * chord * U * 4.0 * row["sears"] * span)
    # stagger 0: all tangential, along +z at vane 0 — silent in the x-y plane, sin(theta) across it
    flat = tones(1, gust=g, harmonics=1, vane_radius_m=0.0, stagger_deg=0.0, angle_deg=60.0, azimuth_deg=0.0)[0]
    side = tones(1, gust=g, harmonics=1, vane_radius_m=0.0, stagger_deg=0.0, angle_deg=60.0, azimuth_deg=90.0)[0]
    w = side["frequency_hz"] * 2 * math.pi
    assert flat["p_rms_Pa"] < 1e-12 * side["p_rms_Pa"]
    assert side["p_rms_Pa"] == pytest.approx(w * side["vane_force_N"] * math.sin(math.radians(60.0)) / (4 * math.pi * C) / math.sqrt(2), rel=1e-9)


@pytest.mark.parametrize("vanes", [1, 7, 12, 17])
def test_mode_sum_equals_the_vane_dipoles_one_by_one(vanes):
    for angle, azimuth in ((30.0, 0.0), (75.0, 50.0), (120.0, 200.0), (90.0, 10.0)):
        rows = tones(vanes, harmonics=4, angle_deg=angle, azimuth_deg=azimuth, stagger_deg=25.0, distance=2.0)
        for row in rows:
            assert row["p_rms_Pa"] == pytest.approx(brute_force(vanes, row, R_VANE, 25.0, angle, azimuth, 2.0), rel=1e-7)
            # the azimuthal rms is the energy sum of the modes
            assert row["p_rms_ring_Pa"] == pytest.approx(math.sqrt(sum(d["p_rms_Pa"] ** 2 for d in row["modes"])))
            assert all(d["n"] == row["harmonic"] * B - d["s"] * vanes for d in row["modes"])


def test_far_field_falls_as_one_over_r():
    near, far = tones(7, distance=1.0), tones(7, distance=10.0)
    for a, b in zip(near, far):
        assert b["spl_db"] == pytest.approx(a["spl_db"] - 20.0)


def test_equal_blade_and_vane_counts_give_a_plane_wave():
    for row in tones(B, angle_deg=30.0, stagger_deg=30.0):
        assert row["dominant_n"] == 0
    # on the axis only n = 0 radiates: all V vanes push in phase, V F_a sin(xi) as one compact dipole
    for row in tones(B, angle_deg=0.0, stagger_deg=30.0):
        w = row["frequency_hz"] * 2 * math.pi
        assert row["p_rms_Pa"] == pytest.approx(w * B * row["vane_force_N"] * 0.5 / (4 * math.pi * C) / math.sqrt(2), rel=1e-9)


def test_high_order_modes_are_cut_off():
    # at equal vane force: 11 vanes leave n = 1 at 1 x BPF (cut on), 25 vanes n = 12 (k R ~ 3.3: cut off)
    a, b = tones(11, harmonics=1)[0], tones(25, harmonics=1)[0]
    assert a["vane_force_N"] == pytest.approx(b["vane_force_N"])
    assert a["dominant_n"] == 1 and abs(b["dominant_n"]) >= 12
    assert 20 * math.log10((a["p_rms_Pa"] / 11) / (b["p_rms_Pa"] / 25)) > 60.0
    kr = B * OMEGA * R_VANE / C
    for d in a["modes"]:
        assert d["cut_on_ratio"] == pytest.approx(kr / abs(d["n"]))


def test_duct_attenuates_only_cut_off_modes():
    free = tones(17, harmonics=1)[0]
    duct = tones(17, harmonics=1, duct_radius_m=R_TIP, duct_length_m=0.03)[0]
    kRd = B * OMEGA * R_TIP / C                                            # ~4.2 at 1 x BPF
    for f, d in zip(free["modes"], duct["modes"]):
        jp = {1: 1.8411837813, 2: 3.0542369282, 5: 6.4156163757, 12: 13.8788430697}.get(abs(d["n"]))
        if jp is not None and jp < kRd:
            assert d["duct_attenuation_db"] == 0.0 and d["p_rms_Pa"] == pytest.approx(f["p_rms_Pa"])
        elif jp is not None:
            att = 0.03 * math.sqrt(jp ** 2 - kRd ** 2) / R_TIP * 20 / math.log(10)
            assert d["duct_attenuation_db"] == pytest.approx(att, rel=1e-6)
            assert d["p_rms_Pa"] == pytest.approx(f["p_rms_Pa"] * 10 ** (-att / 20), rel=1e-6)
    assert duct["p_rms_Pa"] < free["p_rms_Pa"] * 10 ** (-20 / 20)           # n = -5 is cut off: > 20 dB quieter
    seven = tones(7, harmonics=1, duct_radius_m=R_TIP, duct_length_m=0.03)[0]
    assert seven["dominant_n"] == -2 and [d for d in seven["modes"] if d["n"] == -2][0]["duct_attenuation_db"] == 0.0


def test_interaction_tones_validation():
    with pytest.raises(ValueError):
        tones(7, gust=(0.0, 1.0), harmonics=4)                             # gusts for harmonics 2..4 missing
    with pytest.raises(ValueError):
        tones(7, distance=0.0)
    with pytest.raises(ValueError):
        tones(7, vane_radius_m=-0.01)
    with pytest.raises(ValueError):
        tones(7, duct_radius_m=0.0, duct_length_m=0.01)
    water = fn.interaction_tones(B, 7, RPM, vane_radius_m=R_VANE, vane_chord_m=0.015, vane_span_m=0.025, vane_inflow_m_s=5.0,
                                 gust_m_s=(0.0, 0.5), distance=1.0, medium=boreas.SEA_WATER, harmonics=1)[0]
    assert water["spl_db"] == pytest.approx(20 * math.log10(water["p_rms_Pa"] / 1e-6))
