"""Air propeller pod (notebook 25): the pod and pylon in both layouts, the movie outline, the pylon wake models and
the sound synthesis.

Run: cd /home/user/vegeta && python3 -m pytest -q notebooks/designs/tests/test_air_propeller.py
"""
import math
import wave

import numpy as np
import pytest

import air_propeller as ap


@pytest.mark.parametrize("layout", ap.LAYOUTS)
def test_pod_builds_on_the_right_side_of_the_disc(layout):
    s = ap.PropPod().generate(layout=layout).shape
    bb = s.BoundingBox()
    assert s.isValid() and len(s.Solids()) == 1
    assert bb.ymax == pytest.approx(230.0, abs=0.5) and bb.zmax == pytest.approx(23.0, abs=0.5)
    if layout == "tractor":
        assert bb.xmin == pytest.approx(8.0, abs=0.1) and bb.xmax > 200
    else:
        assert bb.xmax == pytest.approx(-8.0, abs=0.1) and bb.xmin < -200


def test_outline_matches_the_layout():
    t, p = ap.outline({"layout": "tractor"}), ap.outline({"layout": "pusher"})
    assert t["side"][0][:, 0].min() > 0 and p["side"][0][:, 0].max() < 0
    assert t["side"][1][:, 0].min() == pytest.approx(0.064) and p["side"][1][:, 0].max() == pytest.approx(-0.064)
    assert t["side"][1][:, 1].max() == pytest.approx(0.23)


def test_pusher_wake_is_deep_and_narrow_tractor_blockage_is_weak():
    R = 127.0
    pw, tw = ap.pylon_wake({"layout": "pusher"}, R), ap.pylon_wake({"layout": "tractor"}, R)
    xc = 64 / 90
    assert pw(0.7, 0.0) == pytest.approx(2.42 * math.sqrt(0.012) / (xc + 0.3), rel=1e-3)
    assert pw(0.7, 30.0) < 1e-3 and pw(0.12, 0.0) == 0.0               # narrow; nothing inside the pod
    assert tw(0.7, 0.0) == pytest.approx(0.12 * 90 / 2 / (math.pi * 64), rel=1e-3)
    hp, ht = pw.harmonics(0.7, 10), tw.harmonics(0.7, 10)
    assert hp[9] > 0.8 * hp[0] and ht[9] < 0.05 * ht[0]                 # a narrow wake keeps its high orders


def test_synthesized_sound_has_the_requested_levels(tmp_path):
    x = ap.synthesize([(200.0, 74.0)], -np.inf, seconds=2.0)
    rms = np.sqrt(np.mean(x[4410:-4410] ** 2))
    assert 20 * math.log10(rms / 20e-6) == pytest.approx(74.0, abs=0.1)
    y = ap.synthesize([], 60.0, seconds=2.0)
    assert 20 * math.log10(np.sqrt(np.mean(y[4410:-4410] ** 2)) / 20e-6) == pytest.approx(60.0, abs=0.3)
    f = ap.write_wav(tmp_path / "a.wav", x, full_scale_pa=1.0)
    with wave.open(str(f)) as w:
        assert w.getframerate() == 44100 and w.getnframes() == len(x) and w.getsampwidth() == 2


POD = dict(pod_diameter=46.0, pod_length=240.0, pylon_chord=90.0, pylon_thickness=0.12, pylon_height=230.0, pylon_gap=64.0)


def test_pod_friction_is_a_flat_plate_with_a_form_factor():
    f = ap.pod_friction(dict(POD, layout="pusher"), 20.0)
    re = 20.0 * 0.24 / 1.5e-5
    assert f["cf"] == pytest.approx(0.455 / math.log10(re) ** 2.58)
    assert f["form_factor"] == pytest.approx(1 + 1.5 * (46 / 240) ** 1.5 + 7 * (46 / 240) ** 3)
    assert 0.02 < f["segment_area_m2"].sum() < math.pi * 0.046 * 0.24       # less than the circumscribed cylinder
    assert f["cd_area_m2"] == pytest.approx(f["cf"] * f["form_factor"] * f["segment_area_m2"].sum())


def test_disc_solid_angle_on_and_off_the_axis():
    x, a = 0.02, 0.1
    assert float(ap.disc_solid_angle(x, 0.0, a)) == pytest.approx(2 * math.pi * (1 - x / math.hypot(x, a)), rel=1e-9)
    for (x, rho, a) in ((0.008, 0.015, 0.0127), (0.05, 0.2, 0.127), (0.002, 0.05, 0.127)):
        s_ = np.linspace(0, a, 2001)
        th = np.linspace(0, 2 * np.pi, 2001)
        S, T = np.meshgrid(s_, th, indexing="ij")
        brute = np.trapezoid(np.trapezoid(x * S / (x ** 2 + rho ** 2 + S ** 2 - 2 * rho * S * np.cos(T)) ** 1.5, th, axis=1), s_)
        assert float(ap.disc_solid_angle(x, rho, a)) == pytest.approx(brute, rel=1e-4)
    assert float(ap.disc_solid_angle(1e-6, 0.05, 0.1)) == pytest.approx(2 * math.pi, rel=1e-3)   # just above the disc


def test_bodies_are_closed_and_the_wake_follows_the_layout():
    for lay in ap.LAYOUTS:
        p = ap._defaults(dict(POD, layout=lay))
        xs, qs = ap._line_sources(ap._body_area(p))
        assert abs(qs.sum()) < 1e-12 and qs.max() > 0 and qs.min() < 0           # a closed body: sources and sinks cancel
        X, Y, q, _, _ = ap._pylon_sheet(p)
        assert np.allclose(q.sum(axis=0), 0.0, atol=1e-12)                       # the strut closes along every chord
    t = ap.installation_wake(dict(POD, layout="tractor"), 127.0, 20.0)
    u = ap.installation_wake(dict(POD, layout="pusher"), 127.0, 20.0)
    assert 0.0 < t.mean(0.7) < 0.01 and t(0.09, 90.0) > t(0.7, 90.0) > 0       # blockage ahead of the bodies, strongest near the hub
    assert u(0.7, 0.0) > 0.2 and u.mean(0.7) > t.mean(0.7)                       # the pylon's viscous lane behind a pusher
    half_body = ap.pylon_wake(dict(POD, layout="tractor"), 127.0)
    assert t(0.7, 0.0) - t(0.7, 180.0) < 0.6 * half_body(0.7, 0.0)               # a closed strut blocks less than a half-body
    hp, hu = u.harmonics(0.7, 10), ap.pylon_wake(dict(POD, layout="pusher"), 127.0).harmonics(0.7, 10)
    assert np.allclose(hp[4:], hu[4:], rtol=0.05) and 1.0 < hp[0] / hu[0] < 1.5  # high orders: the viscous lane; low ones: + the bodies
    far = ap.potential_wake_w(ap._defaults(dict(POD, layout="tractor", gap=2000.0, pylon_gap=2000.0)), np.array([0.09]), np.array([90.0]))
    assert abs(float(far[0, 0])) < 1e-3


def test_loaded_disc_reduces_to_the_uniform_disc():
    p = dict(POD, layout="pusher")
    R, T = 0.127, 5.0
    r = np.linspace(0.0005, R - 0.0005, 127)                 # uniform jump T/A from (almost) the axis to the tip
    dTdr = T / (math.pi * R ** 2) * 2 * math.pi * r
    loaded = ap.installation_drag(p, 127.0, 20.0, T, loading=(r, dTdr))
    uniform = ap.installation_drag(p, 127.0, 20.0, T)
    for k in ("pressure_N", "pylon_pressure_N", "pod_friction_N", "pylon_N"):
        assert loaded[k] == pytest.approx(uniform[k], rel=0.03, abs=1e-5), k
    root_light = dTdr * np.clip((r - 0.03) / 0.03, 0, 1) ** 2                    # no load where the pod is
    light = ap.installation_drag(p, 127.0, 20.0, T, loading=(r, root_light * T / np.trapezoid(root_light, r)))
    assert light["pressure_N"] < uniform["pressure_N"]


def test_installation_drag_signs_and_limits():
    for lay in ap.LAYOUTS:
        p = dict(POD, layout=lay)
        zero = ap.installation_drag(p, 127.0, 20.0, 0.0)
        assert zero["total_N"] == 0.0 and zero["t"] == 0.0
        d = ap.installation_drag(p, 127.0, 20.0, 5.0)
        assert d["v_i"] * (20.0 + d["v_i"]) * 2 * 1.225 * math.pi * 0.127 ** 2 == pytest.approx(5.0)
        assert d["pressure_N"] > 0 and d["pod_friction_N"] > 0 and d["pylon_N"] >= 0 and d["pylon_pressure_N"] > 0
        assert 0.0 < d["t"] < 0.05                                           # a thin pod in a large disc
        assert ap.installation_drag(p, 127.0, 20.0, 10.0)["total_N"] > d["total_N"]
    far = ap.installation_drag(dict(POD, layout="pusher", gap=3000.0), 127.0, 20.0, 5.0)
    assert far["pressure_N"] < 0.01 * ap.installation_drag(dict(POD, layout="pusher"), 127.0, 20.0, 5.0)["pressure_N"]


def test_installation_drag_leaves_out_the_motor_end_face():
    for lay in ap.LAYOUTS:
        p = dict(POD, layout=lay, hub_height=10.0, gap=3.0)
        d = ap.installation_drag(p, 127.0, 20.0, 5.0)
        prof = np.array(ap.PropPod.pod_profile(ap._defaults(p))) / 1000.0
        x, r = prof[:, 0], prof[:, 1]
        x_face = 0.008 if lay == "tractor" else -0.008
        face = (np.abs(x[:-1] - x_face) < 1e-9) & (np.abs(x[1:] - x_face) < 1e-9)
        assert face.sum() == 1                                               # exactly one flat end face, at the disc side
        R, T = 0.127, 5.0
        xm, rm = 0.5 * (x[1:] + x[:-1]), 0.5 * (r[1:] + r[:-1])
        dp = np.sign(xm) * T / (math.pi * R ** 2) * ap.disc_solid_angle(xm, rm, R) / (4 * math.pi)
        ann = np.pi * (r[:-1] ** 2 - r[1:] ** 2)
        assert d["pressure_N"] == pytest.approx(float(np.sum((-dp * ann)[~face])), rel=1e-9)
