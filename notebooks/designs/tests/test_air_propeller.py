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


def test_installation_wake_adds_the_pod_to_the_mean_not_the_harmonics():
    for lay in ap.LAYOUTS:
        p = dict(POD, layout=lay)
        full, pylon = ap.installation_wake(p, 127.0, 20.0), ap.pylon_wake(p, 127.0, r_frac=(0.06, 0.09, 0.12, 0.16, 0.2, 0.28, 0.35, 0.5, 0.7, 0.85, 1.0))
        assert np.allclose(full.harmonics(0.7, 12), pylon.harmonics(0.7, 12), atol=1e-12)
        assert full.mean(0.09) > pylon.mean(0.09) and full.mean(0.09) > full.mean(0.7)
    a, xs = 0.023, 0.005 + 0.003 + 0.0115                                    # tractor: the Rankine nose on the axis
    t = ap.installation_wake(dict(POD, layout="tractor"), 127.0, 20.0, r_frac=(0.0, 0.5, 1.0))
    assert t(0.0, 90.0) == pytest.approx(a ** 2 / (4 * xs ** 2), rel=1e-9)


def test_installation_drag_signs_and_limits():
    for lay in ap.LAYOUTS:
        p = dict(POD, layout=lay)
        zero = ap.installation_drag(p, 127.0, 20.0, 0.0)
        assert zero["total_N"] == 0.0 and zero["t"] == 0.0
        d = ap.installation_drag(p, 127.0, 20.0, 5.0)
        assert d["v_i"] * (20.0 + d["v_i"]) * 2 * 1.225 * math.pi * 0.127 ** 2 == pytest.approx(5.0)
        assert d["pressure_N"] > 0 and d["pod_friction_N"] > 0 and d["pylon_N"] >= 0
        assert 0.0 < d["t"] < 0.05                                           # a thin pod in a large disc
        assert ap.installation_drag(p, 127.0, 20.0, 10.0)["total_N"] > d["total_N"]
    far = ap.installation_drag(dict(POD, layout="pusher", gap=3000.0), 127.0, 20.0, 5.0)
    assert far["pressure_N"] < 0.01 * ap.installation_drag(dict(POD, layout="pusher"), 127.0, 20.0, 5.0)["pressure_N"]


def test_installation_drag_leaves_out_the_motor_end_face():
    for lay in ap.LAYOUTS:
        p = dict(POD, layout=lay, hub_height=10.0, gap=3.0)
        d = ap.installation_drag(p, 127.0, 20.0, 5.0)
        prof = np.array(ap.PropPod.pod_profile(dict({q.name: q.default for q in ap.PropPod.parameters}, **p))) / 1000.0
        x, r = prof[:, 0], prof[:, 1]
        x_face = 0.008 if lay == "tractor" else -0.008
        face = (np.abs(x[:-1] - x_face) < 1e-9) & (np.abs(x[1:] - x_face) < 1e-9)
        assert face.sum() == 1                                               # exactly one flat end face, at the disc side
        R, V, T = 0.127, 20.0, 5.0
        v_i = 0.5 * (-V + math.sqrt(V ** 2 + 2 * T / (1.225 * math.pi * R ** 2)))
        xm = 0.5 * (x[1:] + x[:-1])
        u = v_i * (1 + xm / np.sqrt(xm ** 2 + R ** 2))
        dp = np.where(xm < 0, 0.0, T / (math.pi * R ** 2)) - 1.225 * (V * u + 0.5 * u ** 2)
        ann = np.pi * (r[:-1] ** 2 - r[1:] ** 2)
        assert d["pressure_N"] == pytest.approx(float(np.sum((-dp * ann)[~face])), rel=1e-9)
