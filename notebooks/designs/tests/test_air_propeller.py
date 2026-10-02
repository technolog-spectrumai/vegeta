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
