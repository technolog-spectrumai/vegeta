"""Kruse–Kim against the free-space-optics handbook numbers."""
import math

import pytest

from vegeta import hyperion as hy


def test_kruse_kim():
    assert hy.kruse_q(60e3) == 1.6 and hy.kruse_q(10e3) == 1.3 and hy.kruse_q(3e3) == pytest.approx(0.82) and hy.kruse_q(700) == pytest.approx(0.2) and hy.kruse_q(200) == 0.0
    # 23 km "clear" visibility at 1550 nm: the textbook 0.2 dB/km; at 1070 nm a little more; fog at 300 m: ~57 dB/km whatever the wavelength
    clear = hy.Atmosphere("c", 23e3)
    assert clear.loss_db_per_km(1550e-9) == pytest.approx(0.19, abs=0.05)
    assert clear.loss_db_per_km(1070e-9) > clear.loss_db_per_km(1550e-9)
    fog = hy.ATMOSPHERES["foggy autumn morning"]
    assert fog.loss_db_per_km(1070e-9) == pytest.approx(56.6, abs=1.0) and fog.loss_db_per_km(1070e-9) == pytest.approx(fog.loss_db_per_km(1550e-9))
    t = hy.transmission_table([10, 200], 1070e-9)
    assert t["clear day"]["transmission"][200.0] > 0.98 and t["foggy autumn morning"]["transmission"][200.0] == pytest.approx(math.exp(-3.912 / 300 * 200), rel=1e-9)
    assert t["dusty town"]["transmission"][200.0] < t["wet day"]["transmission"][200.0] < t["clear day"]["transmission"][200.0]
