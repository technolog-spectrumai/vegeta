"""The 1-D channel model: a short run on a synthetic field profile stays physical."""
import math

import numpy as np
import pytest

pytest.importorskip("fipy")

from vegeta.fulgur.channel import ChannelModel, k_ionisation
from vegeta.fulgur.hall import point_for_power, scale_channel


def test_ionisation_rate_is_sane():
    assert 2e-13 < k_ionisation(20.0) < 8e-13                      # xenon at 20 eV
    assert k_ionisation(5.0) < k_ionisation(20.0) < k_ionisation(40.0)


def test_short_run_stays_physical():
    ch = scale_channel(300.0, 280.0)
    L = ch.L
    b = lambda z: ch.b_exit * math.exp(-((z - 0.9 * L) / (0.45 * L)) ** 2)     # a bell peaking near the exit
    pt = point_for_power(300.0, 280.0)
    mdl = ChannelModel(ch, b, 280.0, pt.mdot_a, n_cells=60)
    s = mdl.run(t_end=4e-6, dt=2e-8)
    p = s["profiles"]
    assert s["I_d"] > 0 and np.isfinite(s["I_d"])
    assert abs(np.sum(p["E"]) * mdl.dz - 280.0) < 1.0                 # the potential drop is the discharge voltage
    assert p["phi"][0] == pytest.approx(280.0, abs=1.0) and abs(p["phi"][-1]) < 1.0
    assert (p["E"][mdl.z >= L] >= 0).all()                           # no reversed field in the plume
    assert p["T_e"].max() <= mdl.te_max and p["T_e"].min() >= mdl.te_min
    assert p["n_n"][0] > p["n_n"][-1]                                # the neutrals are eaten along the channel
    assert p["v"][-1] > 0                                            # ions leave
