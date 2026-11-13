"""Gyrus's channel: the midline field comes from the magnetostatic solve, the model builds on it, a short run is
physical (the long run is the study, minutes)."""
import numpy as np
import pytest

pytest.importorskip("fipy")

import gyrus_channel as gc


def test_midline_field_has_the_hall_shape():
    z, b, sol = gc.midline_field(lc_fine=0.6e-3)
    L = sol["masks"] and gc.scale_channel(300.0, 280.0).L
    inside = z < L
    assert b[inside][:3].mean() < 0.3 * b[inside].max()               # weak at the anode
    assert z[inside][np.argmax(b[inside])] > 0.6 * L                  # peaking near the exit
    assert b[~inside][-1] < 0.5 * b.max()                             # falling off in the plume


def test_model_runs_briefly():
    mdl = gc.model(n_cells=60)
    s = mdl.run(t_end=3e-6, dt=2e-8)
    assert s["I_d"] > 0 and abs(s["profiles"]["phi"][-1]) < 1.0 and s["profiles"]["v"][-1] > 0
