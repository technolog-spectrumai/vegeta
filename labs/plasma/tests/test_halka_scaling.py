"""Halka scaled: the calibration points come from the section solves (the 1-D numbers under the mocks), the sweep
and the decision run."""
import pytest

pytest.importorskip("fipy")

from vegeta import mock

import halka_scaling as hs


@pytest.mark.skipif(not mock.installed(), reason="nine 2-D solves: mocks only in the lab's tests")
def test_calibration_sweep_and_decision_under_the_mocks():
    cal = hs.calibrate(lc=4.0)
    assert len(cal.points) == 9 and cal.k_current > 0 and cal.k_onset > 0
    rows = hs.sweep(cal, diameters=(0.2, 0.5), dzs=(0.012, 0.02), stages=(1, 3))
    assert len(rows) == 8 and all(r["thrust"] > 0 and r["mass_total"] > 0 for r in rows)
    dec = hs.decision(rows, cal)
    assert dec["verdict"] in ("hover demonstrator", "winged demonstrator", "no demonstrator: scale up or change the build")
    assert max(r["T/W"] for r in rows) < 1.0                                               # no hover with the supply aboard
