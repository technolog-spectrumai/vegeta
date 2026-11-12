"""Timon changes Pumba's tool in MuJoCo (designs/timon_service_sim.py): both robots in one world, the tools as free
bodies, welds that switch on only within tolerance — the service completes and every hand-over is in tolerance.

Run (slow, ~1.5 min): python tools/lab.py test land -- -m slow -k service_sim
"""
import pytest

import timon_service_sim as tss


@pytest.mark.slow
def test_tool_change_in_physics():
    pytest.importorskip("mujoco")
    sim = tss.ServiceSim(fps=1)
    r = sim.run()
    ev = [e for _, e in r["events"]]
    assert not any(e.startswith("FAILED") for e in ev), ev
    assert ev[-2:] == ["on: latch_saw", "off: grip_saw"]
    d, a = r["errors"]["latch_saw"]
    assert d <= tss.LATCH_TOL[0] and a <= tss.LATCH_TOL[1]
    hd, ha = r["errors"]["rack_grab"]
    assert hd <= tss.GRIP_TOL and ha <= 5.0
    # the saw hangs on Pumba's coupler at the end, the grab sits in Timon's holster
    assert sim.d.eq_active[sim.eid["latch_saw"]] and sim.d.eq_active[sim.eid["rack_grab"]]
