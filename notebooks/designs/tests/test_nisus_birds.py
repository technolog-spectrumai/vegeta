"""NISUS bird photography (notebook 32): the seam between the 6-DOF aircraft and the mission software, the
autopilot's mode 'birds' and its failsafe, a short hunt, and (slow) a whole mission.

Run: cd /home/user/vegeta && .venv/bin/python -m pytest -q notebooks/designs/tests/test_nisus_birds.py -m "not slow"
"""
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("mujoco")

import nisus_birds as nb  # noqa: E402


def test_scenarios_and_configs():
    ss = nb.bird_scenarios()
    assert len(ss) == 6 and len({s.slug for s in ss}) == 6
    cfg = ss[0].config()
    assert cfg.camera.name.startswith("IMX219") and cfg.detector.device == "orin_nano_super"
    c = ss[0].centre()
    assert np.allclose(cfg.guidance.search_centre, c) and np.allclose(cfg.scheduler.area_centre, c)
    birds = ss[0].make_birds()
    assert birds.n == 6 and np.all(np.linalg.norm(birds.pos[:, :2] - c, axis=1) < 300)


def test_short_hunt_in_mujoco():
    """Launch, climb, outbound and the first seconds of the hunt: the mission software starts in the survey phase,
    detects and tracks, and the autopilot flies its guidance (no failsafe)."""
    bs = replace(nb.BirdScenario(), bird_time_s=60.0)
    ep = nb.run(bs, duration=70.0)
    phases = [m[1] for m in ep.log["mission"]]
    assert phases[:5] == ["prelaunch", "launch", "climb", "outbound", "survey"]
    loop = ep.birds.loop
    assert loop.active and ep.birds.t_start is not None
    assert loop.detector.frames > 50 and loop.stack.tracker.stats["updates"] > 0
    assert not any(e[1] == "failsafe" for e in ep.log["events"])
    assert ep.birds.command() is not None and ep.birds.command().mode in ("search", "intercept", "pass", "breakoff", "reposition")


def test_companion_silence_triggers_the_return():
    """The Jetson fails during the hunt: no guidance → hold, then return after the timeout, as with the waypoints."""
    bs = replace(nb.BirdScenario(), jetson_failure=(40.0, 1000.0), bird_time_s=300.0)
    ep = nb.run(bs, duration=60.0)
    ev = [e for e in ep.log["events"] if e[1] == "failsafe"]
    assert ev and 44.0 < ev[0][0] < 47.0 and ep.log["return_reason"] == "companion lost"
    assert not ep.birds.loop.active


@pytest.mark.slow
def test_whole_bird_mission_lands_with_photos():
    ep = nb.run(nb.BirdScenario())
    s = nb.summary(ep)
    assert s["landed"] and s["return"] == "mission complete"
    assert s["photos"] > 0 and s["closest approach [m]"] > 5.0
