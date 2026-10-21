"""The racing quad flies as it did in Vegeta: golden.json holds the numbers Vegeta's smoke suite recorded at c15e84e,
before the racing drones graduated (the same cases, the same tolerances)."""
import pytest


def test_reference_quad_and_mock_lap(golden):
    from races import courses, mock, refquad
    s = refquad.summary()
    golden("robotics.races_refquad", {k: v for k, v in s.items() if isinstance(v, float)})
    st = mock.run_mock(courses.hangar())["stats"]
    golden("robotics.races_mock_hangar", {k: st[k] for k in ("status", "lap time [s]", "energy [Wh]", "collisions")},
           rel=1e-9)


def test_mujoco_second_with_tuned_autopilot(golden):
    pytest.importorskip("mujoco")
    from races import autopilot, courses, sim
    r = sim.run_mujoco(courses.hangar(), control_dt=0.004, duration=1.0, **autopilot.tuned("hangar"))
    golden("robotics.races_mujoco_hangar_1s", {"last": r["history"][-1], "rows": len(r["history"])}, rel=1e-6, abs=1e-9)


def test_lab_run_is_what_the_ingest_reads(tmp_path):
    """races/lab/race.py (the mock, no videos) writes the run directory races.ingest digests."""
    import json
    import subprocess
    import sys
    from pathlib import Path
    from races import ingest
    root = Path(__file__).resolve().parents[2]
    run = tmp_path / "runs" / "hangar" / "20260101-000000_screening_mock"
    p = subprocess.run([sys.executable, str(root / "races" / "lab" / "race.py"), "--chapter", "hangar", "--mock", "--no-video",
                        "--out", str(run), "--quiet"], capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stdout[-1500:] + p.stderr[-1500:]
    assert json.loads((run / "summary.json").read_text())["state"] == "completed"
    assert ingest.ingest(tmp_path / "runs", with_mock=True)["courses"]["hangar"]["run"]["referee"]["status"] == "finished"
