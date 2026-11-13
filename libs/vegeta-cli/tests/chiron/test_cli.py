"""The chiron command: info, run, metrics, render; exit codes 0 ok / 1 failed run / 2 invalid input.

Uses the toy robot and controller of ``test_experiments.py`` by file path, as a study uses its design files.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")

from vegeta.chiron import Episode  # noqa: E402
from vegeta.chiron import cli  # noqa: E402
from vegeta.chiron import experiments as ex  # noqa: E402

TOY = Path(__file__).resolve().parent / "test_experiments.py"
ROBOT = f"{TOY}:toy_robot"
WALK = f"{TOY}:toy_walk"
RESULT_KEYS = {"kind", "status", "metrics", "artifacts", "messages", "duration_s", "execution", "metadata"}
FAST = ["--control-dt", "0.002", "--settle", "0.2"]


def run(argv, capsys):
    code = cli.main(argv)
    out = capsys.readouterr()
    return code, out.out, out.err


# ----------------------------------------------------------------------------------------------- info
def test_info_text_and_json(capsys):
    code, out, _ = run(["info", ROBOT, "-p", "trunk_mass=2.5"], capsys)
    assert code == 0
    assert "toy  (chiron robot, 3.18 kg)" in out and "trunk 2.5" in out
    assert "FL_knee" in out and "knee" in out and "stall 6 N·m" in out and "8 joint(s)" in out
    assert "standing on flat ground" in out
    code, out, _ = run(["info", ROBOT, "--json"], capsys)
    d = json.loads(out)
    assert code == 0 and d["robot"]["total_mass_kg"] == pytest.approx(2.68)
    assert d["robot"]["n_actuated"] == 8 and len(d["robot"]["feet"]) == 4
    assert d["lab"]["nominal_hip_height_m"] > 0.2


def test_info_through_the_vegeta_command(capsys):
    from vegeta.cli import main

    assert main(["chiron", "info", ROBOT]) == 0 and "toy" in capsys.readouterr().out
    assert main(["chiron", "info", f"{TOY}:nope"]) == 2
    assert capsys.readouterr().err.startswith("vegeta chiron: error")


# ----------------------------------------------------------------------------------------------- run, metrics
def test_run_writes_episode_and_summary_then_metrics(tmp_path, capsys):
    out_dir = tmp_path / "run"
    argv = ["run", ROBOT, "--controller", WALK, "-c", "stride=0.08", "--terrain", "rough:0.003", "-t",
            "correlation_length=0.04", "-t", "start=0.2", "--seed", "3", "--course", "0.4", "--v-target", "0.1",
            "-r", "timeout=1.5", "--push", "trunk:0.3:0.8:0.05", "--feasibility-every", "20", "--out", str(out_dir),
            *FAST]
    code, out, _ = run(argv, capsys)
    assert code == 0, out
    assert "toy / toy walk on rough 0.003, seed 3:" in out and "achieved speed" in out
    ep = Episode.load(out_dir / "episode.npz")
    assert ep.log["terrain"]["kind"] == "rough" and ep.log["terrain"]["rms"] == 0.003
    assert ep.log["terrain"]["level"] == 0.003 and ep.log["terrain"]["seed"] == 3
    assert ep.log["disturbances"][0]["impulse_Ns"] == pytest.approx(0.3)
    s = json.loads((out_dir / "summary.json").read_text())
    assert set(s) == RESULT_KEYS and s["kind"] == "chiron.run" and s["status"] == "success"
    assert s["metrics"]["outcome"] == ep.outcome["reason"] and "cot_mech" in s["metrics"]
    assert s["metrics"]["impulse_Ns"] == pytest.approx(0.3)
    assert s["metadata"]["trial"]["seed"] == 3 and s["metadata"]["trial_id"]
    assert Path(s["artifacts"]["episode"]).name == "episode.npz"
    # the CLI run is the Trial the command line describes, bit for bit
    trial = ex.Trial.from_dict(s["metadata"]["trial"])
    assert np.array_equal(trial.run().log["com"], ep.log["com"])
    # metrics of the saved episode
    code, out, _ = run(["metrics", str(out_dir / "episode.npz"), "--json", "--feasibility-every", "20",
                        "--out", str(tmp_path / "m.json")], capsys)
    m = json.loads(out)
    assert code == 0 and m["reason"] == ep.outcome["reason"] and m["terrain_kind"] == "rough"
    assert m["cot_mech"] == pytest.approx(s["metrics"]["cot_mech"])
    r = json.loads((tmp_path / "m.json").read_text())
    assert set(r) == RESULT_KEYS and r["kind"] == "chiron.metrics"
    code, out, _ = run(["metrics", str(out_dir / "episode.npz"), "--feasibility-every", "0"], capsys)
    assert code == 0 and "chiron.metrics: SUCCESS" in out and "achieved_speed_m_s" in out


def test_run_json_duration_only_and_require_success(tmp_path, capsys):
    argv = ["run", ROBOT, "--duration", "0.3", "--no-metrics", "--json", "-o", str(tmp_path / "a"), *FAST]
    code, out, _ = run(argv, capsys)
    s = json.loads(out)
    assert code == 0 and s["metrics"]["outcome"] == "completed" and "cot_mech" not in s["metrics"]
    assert s["metadata"]["trial"]["controller"] is None                       # holds the standing pose
    argv = ["run", ROBOT, "--controller", WALK, "--course", "0.4", "--v-target", "0.1", "--duration", "0.3",
            "--no-metrics", "--require-success", "-o", str(tmp_path / "b"), *FAST]
    code, out, _ = run(argv, capsys)
    assert code == 1 and "STOPPED" in out


def test_terrain_factory(tmp_path, capsys):
    argv = ["run", ROBOT, "--controller", WALK, "--terrain", "ridges:0.01", "--terrain-factory",
            f"{TOY}:flat_or_ridges", "-t", "spacing=0.2", "--duration", "0.3", "--no-metrics", "--seed", "4",
            "-o", str(tmp_path), *FAST]
    code, out, _ = run(argv, capsys)
    assert code == 0, out
    log = Episode.load(tmp_path / "episode.npz").log
    assert log["terrain"]["kind"] == "long_bumps" and log["terrain"]["spacing"] == 0.2
    assert log["terrain"]["seed"] == 4 and log["terrain"]["level"] == 0.01


def test_failed_run_exit_1_with_summary(tmp_path, capsys):
    argv = ["run", ROBOT, "--controller", f"{TOY}:broken_controller", "--duration", "1", "-o", str(tmp_path),
            *FAST]
    code, out, _ = run(argv, capsys)
    assert code == 1 and "controller diverged" in out
    s = json.loads((tmp_path / "summary.json").read_text())
    assert s["status"] == "failed" and "diverged" in s["messages"][0]


# ----------------------------------------------------------------------------------------------- invalid input
@pytest.mark.parametrize("argv, message", [
    (["run", "nope.py:make", "--duration", "1"], "no such file"),
    (["run", f"{TOY}:no_such", "--duration", "1"], "no attribute"),
    (["run", ROBOT, "-p", "mass=3", "--duration", "1"], "unexpected keyword"),
    (["run", ROBOT, "-p", "mass", "--duration", "1"], "name=value"),
    (["run", ROBOT, "--terrain", "moon", "--duration", "1"], "unknown terrain kind"),
    (["run", ROBOT, "--terrain", "long_bumps:0.02", "--duration", "1"], "needs spacing, width"),
    (["run", ROBOT, "--terrain", "rough:abc", "--duration", "1"], "must be a number"),
    (["run", ROBOT, "--terrain", "steps:0.02", "-t", "spacing=0.3", "-t", "bogus=1", "--duration", "1"],
     "no parameter"),
    (["run", ROBOT, "--terrain", "flat:0.1", "--duration", "1"], "flat terrain"),
    (["run", ROBOT], "--course"),
    (["run", ROBOT, "--course", "1.5"], "--v-target"),
    (["run", ROBOT, "--duration", "1", "-r", "max_tilt_deg=45"], "--rule needs --course"),
    (["run", ROBOT, "--duration", "1", "--push", "trunk:3"], "--push expects"),
    (["run", ROBOT, "--duration", "1", "--push", "nobody:1:0.5:0.05"], "nobody"),
    (["info", f"{TOY}:toy_walk"], "not a chiron.Robot"),
    (["metrics", "missing.npz"], "no such file"),
])
def test_invalid_input_exit_2(argv, message, tmp_path, capsys):
    if argv[0] == "run":
        argv = argv + ["-o", str(tmp_path)]
    code, _, err = run(argv, capsys)
    assert code == 2 and message in err, err


def test_metrics_on_a_file_that_is_not_an_episode(tmp_path, capsys):
    bad = tmp_path / "x.npz"
    np.savez(bad, a=np.zeros(3))
    code, _, err = run(["metrics", str(bad)], capsys)
    assert code == 2 and "not a Chiron episode" in err


def test_parse_helpers():
    assert cli.parse_value("0.2") == 0.2 and cli.parse_value("true") is True and cli.parse_value("None") is None
    assert cli.parse_value("spring") == "spring" and cli.parse_value("(1, 2)") == (1, 2)
    spec = cli.parse_terrain("alt_bumps:0.027", {"spacing": 0.17, "width": 0.085, "track_y": 0.13})
    assert spec == {"kind": "alt_bumps", "spacing": 0.17, "width": 0.085, "track_y": 0.13, "height": 0.027,
                    "level": 0.027}
    assert cli.parse_terrain("cross_slope:10", {})["angle_deg"] == 10.0
    assert cli.parse_terrain("rough:0.01", {"correlation_length": 0.04})["rms"] == 0.01
    with pytest.raises(cli.InvalidInput):
        cli.parse_terrain("rough:0.01", {"correlation_length": 0.04, "rms": 0.02})          # conflicting level
    assert cli.parse_push("segment 2:3:3.0:0.05") == {"body": "segment 2", "impulse": 3.0, "t_start": 3.0,
                                                      "duration": 0.05}
    assert cli.parse_push("trunk:1:0.5:0.05:1,0,0")["direction"] == [1.0, 0.0, 0.0]


def test_module_entry_point():
    r = subprocess.run([sys.executable, "-m", "vegeta.chiron.cli", "--help"], capture_output=True, text=True)
    assert r.returncode == 0 and "info" in r.stdout and "metrics" in r.stdout


@pytest.mark.skipif(shutil.which("xvfb-run") is None, reason="needs xvfb-run for off-screen rendering")
def test_render_video(tmp_path):
    pytest.importorskip("pyvista")
    out_dir = tmp_path / "run"
    code = cli.main(["run", ROBOT, "--controller", WALK, "--duration", "0.4", "--no-metrics", "--log-geoms",
                     "-o", str(out_dir), *FAST])
    assert code == 0
    env = {**os.environ, "PYVISTA_OFF_SCREEN": "true"}
    r = subprocess.run(["xvfb-run", "-a", sys.executable, "-m", "vegeta.chiron.cli", "render",
                        str(out_dir / "episode.npz"), "-o", str(tmp_path / "walk.mp4"), "--size", "320x180",
                        "--fps", "10"], capture_output=True, text=True, env=env, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    assert (tmp_path / "walk.mp4").stat().st_size > 1000
