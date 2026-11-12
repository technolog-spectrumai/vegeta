import json
import subprocess
import sys

import pytest

from vegeta.fidia import cli

pytestmark = pytest.mark.slow


def test_offline_run_resume_show(tmp_path, capsys):
    out = tmp_path / "stool"
    assert cli.main(["run", "a small stool with three legs", "--offline", "--out", str(out), "--quiet"]) == 0
    assert (out / "best" / "export" / "model.glb").is_file()
    assert (out / "best" / "export" / "model.obj").is_file() and (out / "best" / "export" / "model.mtl").is_file()
    printed = capsys.readouterr().out
    assert "best: rev-002" in printed and "model.glb" in printed
    assert cli.main(["resume", str(out), "--offline", "--quiet", "--feedback", "make the seat red"]) == 0
    assert json.loads((out / "run.json").read_text())["best"] == "rev-003"
    capsys.readouterr()
    assert cli.main(["show", str(out)]) == 0
    shown = capsys.readouterr().out
    assert "rev-001" in shown and "floating" in shown and "rev-003" in shown


def test_live_run_needs_a_key(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    assert cli.main(["run", "a cup", "--out", str(tmp_path / "x")]) == 2
    assert "no API key" in capsys.readouterr().err


def test_plugin_help():
    r = subprocess.run([sys.executable, "-m", "vegeta.cli", "fidia", "--help"], capture_output=True, text=True)
    if r.returncode != 0:  # the umbrella CLI may not be runnable as a module; use the console script
        r = subprocess.run(["vegeta", "fidia", "--help"], capture_output=True, text=True)
    assert r.returncode == 0 and "run" in r.stdout and "resume" in r.stdout and "propose" in r.stdout
