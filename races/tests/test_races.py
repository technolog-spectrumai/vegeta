"""Smoke tests: the scenes, the zones' clearance, the mock laps, the rules, the CLI. No MuJoCo run."""
import xml.etree.ElementTree as ET

import pytest

from races import cli, courses, mock, refquad, replay, rules


@pytest.fixture(scope="module", params=list(courses.COURSES))
def course(request):
    return courses.build(request.param)


def test_reference_quad_budget_and_drive():
    s = refquad.summary()
    assert 0.5 < s["all-up mass [kg]"] < 0.65 and s["motor"].endswith("KV") and "6S" in s["voltage"]
    assert 8 < s["hover thrust / weight"] < 16 and 50 < s["hover power [W]"] < 200


def test_scene_builds_and_parses(course, tmp_path):
    xml = courses.to_mjcf(course)
    root = ET.fromstring(xml)
    names = {g.get("name") for g in root.iter("geom")}
    assert "ground" in names and "plate" in names and any(n and n.startswith("zone_") for n in names)
    cams = {c.get("name") for c in root.iter("camera")}
    assert {"overview", "grandstand", "onboard", "chase"} <= cams


def test_zones_clear_the_scene_and_the_line_is_flyable(course):
    ck = course.clearance_check()
    assert ck["ok"].all(), ck[~ck["ok"]]
    assert course.zones[0].kind == "start" and course.zones[-1].kind == "finish"
    assert any(z.kind == "checkpoint" for z in course.zones)
    assert len(course.start_grid) == 4
    p = mock.schedule(course)
    assert p.v.min() >= 2.0 and p.v.max() <= 28.0 and p.time_estimate() < 120


def test_mock_completes_every_course(course):
    r = mock.run_mock(course)
    st = r["stats"]
    assert st["status"] == "finished", (course.slug, st, r["events"][-5:])
    assert st["collisions"] == 0 and st["missed gates"] == 0
    assert st["lap time [s]"] < 1.2 * st["schedule time [s]"] + 10
    assert 0 < st["energy [Wh]"] < refquad.ENERGY_WH
    assert len(r["gate times"]) == len(course.zones) - 1


def test_rules_reset_count_stop():
    c = courses.farm()
    r = mock.run_mock(c, crash_at=5)
    assert r["stats"]["status"] == "finished" and r["stats"]["resets"] == 1 and r["stats"]["penalty [s]"] == 5.0
    r = mock.run_mock(c, crash_at=5, rules=rules.Rules(collision="count"))
    assert r["stats"]["status"] == "finished" and r["stats"]["collisions"] == 1 and r["stats"]["penalty [s]"] == 0.0
    r = mock.run_mock(c, crash_at=5, rules=rules.Rules(collision="stop"))
    assert r["stats"]["status"] == "dnf" and r["stats"]["reason"] == "collision"


def test_cli_list_build_and_mock_run(tmp_path, capsys):
    assert cli.main(["list"]) == 0
    assert cli.main(["build", "--out", str(tmp_path / "scenes")]) == 0
    assert (tmp_path / "scenes" / "quarry.xml").exists()
    assert cli.main(["run", "--course", "hangar", "--mock", "--out", str(tmp_path / "runs")]) == 0
    assert (tmp_path / "runs" / "hangar_refquad_mock" / "stats.json").exists()
    out = capsys.readouterr().out
    assert "finished" in out


def test_metrics_and_envelope():
    from races import metrics
    c = courses.hangar()
    r = mock.run_mock(c)
    m = metrics.from_history(r["history"], path=r["path"])
    assert 0.3 < m["peak lateral acceleration [g]"] < 4.0 and 0 < m["least turn radius above 3 m/s [m]"] < 50
    assert 0 < m["peak roll / pitch rate (thrust-vector slew) [deg/s]"] < 3000 and m["tracking error to the racing line, RMS [m]"] < 2.0
    e = metrics.envelope()
    assert 8 < e["static thrust / weight"] < 16 and e["least turn radius at 20 m/s, 70° tilt [m]"] > 5
    assert len(metrics.literature()) >= 4


def test_chapter_script_with_the_mock(tmp_path):
    """The workflow chapter (wf R1) under the run contract, on the mock: a completed run with the summary, the
    statistics, the manoeuvre metrics and three videos (mp4 with ffmpeg, else gif)."""
    import json
    import subprocess
    import sys
    from pathlib import Path
    script = Path(__file__).resolve().parents[2] / "workflows" / "races" / "race.py"
    out = tmp_path / "run"
    p = subprocess.run([sys.executable, str(script), "--chapter", "hangar", "--mock", "--quality", "screening", "--out", str(out), "--no-progress", "--quiet", "--ignore-prerequisites"], capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    s = json.loads((out / "summary.json").read_text())
    assert s["state"] == "completed" and s["counts"]["solved"] == 4 and s["mode"] == "mock"
    assert s["stats"]["status"] == "finished" and "peak lateral acceleration [g]" in s["manoeuvre"]
    assert set(s["videos"]) == {"chase", "onboard", "overview"} and all((out / v).exists() for v in s["videos"].values())


def test_ingest_digests_a_run_tree(tmp_path):
    """The ingest over a runs tree with one completed (mock) chapter run: the digest has the lap, the splits, the
    trajectory, the comparison and the other runs."""
    import json
    import subprocess
    import sys
    from pathlib import Path
    from races import ingest
    script = Path(__file__).resolve().parents[2] / "workflows" / "races" / "race.py"
    run = tmp_path / "runs" / "hangar" / "20260101-000000_refquad_screening_mock"
    p = subprocess.run([sys.executable, str(script), "--chapter", "hangar", "--mock", "--no-video", "--out", str(run), "--no-progress", "--quiet", "--ignore-prerequisites"], capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, p.stdout[-1500:] + p.stderr[-1500:]
    (tmp_path / "runs" / "farm" / "20260101-000001_refquad_screening").mkdir(parents=True)
    doc = ingest.ingest(tmp_path / "runs", with_mock=False)
    h = doc["courses"]["hangar"]["run"]
    assert h["referee"]["status"] == "finished" and len(h["splits"]) == 17 and len(h["trajectory"]["t"]) > 10 and h["closest_approach"]["distance_m"] > 0
    assert doc["comparison"]["hangar"]["mujoco"]["lap time [s]"] > 0 and doc["courses"]["farm"]["run"] is None
    out = ingest.write(doc, tmp_path / "ingest_race.json")
    assert json.loads(out.read_text())["schema"] == 1


def test_analysis_and_design_factors():
    from races import analysis, ingest
    c = courses.farm()
    r = mock.run_mock(c)
    R = analysis.lap_report(r["history"], c)
    assert abs(R["regimes"]["time share"].sum() - 1.0) < 1e-6 and len(R["corners"]) > 3 and len(R["straights"]) > 2 and R["power"]["energy [Wh]"] > 0
    df = analysis.design_factors({"farm (mock)": R, "hangar (mock)": analysis.lap_report(mock.run_mock(courses.hangar())["history"], courses.hangar())})
    assert "all courses" in df.columns and df.loc["hover thrust / weight required", "all courses"] >= 2.0
    f = analysis.factors_for_raptors(df)
    assert len(f["reading"]) == 5 and "all_courses" in f and "margins" in f
    assert set(ingest.expected_locations()) >= {"workflow runs (wf R1–R4)", "thin CLI runs (python -m races run)"}
    sc = ingest.scan(roots=())
    assert list(sc.columns)[:3] == ["course", "run", "state"]
