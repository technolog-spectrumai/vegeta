"""The Kestrel campaigns and what the notebook may do with their runs: compatibility, missing prerequisites,
imported legacy results, prepare-only smoke, and the guarantee that the notebook launches no solver."""
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

import runkit
import kestrel_campaigns as kcp

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable


@pytest.fixture(scope="module")
def fp2():
    return kcp.fingerprint("4Q-2")


def test_campaign_catalogue_sizes():
    assert len(kcp.cfd_specs("4Q-2", "sweep")) == 10 and len(kcp.cfd_specs("4Q-2", "stall")) == 5 and len(kcp.cfd_specs("4Q-2", "sweep-fine")) == 13
    cruise = kcp.cfd_specs("4Q-2", "cruise")
    assert len(cruise) == 2 and all(s["props"] for s in cruise.values())
    with pytest.raises(ValueError):
        kcp.cfd_specs("4Q-2", "everything")
    assert runkit.select_cases(list(kcp.cfd_specs("4Q-1", "sweep")), "α 10°,α 15°") == ["bare airframe, α 10°, 10 m/s", "bare airframe, α 15°, 10 m/s"]


def test_fingerprint_compare(fp2):
    assert kcp.compare(fp2, fp2)["level"] == "compatible"
    other = dict(fp2, x_cg_m=fp2["x_cg_m"] + 0.005)
    c = kcp.compare(fp2, other)
    assert c["level"] == "incompatible" and any("CG differs" in r for r in c["reasons"])
    c = kcp.compare(fp2, kcp.fingerprint("4Q-1"))
    assert c["level"] == "incompatible" and any("CAD parameters differ" in r for r in c["reasons"])
    cal = dict(fp2, prop_lab={**fp2["prop_lab"], "cl_scale": 0.9})
    assert kcp.compare(fp2, cal, blown=False)["level"] == "warnings" and kcp.compare(fp2, cal, blown=True)["level"] == "incompatible"
    assert kcp.compare(fp2, None)["level"] == "incompatible"


def _fake_run(root, fp, *, state="completed", summary=True, workflow="airframe_cfd", design=None):
    root.mkdir(parents=True)
    runkit.write_json_atomic(root / "manifest.json", {"schema": 1, "run_id": root.name, "workflow": workflow, "campaign": "sweep", "created_at": "2026-01-01T00:00:00Z",
                                                      "config": {"variant": "4Q-2", "quality": "screening"}, "config_hash": "x", "design": fp if design is None else design, "tool_versions": {}, "artifacts": {}})
    runkit.write_json_atomic(root / "status.json", {"state": state, "cases": {"planned": 1, "done": 1 if state == "completed" else 0}})
    if summary:
        runkit.write_json_atomic(root / "summary.json", {"state": "completed", "counts": {"solved": 1, "n": 1, "converged": 1}, "records": [], "cases": {}})


def test_check_run_missing_unfinished_incompatible_and_usable(tmp_path, fp2):
    ck = kcp.check_run(None, workflow="airframe_cfd", variant="4Q-2", campaign="sweep", expected=fp2)
    assert not ck["ok"] and ck["status"] == "not selected" and "airframe_cfd.py --variant 4Q-2 --campaign sweep" in ck["command"]
    ck = kcp.check_run(tmp_path / "nowhere", workflow="airframe_cfd", variant="4Q-2", campaign="sweep", expected=fp2)
    assert not ck["ok"] and ck["status"].startswith("not found")
    _fake_run(tmp_path / "running", fp2, state="running", summary=False)
    ck = kcp.check_run(tmp_path / "running", workflow="airframe_cfd", variant="4Q-2", campaign="sweep", expected=fp2)
    assert not ck["ok"] and "running" in ck["status"] and "--resume" in ck["status"]
    _fake_run(tmp_path / "other", fp2, design=kcp.fingerprint("4Q-1"))
    ck = kcp.check_run(tmp_path / "other", workflow="airframe_cfd", variant="4Q-2", campaign="sweep", expected=fp2)
    assert not ck["ok"] and ck["status"].startswith("incompatible")           # the file exists and is complete: still not usable
    _fake_run(tmp_path / "good", fp2)
    ck = kcp.check_run(tmp_path / "good", workflow="airframe_cfd", variant="4Q-2", campaign="sweep", expected=fp2)
    assert ck["ok"] and ck["compat"] == "compatible" and ck["summary"]["counts"]["solved"] == 1
    with pytest.raises(kcp.PrerequisiteError, match="structure_fea.py --variant 4Q-2 --campaign all"):
        kcp.require(kcp.check_run(None, workflow="structure_fea", variant="4Q-2", campaign="all", expected=fp2), what="FEA")
    assert kcp.require(kcp.check_run(None, workflow="structure_fea", variant="4Q-2", campaign="all", expected=fp2), what="FEA", required=False) is None
    ck = kcp.check_run(tmp_path / "good", workflow="structure_fea", variant="4Q-2", campaign="all", expected=fp2)
    assert not ck["ok"] and "is a airframe_cfd run, not structure_fea" in ck["status"]      # a CFD run never passes as an FEA one


def test_notebook_prerequisites_raise_one_actionable_error_and_write_nothing(tmp_path, fp2, monkeypatch):
    import kestrel_study as kstu
    runs = {"airframe_cfd": {"4Q-2": {"sweep": str(tmp_path / "missing")}}, "structure_fea": {"4Q-2": None}}
    fps = {"4Q-2": fp2}
    before = sorted(p.name for p in tmp_path.iterdir())
    with pytest.raises(kcp.PrerequisiteError) as exc:
        kstu.prerequisites(runs, {("airframe_cfd", "4Q-2", "sweep"), ("structure_fea", "4Q-2")}, ("4Q-2",), fingerprints=fps)
    msg = str(exc.value)
    assert "airframe_cfd.py --variant 4Q-2 --campaign sweep" in msg and "structure_fea.py --variant 4Q-2 --campaign all" in msg and "KESTREL_SCREENING" in msg
    assert sorted(p.name for p in tmp_path.iterdir()) == before          # no run directory, no solver, nothing invented
    table, summ = kstu.prerequisites(runs, set(), ("4Q-2",), fingerprints=fps)
    assert list(table.loc[table["role"] == "optional"].index) and not summ["airframe_cfd"] and not summ["structure_fea"]


def test_import_legacy_makes_completed_runs_that_say_they_are_assumed(tmp_path, fp2):
    roots = kcp.import_legacy(kcp.kcfd.RESULTS_JSON, tmp_path, keys=("4Q-2",))
    assert len(roots) == 1
    s = json.loads((roots[0] / "summary.json").read_text())
    assert s["state"] == "completed" and s["counts"]["n"] == 29 and s["counts"]["solved"] == 29 and "bare_sweep" in s["maps"]
    assert s["maps"]["bare_sweep"]["valid_domain"]["alpha_deg"] == [0.0, 90.0] and (roots[0] / "bare_sweep.csv").exists()
    assert all(r["converged"] in (True, False, None) for r in s["records"])
    ck = kcp.check_run(roots[0], workflow="airframe_cfd", variant="4Q-2", campaign="sweep", expected=fp2)
    assert ck["ok"] and ck["compat"] == "warnings" and any("assumed" in r for r in ck["reasons"]) and "imported" in ck["provenance"]
    # the notebook's reading of it: an unconverged screening sweep is indicative, never validated
    import kestrel_study as kstu
    c = kstu.cfd_sweep(s, kstu.build("4Q-2"))
    assert c["label"].startswith("solver (unconverged") and c["n_bare"] == 13 and c["numbers"]["CFD"]["CL max below 20°"] < 0.6
    corr = kcp.airframe_correction(s, kcp.vehicle("4Q-2"), source="sweep")
    assert corr is not None and corr.source == "sweep"
    assert kcp.airframe_correction(s, kcp.vehicle("4Q-2"), source="sweep", converged_only=True) is not None or s["counts"]["converged"] == 0


SOLVER_CALLS = re.compile(r"\.run\(|run_cases\(|\.ensure\(|\.solve\(|solve_modes\(|\.mesh\(|subprocess|os\.system|CFDCase\(")


def test_the_notebook_and_its_modules_launch_no_solver():
    nb = json.loads((ROOT / "notebooks" / "43_kestrel_study.ipynb").read_text())
    src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert not SOLVER_CALLS.search(src), SOLVER_CALLS.search(src)
    for mod in ("kestrel_study.py", "kestrel_campaigns.py"):
        text = (ROOT / "notebooks" / "designs" / mod).read_text()
        hits = [m.group(0) for m in SOLVER_CALLS.finditer(text)]
        assert not hits, (mod, hits)
    assert "import_legacy" in src or "RUNS" in src                              # the selection is explicit in the notebook


def test_scripts_list_without_writing(tmp_path):
    out = subprocess.run([PY, str(ROOT / "workflows" / "kestrel" / "airframe_cfd.py"), "--variant", "4Q-1", "--campaign", "stall", "--list"], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0 and "5 case(s)" in out.stdout and "rans_ksst_external" in out.stdout
    out = subprocess.run([PY, str(ROOT / "workflows" / "kestrel" / "structure_fea.py"), "--variant", "4Q-2", "--campaign", "frame", "--list"], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0 and "frame_posts_A" in out.stdout and "frame_posts_modes" in out.stdout


@pytest.mark.slow
def test_airframe_cfd_prepare_only_smoke(tmp_path):
    """One bare case prepared (the STL exported, the OpenFOAM dictionaries written), no solver: the run says 'prepared'."""
    out = subprocess.run([PY, str(ROOT / "workflows" / "kestrel" / "airframe_cfd.py"), "--variant", "4Q-2", "--campaign", "stall", "--cases", "α 12°", "--prepare-only",
                          "--out", str(tmp_path / "run"), "--quiet"], capture_output=True, text=True, timeout=900)
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    st = json.loads((tmp_path / "run" / "status.json").read_text())
    assert st["state"] == "prepared" and not (tmp_path / "run" / "summary.json").exists()
    p = json.loads((tmp_path / "run" / "summary.partial.json").read_text())
    assert list(p["cases"]) == ["bare airframe, α 12°, 10 m/s"] and p["cases"]["bare airframe, α 12°, 10 m/s"]["status"] == "NOT RUN"
    case_dirs = [d for d in (tmp_path / "run" / "cases").iterdir() if d.is_dir() and d.name != "stl"]
    assert len(case_dirs) == 1 and (case_dirs[0] / "system" / "controlDict").exists()
    m = json.loads((tmp_path / "run" / "manifest.json").read_text())
    assert m["design"]["variant"] == "4Q-2" and m["config"]["quality"] == "screening"
    # resume with the same configuration opens; another refuses
    out = subprocess.run([PY, str(ROOT / "workflows" / "kestrel" / "airframe_cfd.py"), "--resume", str(tmp_path / "run"), "--prepare-only", "--cases", "α 12°", "--quiet"],
                         capture_output=True, text=True, timeout=900)
    assert out.returncode == 0 and len(json.loads((tmp_path / "run" / "manifest.json").read_text())["resumed"]) == 1
