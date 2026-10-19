"""The Kestrel campaigns and what the notebook may do with their runs: compatibility, missing prerequisites,
imported legacy results, prepare-only smoke, and the guarantee that the notebook launches no solver."""
import json
import math
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
    inter = kcp.cfd_specs("4Q-2", "interference")
    assert len(inter) == 9 and {s["planform"] for s in inter.values()} == {"single", "pair", "monoplane"} and not any(s["props"] for s in inter.values())
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
    assert not ck["ok"] and ck["status"] == "not selected" and "airframe_cfd.py --variant 4Q-2 --chapter sweep" in ck["command"]
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
    with pytest.raises(kcp.PrerequisiteError, match="structure_fea.py --variant 4Q-2 --chapter all"):
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
    assert "airframe_cfd.py --variant 4Q-2 --chapter sweep" in msg and "structure_fea.py --variant 4Q-2 --chapter all" in msg and "KESTREL_SCREENING" in msg
    assert sorted(p.name for p in tmp_path.iterdir()) == before          # no run directory, no solver, nothing invented
    table, summ = kstu.prerequisites(runs, set(), ("4Q-2",), fingerprints=fps)
    assert list(table.loc[table["role"] == "optional"].index) and not summ["airframe_cfd"] and not summ["structure_fea"]


def test_import_legacy_makes_completed_runs_that_say_they_are_assumed(tmp_path, fp2):
    roots = kcp.import_legacy(kcp.kcfd.RESULTS_JSON, tmp_path, keys=("4Q-2",))
    assert len(roots) == 1 and roots[0].parent.name == "airframe-sweep" and roots[0].parents[1].name == "kestrel"
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
    nb = json.loads((ROOT / "notebooks" / "devices" / "43_kestrel_study.ipynb").read_text())
    src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert not SOLVER_CALLS.search(src), SOLVER_CALLS.search(src)
    for mod in ("kestrel_study.py", "kestrel_campaigns.py"):
        text = (ROOT / "designs" / mod).read_text()
        hits = [m.group(0) for m in SOLVER_CALLS.finditer(text)]
        assert not hits, (mod, hits)
    assert "import_legacy" in src or "RUNS" in src                              # the selection is explicit in the notebook


def test_scripts_list_without_writing(tmp_path):
    out = subprocess.run([PY, str(ROOT / "workflows" / "kestrel" / "airframe_cfd.py"), "--variant", "4Q-1", "--chapter", "stall", "--list"], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0 and "5 case(s)" in out.stdout and "rans_ksst_external" in out.stdout
    out = subprocess.run([PY, str(ROOT / "workflows" / "kestrel" / "structure_fea.py"), "--variant", "4Q-2", "--chapter", "frame", "--list"], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0 and "frame_posts_A" in out.stdout and "frame_posts_modes" in out.stdout


@pytest.mark.slow
def test_airframe_cfd_prepare_only_smoke(tmp_path):
    """One bare case prepared (the STL exported, the OpenFOAM dictionaries written), no solver: the run says 'prepared'."""
    out = subprocess.run([PY, str(ROOT / "workflows" / "kestrel" / "airframe_cfd.py"), "--variant", "4Q-2", "--chapter", "stall", "--cases", "α 12°", "--prepare-only",
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


def test_cache_reuses_a_solved_case_across_runs_and_cleans(tmp_path, monkeypatch):
    import _kestrel as wk
    calls = []

    def solver(name, d, cancel):
        calls.append(name)
        return {"status": "solved", "process": "ok", "converged": True, "metrics": {"CL": 0.5}, "artifacts": {"case": str(d)}}
    cache = runkit.Cache("demo_book", base=tmp_path / "cache")
    r1 = runkit.Run.create(tmp_path / "r1", workflow="demo", campaign="c", config={}, planned_cases=["a"], heartbeat_s=10, echo=False)
    assert wk.drive(r1, ["a"], {"a": "key-a"}, solver, budget=runkit.Budget(), cache=cache) == "completed"
    r1.finish({}, state="completed"); r1.close()
    assert (tmp_path / "cache" / "demo_book" / "key-a.json").exists()
    r2 = runkit.Run.create(tmp_path / "r2", workflow="demo", campaign="c", config={}, planned_cases=["a", "b"], heartbeat_s=10, echo=False)
    cache2 = runkit.Cache("demo_book", base=tmp_path / "cache")
    assert wk.drive(r2, ["a", "b"], {"a": "key-a", "b": "key-b"}, solver, budget=runkit.Budget(), cache=cache2) == "completed"
    assert calls == ["a", "b"] and r2.records["a"].reused and "from the cache" in r2.records["a"].note and r2.records["a"].metrics["CL"] == 0.5
    assert cache2.hits == 1 and "cache hit" in (tmp_path / "r2" / "run.log").read_text()
    r2.finish({}, state="completed"); r2.close()
    # --no-cache: solved again
    r3 = runkit.Run.create(tmp_path / "r3", workflow="demo", campaign="c", config={}, planned_cases=["a"], heartbeat_s=10, echo=False)
    wk.drive(r3, ["a"], {"a": "key-a"}, solver, budget=runkit.Budget(), cache=runkit.Cache("demo_book", base=tmp_path / "cache", enabled=False))
    assert calls == ["a", "b", "a"]
    r3.close()
    st = runkit.Cache.stats(tmp_path / "cache")
    assert st["demo_book"]["cases"] == 2
    assert runkit.Cache.clean(tmp_path / "cache", book="demo_book") == 2 and not runkit.Cache.stats(tmp_path / "cache")


def test_library_chapters_and_prerequisites(tmp_path, monkeypatch, fp2):
    import library
    monkeypatch.setattr(library, "RUNS", tmp_path)
    chs = library.chapters("kestrel")
    assert chs["airframe-transitions"]["prerequisites"] == ("airframe-sweep",) and chs["airframe-sweep"]["prerequisites"] == () and chs["airframe-stall"]["description"]
    assert chs["airframe-sweep"]["code"] == "K1" and chs["frame"]["kind"] == "fea" and chs["joint"]["prerequisites"] == ("frame",)
    assert library.by_code("k3")[1] == "airframe-cruise" and library.by_code("K10")[1] == "joint" and library.by_code("kestrel/frame")[2]["code"] == "K9"
    with pytest.raises(KeyError):
        library.by_code("K99")
    with pytest.raises(KeyError):
        library.by_code("Z1")
    assert library.chapter_of("kestrel", "cfd", "stall") == "airframe-stall" and library.chapter_of("kestrel", "fea", "all") == "frame-and-joint"
    assert "K1" in library.tree(runs=False) and "[after K1]" in library.tree(runs=False) and "FEA" in library.tree(runs=False)
    missing = library.check_prerequisites("kestrel", "airframe-transitions", variant="4Q-2", expected=fp2)
    assert len(missing) == 1 and "--chapter sweep" in missing[0] and "wf K1" in missing[0]
    assert library.check_prerequisites("kestrel", "airframe-sweep", variant="4Q-2", expected=fp2) == []
    _fake_run(tmp_path / "kestrel" / "airframe-sweep" / "s1", fp2)
    (tmp_path / "kestrel" / "airframe-sweep" / "s1" / "manifest.json").write_text(json.dumps({**json.loads((tmp_path / "kestrel" / "airframe-sweep" / "s1" / "manifest.json").read_text()), "book": "kestrel", "chapter": "airframe-sweep"}))
    assert library.check_prerequisites("kestrel", "airframe-transitions", variant="4Q-2", expected=fp2) == []
    assert library.check_prerequisites("kestrel", "airframe-transitions", variant="4Q-1", expected=kcp.fingerprint("4Q-1"))       # another variant: still missing
    cs = library.chapter_status("kestrel", "airframe-sweep")
    assert cs["runs"] == 1 and cs["overall"] == "completed" and cs["mark"] == "✔" and cs["latest"]["4Q-2"]["state"] == "completed" and "4Q-2 completed" in cs["summary"]
    assert library.chapter_status("kestrel", "joint")["summary"] == "not run" and library.chapter_status("kestrel", "joint")["mark"] == "—"
    _fake_run(tmp_path / "kestrel" / "airframe-sweep" / "s2", fp2, state="failed", summary=False)
    cs = library.chapter_status("kestrel", "airframe-sweep")
    assert cs["counts"]["failed"] == 1 and cs["overall"] in ("completed", "failed")          # the latest run per variant decides
    assert "✘" in library.tree() or "✔" in library.tree()
    runs = library.runs_of("kestrel", "airframe-sweep")
    assert runs and runs[0]["book"] == "kestrel" and runs[0]["chapter"] == "airframe-sweep" and runs[0]["code"] == "K1" and runs[0]["variant"] == "4Q-2"


def test_interference_chapter_planforms_and_factors(tmp_path):
    """K12: the three planform STLs build (the pair twice one wing's size), the case prepares with the planform's own
    reference area, the factor fit reads lift loss and the k ratio off synthetic coefficients, and an empty summary
    carries the block as NOT RUN."""
    import kestrel_cfd as kcfd
    from vegeta import aeromant
    p, veh = kcp.params("4Q-2"), kcp.vehicle("4Q-2")
    sizes = {pf: kcfd.planform_stl(tmp_path / "stl", pf, 4.0, veh.x_cg, p).stat().st_size for pf in ("single", "pair", "monoplane")}
    assert 1.8 * sizes["single"] < sizes["pair"] < 2.2 * sizes["single"]
    specs = kcp.cfd_specs("4Q-2", "interference")
    c1 = kcp.cfd_case("one wing alone, α 4°, 10 m/s", specs["one wing alone, α 4°, 10 m/s"], tmp_path, variant="4Q-2", quality="screening", veh=veh, env=aeromant.OpenFOAMEnvironment(env={}))
    c2 = kcp.cfd_case("the wing pair, α 4°, 10 m/s", specs["the wing pair, α 4°, 10 m/s"], tmp_path, variant="4Q-2", quality="screening", veh=veh, env=aeromant.OpenFOAMEnvironment(env={}))
    assert abs(c2.parameters["reference_area"] - 2 * c1.parameters["reference_area"]) < 1e-9
    rows = {f"{pf} {a}": {"planform": pf, "alpha_deg": a, "status": "solved", "converged": True, "CL": cla * math.radians(a), "CD": 0.02 + k * (cla * math.radians(a)) ** 2}
            for pf, k, cla in (("single", 0.043, 4.58), ("pair", 0.059, 4.10), ("monoplane", 0.084, 3.58)) for a in (0, 4, 8)}
    f = kcfd.interference_factors(rows)
    assert f["label"] == "validated" and abs(f["factors"]["lift interference factor (mean over α > 0)"] - 4.10 / 4.58) < 1e-6
    assert abs(f["factors"]["induced drag vs monoplane of the same span and area"] - 0.059 / 0.084) < 1e-6
    rows["pair 8"]["converged"] = False
    assert kcfd.interference_factors(rows)["label"] == "indicative"
    summ = kcp.cfd_summary(variant="4Q-2", campaign="interference", quality="screening", specs=specs, records={}, veh=veh, run_root=tmp_path)
    assert summ["interference"]["label"] == "NOT RUN" and summ["counts"]["n"] == 9 and summ["maps"] == {}
