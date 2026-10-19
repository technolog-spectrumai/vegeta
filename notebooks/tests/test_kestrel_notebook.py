"""Notebook 43's side of the Kestrel exchange as it stands before the Porunga keys (todo S.11): its contract in
notebooks/flow.py, its prerequisite errors, and the guarantee that it and its modules launch no solver."""
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "designs"))
import kestrel_campaigns as kcp  # noqa: E402
import kestrel_study as kstu  # noqa: E402
from vegeta import yanus  # noqa: E402

FLOW = yanus.load_flow(ROOT / "notebooks" / "flow.py")
SOLVER_CALLS = re.compile(r"\.run\(|run_cases\(|\.ensure\(|\.solve\(|solve_modes\(|\.mesh\(|subprocess|os\.system|CFDCase\(")


@pytest.fixture(scope="module")
def fp2():
    return kcp.fingerprint("4Q-2")


def test_notebook_43_contract_matches_its_runs_shape():
    g = yanus.enter("43_kestrel_study", flow=FLOW, deep=False, raise_closed=False, screening=False, auto=False)   # the manual shape (43 is yanus_auto)
    runs, required = kstu.runs_from_gate(g)
    assert set(runs["airframe_cfd"]) == {"4Q-1", "4Q-2", "4Q-3"} and "4Q-2" in runs["structure_fea"]
    assert required == {("airframe_cfd", "4Q-2", "sweep"), ("structure_fea", "4Q-2")}
    g = yanus.enter("43_kestrel_study", flow=FLOW, deep=False, raise_closed=False, screening=True, auto=False)
    assert kstu.runs_from_gate(g)[1] == set()


def test_notebook_43_is_yanus_auto_and_its_plan_runs_screening_quality():
    nb = FLOW.notebooks["43_kestrel_study"]
    assert nb.auto and FLOW.notebooks["45_kestrel_x"].auto
    first = yanus.enter(nb, flow=FLOW, deep=False, raise_closed=False, auto=False)
    steps = yanus.auto.plan(FLOW, first)
    cfd = [s for s in steps if s.unit.kind and s.unit.kind.name == "cfd"]
    fea = [s for s in steps if s.unit.kind and s.unit.kind.name == "fea"]
    assert all(s.argv[-2:] == ["--quality", "screening"] for s in cfd)
    assert all("--element-size" in s.argv for s in fea)
    assert all(s.state in ("planned", "skipped") for s in steps)        # a plan starts nothing


def test_notebook_prerequisites_raise_one_actionable_error_and_write_nothing(tmp_path, fp2, monkeypatch):
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


def test_the_notebook_and_its_modules_launch_no_solver():
    nb = json.loads((ROOT / "notebooks" / "devices" / "43_kestrel_study.ipynb").read_text())
    src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert not SOLVER_CALLS.search(src), SOLVER_CALLS.search(src)
    for mod in ("kestrel_study.py", "kestrel_campaigns.py"):
        text = (ROOT / "designs" / mod).read_text()
        hits = [m.group(0) for m in SOLVER_CALLS.finditer(text)]
        assert not hits, (mod, hits)
    assert "import_legacy" in src or "RUNS" in src                              # the selection is explicit in the notebook


def test_the_notebook_reads_an_imported_sweep_as_indicative(tmp_path):
    """An unconverged screening sweep (the legacy import) is indicative, never validated."""
    roots = kcp.import_legacy(kcp.kcfd.RESULTS_JSON, tmp_path, keys=("4Q-2",))
    s = json.loads((roots[0] / "summary.json").read_text())
    c = kstu.cfd_sweep(s, kstu.build("4Q-2"))
    assert c["label"].startswith("solver (unconverged") and c["n_bare"] == 13 and c["numbers"]["CFD"]["CL max below 20°"] < 0.6
