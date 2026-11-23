"""CFDCase.ensure and run_cases: a solved case is read back, never solved twice; a changed input runs again; a batch
resumes. The solver is replaced by writing its force output (no OpenFOAM needed)."""
import threading

import pytest

from vegeta import aeromant
from vegeta.aeromant import CFDCase, OpenFOAMEnvironment, write_stl_ascii
from _rotor import flat_plate_propeller

COM = OpenFOAMEnvironment(env={"WM_PROJECT_VERSION": "v2412"})
PARAMS = dict(rpm=6000.0, diameter=0.127, kinematic_viscosity=1.5e-5, density=1.2)


@pytest.fixture(scope="module")
def prop_stl(tmp_path_factory):
    return write_stl_ascii(flat_plate_propeller(), tmp_path_factory.mktemp("rotor") / "prop.stl")


def _write_forces(case_dir, fx=-2.0, mx=-0.05, n=60):
    pp = case_dir / "postProcessing" / "forces" / "0"
    pp.mkdir(parents=True, exist_ok=True)
    (pp / "force.dat").write_text("# Time forces\n" + "".join(f"{i}\t({fx} 0.01 -0.02)\t(0 0 0)\t(0 0 0)\n" for i in range(1, n + 1)))
    (pp / "moment.dat").write_text("# Time moments\n" + "".join(f"{i}\t({mx} 0.001 0.0)\t(0 0 0)\t(0 0 0)\n" for i in range(1, n + 1)))


@pytest.fixture
def solver(monkeypatch):
    """Replace CFDCase.run: count the calls, write forces (or fail for the workdirs in ``fail``)."""
    calls, fail, lock = [], set(), threading.Lock()

    def fake_run(self, steps=None, *, progress=False, cancel=None, timeout=None, processors=1):
        with lock:
            calls.append(self.workdir.name)
        assert self.is_prepared, "run() on an unprepared case"
        if self.workdir.name in fail:
            return aeromant.Result(kind="aeromant.run").fail("step solver failed: diverged")
        _write_forces(self.workdir)
        return self.results()

    monkeypatch.setattr(CFDCase, "run", fake_run)
    return calls, fail


def _case(stl, workdir, **kw):
    return CFDCase("rotor_mrf_static", stl, dict(PARAMS, **kw), workdir, geometry_units="m", environment=COM)


def test_not_run_touches_nothing(tmp_path, prop_stl, solver):
    case = _case(prop_stl, tmp_path / "c")
    r = case.ensure(run=False)
    assert not r.ok and "NOT RUN" in r.messages[0] and r.metadata["reused"] is False
    assert not (tmp_path / "c").exists() and solver[0] == []


def test_solved_once_then_read_back(tmp_path, prop_stl, solver):
    calls, _ = solver
    case = _case(prop_stl, tmp_path / "c")
    first = case.ensure()
    assert first.ok and first.metadata["reused"] is False and calls == ["c"]
    again = _case(prop_stl, tmp_path / "c").ensure()               # a new object, same inputs: nothing runs
    assert again.ok and again.metadata["reused"] is True and calls == ["c"]
    assert again.metrics["thrust_N"] == pytest.approx(first.metrics["thrust_N"])
    assert _case(prop_stl, tmp_path / "c").is_solved
    assert _case(prop_stl, tmp_path / "c").ensure(run=False).ok    # read back even when running is not allowed


def test_changed_input_runs_again(tmp_path, prop_stl, solver):
    calls, _ = solver
    a = _case(prop_stl, tmp_path / "c")
    a.ensure()
    b = _case(prop_stl, tmp_path / "c", rpm=7000.0)
    assert b.key != a.key and not b.is_solved
    r = b.ensure()
    assert r.ok and r.metadata["reused"] is False and calls == ["c", "c"]
    assert b.is_solved and not a.is_solved                        # the directory now holds the new case


def test_half_run_case_is_prepared_again(tmp_path, prop_stl, solver):
    calls, _ = solver
    case = _case(prop_stl, tmp_path / "c")
    assert case.prepare().ok                                       # prepared, solver never finished
    (tmp_path / "c" / "leftover.txt").write_text("from the interrupted run")
    r = case.ensure()
    assert r.ok and calls == ["c"] and not (tmp_path / "c" / "leftover.txt").exists()


@pytest.mark.parametrize("jobs", [1, 3])
def test_batch_keeps_order_records_failures_and_resumes(tmp_path, prop_stl, solver, jobs):
    calls, fail = solver
    cases = [_case(prop_stl, tmp_path / f"p{i}", rpm=5000.0 + 500 * i) for i in range(4)]
    fail.add("p2")
    rs = aeromant.run_cases(cases, jobs=jobs)
    assert [r.ok for r in rs] == [True, True, False, True]
    assert sorted(calls) == ["p0", "p1", "p2", "p3"]
    fail.clear()
    calls.clear()
    again = aeromant.run_cases([_case(prop_stl, tmp_path / f"p{i}", rpm=5000.0 + 500 * i) for i in range(4)], jobs=jobs)
    assert all(r.ok for r in again) and calls == ["p2"]           # only the failed one ran again
    assert [r.metadata["reused"] for r in again] == [True, True, False, True]


def test_batch_without_running(tmp_path, prop_stl, solver):
    cases = [_case(prop_stl, tmp_path / f"p{i}", rpm=5000.0 + 500 * i) for i in range(2)]
    cases[0].ensure()
    rs = aeromant.run_cases(cases, run=False)
    assert rs[0].ok and not rs[1].ok and "NOT RUN" in rs[1].messages[0] and solver[0] == ["p0"]


def test_batch_refuses_shared_workdir_and_honours_cancel(tmp_path, prop_stl, solver):
    with pytest.raises(ValueError, match="share a workdir"):
        aeromant.run_cases([_case(prop_stl, tmp_path / "x"), _case(prop_stl, tmp_path / "x", rpm=7000.0)])
    stop = threading.Event()
    stop.set()
    rs = aeromant.run_cases([_case(prop_stl, tmp_path / "y")], cancel=stop)
    assert rs[0].status == "cancelled" and solver[0] == []


FAKE_FOAM = '''import sys, pathlib, time
# stands in for every OpenFOAM command: prints a log line, and the solver leaves force output behind
print("fake", " ".join(sys.argv[1:]))
time.sleep(0.05)
if not pathlib.Path("system/controlDict").is_file():     # version probes run outside a case: leave no files
    sys.exit(0)
pp = pathlib.Path("postProcessing/forces/0"); pp.mkdir(parents=True, exist_ok=True)
(pp / "force.dat").write_text("# Time forces\\n" + "".join(f"{i}\\t(-2.0 0.01 -0.02)\\t(0 0 0)\\t(0 0 0)\\n" for i in range(1, 61)))
(pp / "moment.dat").write_text("# Time moments\\n" + "".join(f"{i}\\t(-0.05 0.001 0.0)\\t(0 0 0)\\t(0 0 0)\\n" for i in range(1, 61)))
'''


def test_batch_runs_real_pipelines_side_by_side(tmp_path, prop_stl):
    """The real CFDCase.run (commands, logs, summary.json) in three threads, OpenFOAM replaced by a script."""
    import sys

    script = tmp_path / "fake_foam.py"
    script.write_text(FAKE_FOAM)
    env = OpenFOAMEnvironment(prefix=[sys.executable, str(script)], env={"WM_PROJECT_VERSION": "v2412"})
    cases = [CFDCase("rotor_mrf_static", prop_stl, dict(PARAMS, rpm=5000.0 + 500 * i), tmp_path / f"p{i}",
                     geometry_units="m", environment=env) for i in range(3)]
    rs = aeromant.run_cases(cases, jobs=3)
    assert all(r.ok for r in rs), [r.messages for r in rs]
    assert all((tmp_path / f"p{i}" / "summary.json").is_file() and (tmp_path / f"p{i}" / "log.solver").is_file() for i in range(3))
    assert [r.metadata["reused"] for r in aeromant.run_cases(cases, jobs=3)] == [True, True, True]
