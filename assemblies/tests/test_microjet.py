"""The microjet's components and workflow: geometry, case builders, the tree, reuse from the saved .vida, the export.
No OpenFOAM needed: the CFD batch is replaced by results of a known speed line. The real FEA run is marked slow."""
import math
import shutil

import numpy as np
import pytest

pytest.importorskip("cadquery")
from vegeta import aeromant                                            # noqa: E402
from vegeta.boreas import microjet as mj                               # noqa: E402

from assemblies import vida                                            # noqa: E402
from assemblies.components import cycle, impeller, turbojet as tj, wheels   # noqa: E402
from assemblies.workflows import microjet                              # noqa: E402

ENGINE = mj.from_catalogue("140 N class")
P = tj.sized(ENGINE)


@pytest.mark.parametrize("part", ["impeller", "turbine", "engine"])
def test_parts_build(part):
    g = tj.Turbojet().generate(**dict(P, part=part))
    assert g.shape.isValid() and len(g.shape.Solids()) == 1


@pytest.mark.parametrize("fidelity,n,iterations", [("smoke", 2, 100), ("quick", 2, 300), ("full", 5, 3000)])
def test_compressor_cases_follow_the_fidelity(tmp_path, fidelity, n, iterations):
    node = impeller.assembly(ENGINE, P, fidelity)
    cs = impeller.cases(node, tmp_path)
    assert len(cs) == n and len({c.workdir for c in cs}) == n
    assert all(c.template.name == "compressor_mrf" and c.parameters["iterations"] == iterations for c in cs)
    assert np.all(np.diff([c.parameters["outlet_pressure"] for c in cs]) > 0)
    again = impeller.cases(impeller.assembly(ENGINE, P, fidelity), tmp_path)          # STLs rewritten: same case keys
    assert [c.key for c in cs] == [c.key for c in again]


def test_wheel_models():
    node = wheels.assembly(ENGINE, P, "quick")
    ms = wheels.models(node, {"turbine": "t.step", "impeller": "i.step"})
    assert list(ms) == list(wheels.CASES)
    assert ms["turbine_hot"].material.name.startswith("Inconel") and ms["impeller_hot"].material.name.startswith("Al")
    assert ms["turbine_overspeed"].loads[0].rpm == pytest.approx(1.15 * ENGINE.rpm_max)
    assert ms["turbine_spin"].mesh_settings.element_size == 2.5
    assert wheels.assembly(ENGINE, P, "full").key != node.key


def _speedline_results(cases):
    """A plausible speed line: flow falls and pressure ratio rises with the back pressure."""
    out = []
    for i, c in enumerate(cases):
        r = aeromant.Result(kind="aeromant.results", metadata={"reused": False})
        r.metrics.update(mass_flow_kg_s=0.33 - 0.03 * i, corrected_mass_flow_kg_s=0.34 - 0.03 * i, pressure_ratio_tt=3.4 + 0.3 * i,
                         efficiency_tt=0.74, efficiency_from_torque=0.72, work_coefficient=0.86, shaft_power_W=48000.0)
        out.append(r)
    return out


@pytest.fixture
def fake_cfd(monkeypatch):
    calls = []

    def run_cases(cases, *, jobs=1, processors=1, run=True, progress=False, cancel=None, timeout=None):
        calls.append(len(cases))
        if not run:
            return [aeromant.Result(kind="aeromant.run").fail("NOT RUN") for _ in cases]
        return _speedline_results(cases)

    monkeypatch.setattr(impeller.aeromant, "run_cases", run_cases)
    monkeypatch.setattr(impeller, "environment", lambda run=True: aeromant.OpenFOAMEnvironment())
    return calls


def test_calibration_from_a_speedline():
    assert cycle.calibrate_from_speedline(ENGINE, "140 N class", None) == (ENGINE, None)
    sl = {"ok": [True, True], "mass_flow_kg_s": [0.30, 0.25], "work_coefficient": [0.86, 0.9], "efficiency_tt": [0.74, 0.7]}
    e, info = cycle.calibrate_from_speedline(ENGINE, "140 N class", sl)
    assert info["eta_c"] == 0.74 and e.eta_c == pytest.approx(0.74)
    assert mj.solve(e, e.rpm_max).thrust == pytest.approx(mj.CATALOGUE["140 N class"]["thrust_N"], rel=0.02)
    assert cycle.calibrate_from_speedline(ENGINE, "140 N class", dict(sl, ok=[False, False]))[1] is None


def test_workflow_without_solvers_then_with_cfd_then_reused(tmp_path, fake_cfd):
    kw = dict(fidelity="smoke", run_fea=False, out=tmp_path / "runs", vida_path=tmp_path / "m.vida",
              export_path=tmp_path / "m.json", progress=False)
    first = microjet.run(run_cfd=False, **kw)
    assert [p for p, _ in first.walk()] == ["", "parts", "compressor", "wheels"]
    assert first.child("compressor").status() == "NOT RUN" and first.child("wheels").status() == "NOT RUN"
    assert first.results["calibration"] is None and "calibration" in first.meta["not_run"][0]
    jet = mj.load(tmp_path / "m.json")                                   # the format notebook 29 reads
    assert isinstance(jet["engine"], mj.Microjet) and jet["map"]["thrust"].shape == (len(cycle.V_MAP), 14)
    assert jet["compressor_cfd"] is None and jet["source"]["fidelity"] == "smoke"
    assert set(jet["wheel_masses_kg"]) == {"impeller_kg", "turbine_kg"}

    second = microjet.run(run_cfd=True, **kw)                            # the parts are reused, the CFD runs
    assert second.child("parts").status() == "reused" and fake_cfd == [2, 2]
    comp = second.child("compressor")
    assert comp.status() == "computed" and comp.results["complete"] and comp.results["speedline"]["ok"] == [True, True]
    assert second.results["calibration"]["eta_c"] == pytest.approx(0.74)
    assert mj.load(tmp_path / "m.json")["compressor_cfd"]["eta_c"] == pytest.approx(0.74)

    third = microjet.run(run_cfd=True, **kw)                             # nothing to solve
    assert third.child("compressor").status() == "reused" and fake_cfd == [2, 2]
    loaded = vida.load(tmp_path / "m.vida")
    assert loaded.child("compressor").results["speedline"]["pressure_ratio_tt"] == pytest.approx([3.4, 3.7])
    assert loaded.child("compressor").files["impeller.stl"].sha256

    redo = microjet.run(run_cfd=True, redo=["compressor"], **kw)         # explicit: compute this node again
    assert redo.child("compressor").status() == "computed" and fake_cfd == [2, 2, 2]
    other = microjet.run(run_cfd=True, **dict(kw, fidelity="quick"))     # other settings: other keys, computed again
    assert other.child("compressor").status() == "computed" and other.child("parts").status() == "reused"


def test_force_starts_over(tmp_path, fake_cfd):
    kw = dict(fidelity="smoke", run_fea=False, out=tmp_path / "runs", vida_path=tmp_path / "m.vida", export=False,
              progress=False)
    microjet.run(**kw)
    (tmp_path / "runs" / "marker").write_text("x")
    again = microjet.run(force=True, **kw)
    assert not (tmp_path / "runs" / "marker").exists() and again.child("compressor").status() == "computed"


@pytest.mark.real_solvers
@pytest.mark.slow
@pytest.mark.requires_ccx
def test_workflow_real_fea_then_read_back(tmp_path):
    if shutil.which("ccx") is None:
        pytest.skip("CalculiX not on PATH")
    kw = dict(fidelity="smoke", run_cfd=False, out=tmp_path / "runs", vida_path=tmp_path / "m.vida", export=False,
              progress=False)
    root = microjet.run(**kw)
    stress = root.child("wheels").results["stress"]
    assert root.child("wheels").results["complete"] and all(v["ok"] for v in stress.values())
    assert stress["turbine_overspeed"]["max_von_mises_MPa"] > stress["turbine_hot"]["max_von_mises_MPa"]
    again = microjet.run(**kw)
    assert again.child("wheels").status() == "reused"
    third = microjet.run(**dict(kw, vida_path=tmp_path / "none.vida"))   # no saved tree: the solved models are read back
    assert third.child("wheels").status() == "computed"
    assert all(math.isclose(third.child("wheels").results["stress"][k]["max_von_mises_MPa"], v["max_von_mises_MPa"])
               for k, v in stress.items())
