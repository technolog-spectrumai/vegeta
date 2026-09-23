from pathlib import Path
import json
import math

import pytest

from vegeta.aeromant import read_checkmesh, read_coefficients, read_solver_log
from vegeta.aeromant.case import read_case_results

DATA = Path(__file__).parent / "data"


def test_coefficients_v1912_format():
    h = read_coefficients(DATA / "coefficient_v1912.dat")
    assert h.columns[:4] == ["Time", "Cm", "Cd", "Cl"]
    assert h["Cd"][-1] == pytest.approx(1.09) and len(h.iterations) == 4


def test_coefficients_v2412_format_aliases_cmpitch():
    h = read_coefficients(DATA / "coefficient_v2412.dat")
    assert "Cm" in h and h["Cm"][-1] == pytest.approx(0.02) and h["Cd"][-1] == pytest.approx(1.2)


def test_restart_files_concatenate_and_deduplicate(tmp_path):
    a = tmp_path / "a.dat"
    b = tmp_path / "b.dat"
    a.write_text("# Time Cm Cd Cl\n1 0 3 0\n2 0 2 0\n")
    b.write_text("# Time Cm Cd Cl\n2 0 1.9 0\n3 0 1.5 0\n")
    h = read_coefficients([a, b])
    assert list(h.iterations) == [1, 2, 3] and h["Cd"][1] == pytest.approx(1.9)


def test_solver_log_and_checkmesh():
    sl = read_solver_log(DATA / "log.solver")
    assert sl.iterations == 2 and sl.converged and sl.completed
    assert sl.residuals["Ux"][-1] == pytest.approx(0.1) and sl.residuals["p"] == pytest.approx([1, 0.3])
    assert "1912" in sl.version
    mc = read_checkmesh(DATA / "log.checkMesh")
    assert mc.cells == 49112 and not mc.ok and mc.failed_checks == 1 and "non-orthogonal" in mc.messages[0]


def test_case_results_from_files(tmp_path):
    case = tmp_path / "case"
    (case / "postProcessing/forceCoeffs/0").mkdir(parents=True)
    (case / "postProcessing/forceCoeffs/0/coefficient.dat").write_text((DATA / "coefficient_v1912.dat").read_text())
    (case / "log.solver").write_text((DATA / "log.solver").read_text())
    params = dict(velocity=2.0, kinematic_viscosity=0.02, density=1.2, reference_area=math.pi / 4, reference_length=1.0)
    (case / "aeromant_case.json").write_text(json.dumps({"config": {"parameters": params}}))
    r = read_case_results(case, average_window=2)
    assert r.ok
    assert r.metrics["Cd"] == pytest.approx(1.09) and r.metrics["Cd_mean_last2"] == pytest.approx(1.095)
    assert r.metrics["drag_force_N"] == pytest.approx(1.09 * 0.5 * 1.2 * 4 * math.pi / 4)
    assert r.metrics["converged"] and r.metrics["reynolds_number"] == pytest.approx(100)


def test_case_results_not_run(tmp_path):
    (tmp_path / "aeromant_case.json").write_text(json.dumps({"config": {"parameters": dict(
        velocity=1.0, kinematic_viscosity=1.0, density=1.0, reference_area=1.0, reference_length=1.0)}}))
    r = read_case_results(tmp_path)
    assert r.status == "failed" and r.metrics["Cd"] is None and "NOT RUN" in r.messages[0]
    assert not read_case_results(tmp_path / "nothing").ok
