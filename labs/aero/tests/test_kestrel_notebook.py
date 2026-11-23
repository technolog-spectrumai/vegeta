"""Notebook 43's side of the Kestrel exchange: it finds the side labs' results in the Porunga cache by key, never by
run directory; mock and real entries never mix; a missing required result stops with the command; nothing is launched."""
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "labs" / "aero" / "designs"))
import kestrel_campaigns as kcp  # noqa: E402
import kestrel_keys as kk  # noqa: E402
import kestrel_study as kstu  # noqa: E402
import porunga  # noqa: E402

SOLVER_CALLS = re.compile(r"\.run\(|run_cases\(|\.ensure\(|\.solve\(|solve_modes\(|\.mesh\(|subprocess|os\.system|CFDCase\(")


@pytest.fixture(scope="module")
def fp2():
    return kcp.fingerprint("4Q-2")


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("PORUNGA_DIR", str(tmp_path / "porunga"))
    return tmp_path / "porunga"


def test_side_lab_and_notebook_derive_the_same_key_in_separate_processes(fp2):
    code = ("import sys; sys.path.insert(0, %r); import kestrel_keys as kk, kestrel_campaigns as kcp; "
            "print(kk.airframe('4Q-2', 'sweep', 'fine', mock=False, fingerprint=kcp.fingerprint('4Q-2')))") % str(ROOT / "labs" / "aero" / "designs")
    other = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert other == kk.airframe("4Q-2", "sweep", "fine", mock=False, fingerprint=fp2)
    assert kk.airframe("4Q-2", "sweep", "fine", mock=True, fingerprint=fp2) != other              # mock is part of the identity
    assert kk.airframe("4Q-2", "sweep", "screening", mock=False, fingerprint=fp2) != other


def test_the_notebook_loads_what_a_side_lab_published_and_mocks_stay_apart(cache, fp2):
    summ = {"state": "completed", "records": [], "quality": "fine", "counts": {"n": 0}}
    porunga.save(kk.airframe("4Q-2", "sweep", "fine", mock=True, fingerprint=fp2), summ, metadata={"mock": True}, root=cache)
    porunga.save(kk.structure("4Q-2", "all", mock=True, fingerprint=fp2), {"state": "completed", "counts": {}}, metadata={"mock": True}, root=cache)
    table, got = kstu.inputs_from_cache(("4Q-2",), mock=True, fingerprints={"4Q-2": fp2})
    assert got["airframe_cfd"]["4Q-2"]["sweep"] == summ and "MOCK" in table.loc["airframe CFD, 4Q-2, sweep", "status"]
    with pytest.raises(kcp.PrerequisiteError, match="airframe_cfd.py --variant 4Q-2 --chapter sweep"):
        kstu.inputs_from_cache(("4Q-2",), mock=False, fingerprints={"4Q-2": fp2})          # a real request never reads a mock


def test_a_missing_optional_result_is_not_run_and_screening_needs_nothing(cache, fp2):
    table, got = kstu.inputs_from_cache(("4Q-2",), mock=False, screening=True, fingerprints={"4Q-2": fp2})
    assert not got["airframe_cfd"] and set(table["status"]) == {"NOT RUN (not in the cache)"}


def test_the_notebook_and_its_modules_launch_no_solver_and_read_no_run_directory():
    nb = json.loads((ROOT / "labs" / "aero" / "exp" / "43_kestrel_study.ipynb").read_text())
    src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert not SOLVER_CALLS.search(src), SOLVER_CALLS.search(src)
    assert "inputs_from_cache" in src and "side_labs/runs" not in src and "GATE" not in src
    for mod in ("kestrel_study.py", "kestrel_campaigns.py", "kestrel_keys.py"):
        hits = [m.group(0) for m in SOLVER_CALLS.finditer((ROOT / "labs" / "aero" / "designs" / mod).read_text())]
        assert not hits, (mod, hits)
