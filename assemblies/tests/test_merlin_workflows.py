"""The propulsor and MERLIN workflows: a smoke library solves; a committed map file with the same space is loaded, not
solved; the MERLIN design export equals the one notebook 26 committed (both without CFD and FEA); the race and mission
run on a saved library; AGUYA's race against MERLIN reads both trees."""
import json
from pathlib import Path

import pytest

pytest.importorskip("cadquery")
from assemblies import vida                                            # noqa: E402
from assemblies.workflows import aguya, merlin, microjet, propulsors   # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def _close(a, b, where):
    if isinstance(b, dict):
        assert set(a) == set(b), where
        for k in b:
            _close(a[k], b[k], f"{where}/{k}")
    elif isinstance(b, (int, float)) and not isinstance(b, bool):
        assert a == pytest.approx(b, rel=1e-9), where
    else:
        assert a == b, where


@pytest.fixture(scope="module")
def libs(tmp_path_factory):
    d = tmp_path_factory.mktemp("libs")
    propulsors.run(fidelity="smoke", kinds=("propellers",), processors=4, out=d / "runs", vida_path=d / "p.vida",
                   progress=False)
    return d / "p.vida"


def test_smoke_library_solves_in_the_librarys_key_order(libs):
    t = vida.load(libs)
    s = t.results["summary"]
    assert s["propellers"]["failed"] == 0 and s["propellers"]["maps"] == 8
    e = t.child("propellers").results["entries"][0]
    assert e["thrust"].shape == (len(e["V"]), len(e["rpm"])) and e["blades"] in (2, 3)
    lib = propulsors.load_library(t)
    assert lib["kinds"] == ["propellers"] and lib["by_id"]


def test_committed_maps_are_loaded_not_solved(tmp_path):
    t = propulsors.run(fidelity="full", kinds=("propellers",), out=tmp_path / "runs", vida_path=tmp_path / "p.vida",
                       progress=False)
    assert t.child("propellers").results["source"] == "file propeller_maps.json" and t.child("propellers").results["n"] == 32


def test_merlin_design_equals_notebook_26s_export(tmp_path, libs):
    root = merlin.run(fidelity="smoke", run_cfd=False, run_fea=False, propulsors_vida=libs, out=tmp_path / "runs",
                      vida_path=tmp_path / "m.vida", export_path=tmp_path / "merlin_design.json", progress=False)
    ours = json.loads((tmp_path / "merlin_design.json").read_text())
    nb = json.loads((REPO / "notebooks/designs/data/merlin_design.json").read_text())       # written without CFD and FEA too
    for k in ("merlin_params", "common_mass_kg", "unit_mass_kg_1900W", "cd0_correction", "cd0_cfd", "mission", "distances_km",
              "max_electrical_w", "drive_efficiency", "extra_cd_area_m2", "cl_max", "oswald", "variants"):
        _close(ours[k], nb[k], k)
    assert ours["gust"]["load_factor"] == pytest.approx(nb["gust"]["load_factor"])
    assert root.child("race").results["n"] > 0
    assert {"tractor", "pusher"} <= set(root.child("mission").results["reach"])   # the smoke corner has no 90 mm fan
    again = merlin.run(fidelity="smoke", run_cfd=False, run_fea=False, propulsors_vida=libs, out=tmp_path / "runs",
                       vida_path=tmp_path / "m.vida", export=False, progress=False)
    assert again.child("race").status() == "reused" and again.child("mission").status() == "reused"


def test_aguya_races_merlin(tmp_path, libs):
    microjet.run(fidelity="smoke", run_cfd=False, run_fea=False, out=tmp_path / "mj", vida_path=tmp_path / "microjet.vida",
                 export=False, progress=False)
    merlin.run(fidelity="smoke", run_cfd=False, run_fea=False, propulsors_vida=libs, out=tmp_path / "me",
               vida_path=tmp_path / "merlin.vida", export=False, progress=False)
    r = aguya.run(fidelity="smoke", run_cfd=False, run_fea=False, engine_vida=tmp_path / "microjet.vida",
                  merlin_vida=tmp_path / "merlin.vida", propulsors_vida=libs, out=tmp_path / "ag", vida_path=tmp_path / "a.vida",
                  export=False, progress=False)
    race = r.child("merlin_race").results
    assert set(race["merlin"]) <= {"edf", "tractor", "pusher"} and race["merlin"]
    assert len(race["aguya"]["reach_s"]) == len(aguya.DISTANCES_KM)
    assert race["aguya"]["reach_s"][2] < min(m["reach_s"][2] for m in race["merlin"].values())     # the jet is faster at 10 km
