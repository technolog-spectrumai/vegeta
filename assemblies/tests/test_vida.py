"""The assembly tree and the .vida file: build, walk, save, load, reuse by key, include levels, format version."""
import json
import math
import zipfile

import numpy as np
import pytest

from assemblies import vida
from assemblies.vida import Assembly


def _tree(tmp_path, back=(60e3, 80e3)):
    root = Assembly("aguya", "aircraft", params={"span": 1.4, "mission_km": [5, 10]})
    eng = root.add(Assembly("engine", "turbojet", params={"engine_class": "140 N class"}))
    comp = eng.add(Assembly("compressor", "compressor", params={"back_pressures": np.array(back), "fidelity": "quick"}))
    eng.add(Assembly("wheels", "wheel_fea", params={"element": 2.5}))
    stl = tmp_path / "impeller.stl"
    stl.write_text("solid x\nendsolid x\n")
    case = tmp_path / "case_p0"
    (case / "system").mkdir(parents=True, exist_ok=True)
    (case / "system" / "controlDict").write_text("FoamFile {}\n")
    comp.record(speedline={"mass_flow": np.array([0.2, 0.25]), "pr": [2.9, float("nan")], "ok": [True, False]},
                calibration={"eta_c": 0.74})
    comp.attach("impeller.stl", stl, "geometry")
    comp.attach("case_p0", case, "cases")
    eng.child("wheels").not_run("run_fea=False")
    return root


def test_tree_paths_and_errors(tmp_path):
    root = _tree(tmp_path)
    assert [p for p, _ in root.walk()] == ["", "engine", "engine/compressor", "engine/wheels"]
    assert root.child("engine/compressor").kind == "compressor"
    with pytest.raises(KeyError, match="no sub-assembly 'fan'"):
        root.child("engine/fan")
    with pytest.raises(ValueError, match="already has"):
        root.add(Assembly("engine", "x"))
    with pytest.raises(ValueError, match="letters, digits"):
        Assembly("a/b", "x")
    with pytest.raises(TypeError, match="no plain form"):
        Assembly("x", "x", params={"f": object()})
    with pytest.raises(TypeError, match="cannot be stored"):
        root.record(bad=object())
    assert root.child("engine/compressor").params["back_pressures"] == [60000.0, 80000.0]   # params are plain JSON


def test_key_follows_params_and_children(tmp_path):
    a, b = _tree(tmp_path), _tree(tmp_path)
    assert a.key == b.key and a.child("engine").key == b.child("engine").key
    c = _tree(tmp_path, back=(60e3, 90e3))
    assert c.child("engine/compressor").key != a.child("engine/compressor").key
    assert c.child("engine").key != a.child("engine").key and c.key != a.key        # the change reaches the root
    assert c.child("engine/wheels").key == a.child("engine/wheels").key              # siblings keep theirs
    a.child("engine/compressor").record(extra=1)
    assert a.key == b.key                                                            # results are not identity


def test_round_trip_results_level(tmp_path):
    root = _tree(tmp_path)
    path = root.save(tmp_path / "aguya.vida")
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
    assert "manifest.json" in names and "tree.json" in names
    assert "nodes/aguya/engine/compressor/results.json" in names and "nodes/aguya/engine/compressor/arrays.npz" in names
    assert not any("/files/" in n for n in names)                                   # results level: no files inside
    back = vida.load(path)
    assert back.key == root.key and str(back).splitlines()[0].startswith("aguya (aircraft)")
    sl = back.child("engine/compressor").results["speedline"]
    assert isinstance(sl["mass_flow"], np.ndarray) and sl["mass_flow"].tolist() == [0.2, 0.25]
    assert sl["pr"][0] == 2.9 and math.isnan(sl["pr"][1]) and sl["ok"] == [True, False]
    assert back.child("engine/wheels").status() == "NOT RUN"
    assert back.meta["manifest"]["include"] == "results" and back.meta["manifest"]["vida"] == vida.FORMAT
    ref = back.child("engine/compressor").files["impeller.stl"]
    assert ref.member is None and ref.sha256 and ref.level == "geometry"
    assert back.child("engine/compressor").file("impeller.stl").read_text().startswith("solid")   # original still there
    (tmp_path / "impeller.stl").write_text("solid changed\nendsolid\n")
    with pytest.raises(FileNotFoundError, match="include='geometry'"):
        back.child("engine/compressor").file("impeller.stl")


@pytest.mark.parametrize("level,stl,case", [("geometry", True, False), ("mesh", True, False), ("cases", True, True)])
def test_include_levels_carry_files(tmp_path, level, stl, case):
    root = _tree(tmp_path)
    path = root.save(tmp_path / "a.vida", include=level)
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
    assert ("nodes/aguya/engine/compressor/files/impeller.stl" in names) == stl
    assert ("nodes/aguya/engine/compressor/files/case_p0/system/controlDict" in names) == case
    (tmp_path / "impeller.stl").unlink()                                             # moved to another machine
    import shutil
    shutil.rmtree(tmp_path / "case_p0")
    back = vida.load(path)
    comp = back.child("engine/compressor")
    assert comp.file("impeller.stl", to=tmp_path / "x").read_text().startswith("solid")
    if case:
        assert (comp.file("case_p0", to=tmp_path / "x") / "system" / "controlDict").is_file()
    resaved = vida.load(back.save(tmp_path / "b.vida", include=level))                # carried over from the archive
    assert resaved.child("engine/compressor").file("impeller.stl", to=tmp_path / "y").is_file()


def test_reuse_copies_only_unchanged_nodes(tmp_path):
    prior = vida.load(_tree(tmp_path).save(tmp_path / "a.vida"))
    same = _tree(tmp_path)
    same.child("engine/compressor").forget()
    assert same.reuse(prior) == ["engine/compressor"]                                 # the only node with results
    assert same.child("engine/compressor").status() == "reused"
    assert same.child("engine/compressor").results["calibration"] == {"eta_c": 0.74}
    changed = _tree(tmp_path, back=(60e3, 95e3))
    changed.child("engine/compressor").forget()
    assert changed.reuse(prior) == [] and changed.child("engine/compressor").results == {}
    assert Assembly("x", "y").reuse(None) == []


def test_graft_a_loaded_tree_as_a_sub_assembly(tmp_path):
    engine = vida.load(_tree(tmp_path).child("engine").copy("microjet").save(tmp_path / "microjet.vida"))
    plane = Assembly("aguya2", "aircraft", params={"span": 1.5})
    plane.add(engine.copy("engine"))
    assert plane.child("engine/compressor").results["calibration"]["eta_c"] == 0.74
    back = vida.load(plane.save(tmp_path / "aguya2.vida"))
    assert back.child("engine/compressor").key == engine.child("compressor").key


def test_newer_format_is_refused(tmp_path):
    path = _tree(tmp_path).save(tmp_path / "a.vida")
    with zipfile.ZipFile(path) as z:
        members = {n: z.read(n) for n in z.namelist()}
    man = json.loads(members["manifest.json"])
    man["vida"] = vida.FORMAT + 1
    members["manifest.json"] = json.dumps(man).encode()
    with zipfile.ZipFile(path, "w") as z:
        for n, b in members.items():
            z.writestr(n, b)
    with pytest.raises(ValueError, match="update Vegeta"):
        vida.load(path)


def test_include_validation(tmp_path):
    with pytest.raises(ValueError, match="include must be"):
        _tree(tmp_path).save(tmp_path / "a.vida", include="everything")
    assert vida._level(("results", "mesh")) == "mesh"


def test_paths_inside_the_repository_are_stored_relative(tmp_path):
    inside = vida.REPO / "runs" / "assemblies" / "_test_vida"
    inside.mkdir(parents=True, exist_ok=True)
    try:
        (inside / "part.step").write_text("ISO-10303-21;\n")
        n = Assembly("x", "y")
        n.attach("part.step", inside / "part.step")
        path = n.save(tmp_path / "x.vida")
        with zipfile.ZipFile(path) as z:
            f = json.loads(z.read("tree.json"))["files"]["part.step"]
        assert f["path_from"] == "repo" and f["path"] == "runs/assemblies/_test_vida/part.step"
        assert vida.load(path).file("part.step") == inside / "part.step"
    finally:
        import shutil
        shutil.rmtree(inside)
