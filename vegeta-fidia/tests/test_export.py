import json

import numpy as np
import pytest
from _helpers import box_part

from vegeta.fidia.export import GLTF_FROM_CAD, export_scene, reimport_check

pytest.importorskip("trimesh")


@pytest.fixture
def parts():
    return [box_part("base", size=(100, 200, 20), color=(0.8, 0.1, 0.1, 1.0)),
            box_part("tower", size=(20, 20, 280), offset=(0, 0, 20), color=(0.1, 0.3, 0.9, 1.0))]


def test_all_formats_round_trip(tmp_path, parts):
    m = export_scene(parts, tmp_path)
    for f in ("model.glb", "model.gltf", "model.bin", "model.obj", "model.mtl", "model.stl", "manifest.json"):
        assert (tmp_path / f).is_file(), f
    assert m["files"]["gltf_buffers"] == ["model.bin"]
    assert m["units"]["glb"] == "m" and m["up_axis"]["obj"] == "Z"
    rep = reimport_check(tmp_path, parts, readers=("trimesh",))
    assert rep["ok"], rep
    assert rep["formats"]["glb"]["trimesh"]["parts"] == 2
    assert rep["formats"]["obj"]["trimesh"]["triangles"] == 24
    assert json.loads((tmp_path / "reimport.json").read_text())["ok"]


def test_parts_sharing_a_colour_keep_their_names(tmp_path, parts):
    legs = [box_part(f"leg_{i}", size=(10, 10, 50), offset=(30 * i, 0, 0), color=(0.2, 0.2, 0.2, 1.0)) for i in range(3)]
    export_scene(legs, tmp_path)
    rep = reimport_check(tmp_path, legs, readers=("trimesh",))
    assert rep["ok"], rep
    assert rep["formats"]["obj"]["trimesh"]["parts"] == 3


def test_pyvista_reader(tmp_path, parts):
    pytest.importorskip("pyvista")
    export_scene(parts, tmp_path)
    rep = reimport_check(tmp_path, parts)
    assert "pyvista" in rep["readers"]
    assert rep["ok"], rep
    assert rep["formats"]["glb"]["pyvista"]["triangles"] == 24


def test_gltf_is_y_up_metres(tmp_path, parts):
    import trimesh

    export_scene(parts, tmp_path)
    scene = trimesh.load(str(tmp_path / "model.glb"), force="scene")
    lo, hi = scene.bounds
    assert hi[1] == pytest.approx(0.3)  # height 300 mm -> +Y 0.3 m
    assert lo[2] == pytest.approx(-0.1) and hi[2] == pytest.approx(0.1)  # depth (CAD y) -> Z
    assert np.allclose(GLTF_FROM_CAD @ [0, 0, 300, 1], [0, 0.3, 0, 1])
    names = {n.get("name") for n in json.loads((tmp_path / "model.gltf").read_text())["nodes"]}
    assert {"base", "tower"} <= names


def test_missing_sidecars_fail(tmp_path, parts):
    export_scene(parts, tmp_path)
    (tmp_path / "model.bin").unlink()
    (tmp_path / "model.mtl").unlink()
    rep = reimport_check(tmp_path, parts, readers=("trimesh",))
    assert not rep["ok"]
    assert any("model.bin is missing" in p for p in rep["formats"]["gltf"]["problems"])
    assert any("model.mtl is missing" in p for p in rep["formats"]["obj"]["problems"])
    assert rep["formats"]["glb"]["ok"]


def test_wrong_colour_and_buffer_size_fail(tmp_path, parts):
    export_scene(parts, tmp_path)
    mtl = tmp_path / "model.mtl"
    text = mtl.read_text()
    kd = [line for line in text.splitlines() if line.startswith("Kd ")][0]
    mtl.write_text(text.replace(kd, "Kd 0.10000000 0.80000000 0.10000000", 1))
    with open(tmp_path / "model.bin", "ab") as f:
        f.write(b"\0\0\0\0")
    rep = reimport_check(tmp_path, parts, readers=("trimesh",))
    assert any("colours differ" in p for p in rep["formats"]["obj"]["problems"])
    assert any("bytes, expected" in p for p in rep["formats"]["gltf"]["problems"])


def test_step_is_copied_and_checked(tmp_path, parts):
    step = tmp_path / "in.step"
    step.write_text("ISO-10303-21;\nHEADER;\n")
    m = export_scene(parts, tmp_path / "out", step=step)
    assert m["files"]["step"] == "model.step"
    assert reimport_check(tmp_path / "out", parts, readers=("trimesh",))["formats"]["step"]["ok"]
