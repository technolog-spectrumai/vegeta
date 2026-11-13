"""Yax: the spec's test vector, STL round trips, mass properties, packages, corruption, frames, CLI."""
import json
import math
import runpy
import shutil
from pathlib import Path

import pytest

import yax
from yax import stl, units

HERE = Path(__file__).resolve().parent
CUBE = HERE.parent / "examples" / "cube.yax"
make = runpy.run_path(str(HERE.parent / "examples" / "make_cube.py"))


def close(a, b, tol=1e-12):
    if isinstance(a, list):
        return all(close(x, y, tol) for x, y in zip(a, b)) and len(a) == len(b)
    return math.isclose(a, b, rel_tol=tol, abs_tol=tol)


def test_cube_vector_is_exact():
    d = yax.load(CUBE)
    assert d.name == "cube" and d.length_unit == "mm" and d.frame == {"up": "+z", "forward": "+x"}
    assert close(d.geometry["volume"], 1000.0) and close(d.geometry["com"], [5.0, 5.0, 5.0])
    assert d.geometry["bbox"] == [[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]]
    assert close(d.mass["kg"], 0.0027)
    i = 0.0027 * 0.01 ** 2 / 6                                     # m a^2 / 6 about the centre
    assert close(d.mass["inertia"], [[i, 0, 0], [0, i, 0], [0, 0, i]], 1e-12)
    assert len(d.mesh()) == 12


def test_cube_vector_is_reproducible(tmp_path):
    fresh = make["make"](tmp_path / "cube.yax")
    assert fresh.manifest == json.loads((CUBE / "yax.json").read_text())


def test_stl_ascii_and_binary_agree(tmp_path):
    tris = make["cube"]()
    stl.write(tmp_path / "b.stl", tris)
    ascii_text = "solid c\n" + "".join(
        "facet normal 0 0 0\nouter loop\n" + "".join(f"vertex {x} {y} {z}\n" for x, y, z in t) + "endloop\nendfacet\n"
        for t in tris) + "endsolid c\n"
    (tmp_path / "a.stl").write_text(ascii_text)
    assert stl.read(tmp_path / "a.stl") == stl.read(tmp_path / "b.stl") == [tuple(map(tuple, t)) for t in tris]


def test_mass_properties_of_an_offset_box():
    tris = [tuple((x * 2 + 1, y * 3 - 4, z * 4 + 7) for x, y, z in t) for t in make["cube"]()]   # 20 x 30 x 40 mm
    p = stl.mass_properties(tris)
    assert close(p["volume"], 24000.0, 1e-9) and close(p["com"], [11.0, 11.0, 27.0], 1e-9)
    m = 24000.0
    assert close([p["inertia"][k][k] for k in range(3)],
                 [m * (30**2 + 40**2) / 12, m * (20**2 + 40**2) / 12, m * (20**2 + 30**2) / 12], 1e-9)


def test_inward_winding_is_rejected():
    with pytest.raises(ValueError):
        stl.mass_properties([(a, c, b) for a, b, c in make["cube"]()])


def test_corruption_is_detected(tmp_path):
    bad = shutil.copytree(CUBE, tmp_path / "cube.yax")
    data = bytearray((bad / "body.stl").read_bytes())
    data[-10] ^= 0xFF
    (bad / "body.stl").write_bytes(bytes(data))
    assert any("SHA-256" in p for p in yax.check(bad))
    with pytest.raises(yax.Invalid):
        yax.load(bad)
    (bad / "body.stl").unlink()
    assert any("missing" in p for p in yax.check(bad))


def test_validate_reports_each_problem():
    m = json.loads((CUBE / "yax.json").read_text())
    assert yax.validate(m) == []
    for change, word in [({"yax": 2}, "yax"), ({"units": {"length": "in"}}, "units.length"),
                         ({"frame": {"up": "+z", "forward": "-z"}}, "frame"), ({"files": {}}, "body.stl"),
                         ({"identity": "0" * 64}, "identity"), ({"parts": [{"name": "x", "file": "x.stl"}]}, "parts")]:
        assert any(word in p for p in yax.validate({**m, **change})), change


def test_identity_ignores_version_and_formatting():
    a = {"tool": "t", "version": "1", "design_class": "C", "parameters": {"a": 1, "b": [2.5]}}
    assert yax.identity(a) == yax.identity({**a, "version": "2", "revision": "abc"})
    assert yax.identity(a) != yax.identity({**a, "parameters": {"a": 2, "b": [2.5]}})


def test_write_replaces_atomically_and_keeps_unknown_fields(tmp_path):
    out = tmp_path / "p.yax"
    src = yax.load(CUBE)
    base = {k: v for k, v in src.manifest.items() if k not in ("files", "identity", "yax")}
    yax.write(out, {**base, "x_custom": {"keep": True}}, {"body.stl": CUBE / "body.stl"})
    yax.write(out, {**base, "name": "second"}, {"body.stl": CUBE / "body.stl"})
    assert yax.load(out).name == "second" and "x_custom" not in yax.load(out).manifest
    assert [p.name for p in tmp_path.iterdir()] == ["p.yax"]                     # no staging leftovers
    with pytest.raises(yax.Invalid):
        yax.write(out, {**base, "units": {"length": "furlong"}}, {"body.stl": CUBE / "body.stl"})
    assert yax.load(out).name == "second"                                        # a failed write leaves the old one


def test_frames_and_units():
    r = units.rotation({"up": "+z", "forward": "+x"}, {"up": "+y", "forward": "+z"})
    assert units.apply(r, [0, 0, 1]) == [0, 1, 0] and units.apply(r, [1, 0, 0]) == [0, 0, 1]
    assert units.rotation({"up": "+z", "forward": "+x"}, {"up": "+z", "forward": "+x"}) == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    assert units.to_metres([10.0, [5.0]], "mm") == [0.01, [0.005]]
    with pytest.raises(ValueError):
        units.basis({"up": "+z", "forward": "-z"})


def test_cli(capsys):
    from yax.cli import main
    assert main(["check", str(CUBE)]) == 0
    assert main(["info", str(CUBE)]) == 0 and '"cube"' in capsys.readouterr().out
