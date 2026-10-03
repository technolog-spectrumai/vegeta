"""The turbojet CAD (notebook 28): the impeller, the turbine wheel and the engine build as single valid solids, the
compressor passage splits into the four closed STLs of Aeromant's compressor_mrf, and the sizing follows the cycle.

Run: cd /home/user/vegeta && .venv/bin/python -m pytest -q notebooks/designs/tests/test_turbojet.py   (~20 s)
"""
import math

import numpy as np
import pytest

pytest.importorskip("cadquery")
import turbojet as tj                                   # noqa: E402
from vegeta.boreas import microjet as mj               # noqa: E402


@pytest.fixture(scope="module")
def params():
    return tj.sized(mj.from_catalogue("140 N class"))


def test_sizing_follows_the_cycle(params):
    e = mj.from_catalogue("140 N class")
    assert params["impeller_diameter"] == pytest.approx(e.impeller_diameter * 1000)
    assert params["nozzle_area"] == pytest.approx(e.a8 * 1e6)
    assert 35 < params["inducer_angle_deg"] < 70


@pytest.mark.parametrize("part", ["impeller", "turbine", "engine"])
def test_parts_build(params, part):
    g = tj.Turbojet().generate(**dict(params, part=part))
    assert g.shape.isValid() and len(g.shape.Solids()) == 1


def test_masses_are_plausible(params):
    m = tj.masses(params)
    assert 0.03 < m["impeller_kg"] < 0.3 and 0.04 < m["turbine_kg"] < 0.3


def test_compressor_surfaces_close_the_passage(params, tmp_path):
    from vegeta.aeromant.stl import Surface, read_stl

    out = tj.compressor_surfaces(params, tmp_path)
    s = {k: read_stl(v) for k, v in out["paths"].items()}
    assert set(s) == {"impeller", "shroud", "inlet", "outlet"} and out["faces"]["inlet"] == 1
    st = tj.Turbojet.stations(params)
    imp_lo, imp_hi = s["impeller"].bbox
    assert np.hypot(imp_hi[1], imp_hi[2]) <= math.sqrt(2) * st["r2"] + 0.5 and imp_hi[1] <= st["r2"] + 0.01
    assert s["shroud"].bbox[1][1] == pytest.approx(st["r3"], abs=0.05)
    vol = Surface(np.vstack([x.triangles for x in s.values()])).volume
    prof = tj.passage_profile(params)
    x, r = prof[:, 0], prof[:, 1]
    revolved = abs(np.sum((x[1:] - x[:-1]) * (r[1:] ** 2 + r[1:] * r[:-1] + r[:-1] ** 2)) * math.pi / 3)
    assert 0.8 * revolved < vol < revolved                     # the passage minus the blades
    loc = np.array(out["location_in_mesh"]) * 1000
    assert st["x_in"] < loc[0] < st["x_spinner"]
