"""Halka in Fulgur: the section builds from the preset, the solve runs (the 1-D numbers under the mocks, the real
corona otherwise on a coarse mesh), and the thrust follows I dz / mu."""
import pytest

pytest.importorskip("fipy")

from vegeta import mock
from vegeta.fulgur.gas import DRY

import halka_fulgur as hf

G = 9.81


def test_rings_of_the_preset_and_the_offset():
    rings = hf.rings_of("mark4c", dz=0.012)
    assert all(z == 0.012 for _, role, z in rings if role == "ground")
    assert all(z == 0.0 for _, role, z in rings if role == "emitter")


def test_solve_runs_at_20kV():
    r = hf.solve_section("mark4c", 20e3, lc=3.0)
    assert r["converged"] and r["I"] > 0 and r["thrust"] > 0
    assert 0.001 < r["thrust_per_watt"] < 0.01                               # 1-10 mN/W
    if mock.installed():
        assert mock.is_mock(r)
    else:
        # the solved thrust is the ion current's axial momentum: I dz / mu within 15 % on the coarse mesh
        dz = max(z for _, role, z in hf.rings_of("mark4c") if role == "ground")
        assert r["thrust"] == pytest.approx(r["I"] * dz / DRY.mobility(), rel=0.15)


def test_one_d_matches_the_preset():
    e = hf.one_d("mark4c", 20e3)
    assert e["I"] > 0 and e["thrust"] > 0
