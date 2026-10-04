"""Shared helpers for the notebook data: ``propeller.noise`` against notebook 08 cell 60 run as it is, and
``_common.solve_modes`` through the mocks (mesh here, or the mesh of a solved case copied)."""
from __future__ import annotations

import math

import pytest
from vegeta import boreas, talos

from assemblies.components import propeller as pr
from assemblies.workflows import _common
from test_life import _dummy, cell


def test_noise_is_notebook_08_cell_60():
    from assemblies.workflows import quadcopter as wq
    prop, sec, system, pts = wq.drive_points(0.48)
    ns = cell("08_quadcopter.ipynb", 60, ["DIST", "ANGLE"])
    rows = {}
    for name, pt in (("hover", pts["hover"]), ("cruise", pts["cruise"]), ("full", pts["full"])):
        tones = boreas.gutin_harmonics(prop, pt.thrust, pt.aero.torque, pt.rpm, ns["DIST"], ns["ANGLE"], boreas.AIR, harmonics=6)
        bb = boreas.broadband_level(prop, pt.thrust, pt.rpm, ns["DIST"], boreas.AIR)
        rows[name] = (tones, bb, 10 * math.log10(10 ** (tones["total_tonal_db"] / 10) + 10 ** (bb / 10)))
    ours = pr.noise(prop, pts)
    for name, (tones, bb, one) in rows.items():
        assert ours[name]["tonal_dB"] == tones["total_tonal_db"] and ours[name]["broadband_dB"] == bb
        assert ours[name]["one_rotor_dB"] == one and ours[name]["BPF_hz"] == tones["blade_pass_hz"]
        assert ours[name]["spl_db"] == [float(x) for x in tones["spl_db"]]


def test_noise_in_water():
    from assemblies.workflows import boat
    spec = pr.get("60 mm 3-blade marine")
    prop, sec = spec.model(), spec.airfoil()
    system = boreas.Propulsion(prop, sec, pr.motor(boat.DRIVE["motor"]), pr.battery(boat.DRIVE["battery"]), rho=1025.0)
    pt = system.at_throttle(0.5, 1.0)
    w = pr.noise(prop, {"half": pt}, medium=boreas.SEA_WATER, harmonics=5)["half"]
    t = boreas.gutin_harmonics(prop, pt.thrust, pt.aero.torque, pt.rpm, 1.0, 90.0, boreas.SEA_WATER, harmonics=5)
    assert w["tonal_dB"] == t["total_tonal_db"] and len(w["frequency_hz"]) == 5


def test_solve_modes_meshes_or_copies_through_the_mocks(tmp_path, solvers, monkeypatch):
    m = _dummy(tmp_path, "modal")
    r = _common.solve_modes(m, tmp_path / "modal", n_modes=4)
    assert r.ok and r.metrics["n_modes"] == 4 and solvers.mesh.call_count == 1
    monkeypatch.setattr(talos.StructuralModel, "mesh_is_current", lambda self, d: str(d).endswith("solved"))
    (tmp_path / "solved").mkdir()
    (tmp_path / "solved" / "mesh.msh").write_text("x")
    r2 = _common.solve_modes(m, tmp_path / "modal2", n_modes=6, mesh_from=tmp_path / "solved")
    assert r2.ok and (tmp_path / "modal2" / "mesh.msh").is_file() and solvers.mesh.call_count == 1   # copied, not meshed
