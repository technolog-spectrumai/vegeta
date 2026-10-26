"""The public API: parameter checks, instability detection, output, checkpoints, units."""
import base64
import json
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from mancha import D3Q19, InvalidParameterError, NumericalInstabilityError, Simulation, UnitConverter
from mancha.boundaries import sphere


def _channel(**kw):
    sim = Simulation(D3Q19(), shape=(12, 8, 6), viscosity=0.05, **kw)
    sim.set_boundary("x-", "velocity", velocity=(0.03, 0, 0))
    sim.set_boundary("x+", "pressure", density=1.0)
    for face in ("y-", "y+", "z-", "z+"):
        sim.set_boundary(face, "wall")
    return sim


@pytest.mark.parametrize("kw", [{}, {"viscosity": 0.1, "tau": 0.8}, {"viscosity": -0.1}, {"viscosity": 0.0}])
def test_viscosity_must_be_given_once_and_positive(kw):
    with pytest.raises(InvalidParameterError):
        Simulation(shape=(4, 4, 4), **kw)


def test_supersonic_inputs_refused_and_fast_ones_warned():
    sim = Simulation(shape=(4, 4, 4), viscosity=0.1)
    with pytest.raises(InvalidParameterError):
        sim.set_boundary("x-", "velocity", velocity=(0.6, 0, 0))
    with pytest.warns(UserWarning):
        sim.set_boundary("x-", "velocity", velocity=(0.2, 0, 0))
    with pytest.raises(InvalidParameterError):
        Simulation(shape=(4, 4, 4), viscosity=0.1, velocity=(0.7, 0, 0))


def test_unpaired_periodic_detected_at_run():
    sim = Simulation(shape=(4, 4, 4), viscosity=0.1).set_boundary("x-", "wall")
    with pytest.raises(InvalidParameterError):
        sim.run(1)


def test_nan_is_reported_not_swallowed():
    sim = Simulation(shape=(6, 6, 6), viscosity=0.1, check_interval=5)
    f = sim.populations()
    f[3, 2, 2, 2] = np.nan
    sim.backend.set_populations(f)
    with pytest.raises(NumericalInstabilityError, match="step 5"):
        sim.run(20)


def test_blow_up_is_reported():
    """A barely viscous fluid driven hard goes unstable; the run must say so, not return garbage."""
    with pytest.warns(UserWarning):
        sim = Simulation(shape=(16, 16, 1), tau=0.5005, check_interval=10)
    k = 2 * np.pi / 16
    x = np.arange(16)
    X, Y = np.meshgrid(x, x, indexing="ij")
    u = np.zeros((3, 16, 16, 1))
    u[0, :, :, 0] = 0.3 * np.sin(k * X) * np.cos(k * Y) + 0.2 * np.sin(3 * k * Y)
    u[1, :, :, 0] = -0.3 * np.cos(k * X) * np.sin(k * Y)
    with pytest.warns(UserWarning):
        sim.initialize(velocity=u)
    with pytest.raises(NumericalInstabilityError):
        sim.run(5000)


def test_history_and_callback():
    seen = []
    sim = _channel(history_interval=10)
    sim.run(30, callback=lambda s: seen.append(s.steps_done), callback_interval=15)
    assert [r.step for r in sim.history] == [0, 10, 20, 30]
    assert seen == [15, 30]
    assert sim.history[-1].kinetic_energy > 0


def test_save_writes_fields_metadata_and_vtk(tmp_path):
    sim = _channel(units=UnitConverter(dx=1e-3, dt=1e-4))
    sim.set_solid(sphere(sim.shape, (5, 3.5, 2.5), 1.5))
    sim.run(20)
    out = sim.save(tmp_path / "run")
    rho, u = np.load(out / "density.npy"), np.load(out / "velocity.npy")
    assert rho.shape == (12, 8, 6) and u.shape == (3, 12, 8, 6)
    meta = json.loads((out / "metadata.json").read_text())
    assert meta["steps"] == 20 and meta["boundaries"]["x+"]["kind"] == "pressure"
    assert meta["collision"] == {"kind": "bgk", "tau": sim.collision.tau} and meta["units"]["dx"] == 1e-3
    assert np.load(out / "solid.npy").sum() == meta["solid_voxels"] > 0
    piece = ET.parse(out / "fields.vti").getroot()
    arrays = {a.get("Name"): a for a in piece.iter("DataArray")}
    assert set(arrays) == {"density", "pressure", "solid", "velocity"}
    header_len = len(base64.b64encode(b"\0" * 8))
    data = np.frombuffer(base64.b64decode(arrays["density"].text[header_len:]), dtype=np.float64)
    assert np.allclose(data, rho.ravel(order="F"))


def test_vtk_reads_back_through_pyvista(tmp_path):
    pv = pytest.importorskip("pyvista")
    sim = _channel()
    sim.run(10)
    grid = pv.read(sim.save(tmp_path) / "fields.vti")
    assert grid.dimensions == (12, 8, 6)
    u = sim.velocity()
    assert np.allclose(grid.point_data["velocity"][:, 0], u[0].ravel(order="F"))
    assert np.allclose(grid.point_data["density"], sim.density().ravel(order="F"))


def test_checkpoint_restart_is_bit_identical(tmp_path):
    a = _channel(history_interval=10)
    a.run(25)
    path = a.checkpoint(tmp_path / "chk.npz")
    a.run(25)
    b = Simulation.restore(path)
    assert b.steps_done == 25 and len(b.history) == 3
    b.run(25)
    assert np.array_equal(a.populations(), b.populations())
    assert b.boundaries.faces == a.boundaries.faces


def test_solid_voxels_report_rest():
    sim = Simulation(shape=(10, 6, 6), viscosity=0.1, velocity=(0.02, 0, 0))
    mask = np.zeros(sim.shape, bool)
    mask[4:6, 2:4, 2:4] = True
    sim.set_solid(mask).run(20)
    assert np.allclose(sim.velocity()[:, mask], 0.0)
    assert np.isfinite(sim.solid_force()).all() and sim.solid_force()[0] > 0


def test_unit_converter():
    u = UnitConverter.from_velocity(dx=0.01, velocity=2.0, lattice_velocity=0.05, rho=1000.0)
    assert np.isclose(u.dt, 2.5e-4)
    assert np.isclose(u.velocity(0.05), 2.0) and np.isclose(u.to_lattice_velocity(2.0), 0.05)
    nu = 1e-6
    assert np.isclose(u.viscosity(u.to_lattice_viscosity(nu)), nu)
    v = UnitConverter.from_viscosity(dx=0.01, viscosity=1e-6, lattice_viscosity=0.01)
    assert np.isclose(v.to_lattice_viscosity(1e-6), 0.01)
    assert np.isclose(u.pressure(1.0 + 3e-3), 1e-3 * 1000.0 * (0.01 / 2.5e-4) ** 2)
    with pytest.raises(InvalidParameterError):
        UnitConverter(dx=0.0, dt=1.0)


def test_readme_example_runs(tmp_path):
    """The README example, on a small grid and few steps."""
    sim = Simulation(lattice=D3Q19(), shape=(16, 12, 12), viscosity=0.02, backend="numpy")
    sim.set_boundary("x-", "velocity", velocity=(0.05, 0, 0))
    sim.set_boundary("x+", "pressure", density=1.0)
    for face in ("y-", "y+", "z-", "z+"):
        sim.set_boundary(face, "wall")
    sim.run(steps=50)
    assert sim.velocity().shape == (3, 16, 12, 12) and sim.density().shape == (16, 12, 12)
    sim.save(tmp_path / "results")
    assert (tmp_path / "results" / "fields.vti").exists()
