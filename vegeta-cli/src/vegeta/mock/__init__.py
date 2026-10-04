"""Mocks for the long simulations: test a notebook end to end before the full run.

In a notebook (the first cell)::

    DEBUG = False                      # True, or VEGETA_DEBUG=1 in the environment: mocks instead of the simulations
    from vegeta import mock
    DEBUG = mock.setup(DEBUG)          # installs the mocks when DEBUG (or VEGETA_DEBUG=1) is on

    VEGETA_DEBUG=1 jupyter nbconvert --to notebook --execute experiment.ipynb --output experiment-run.ipynb

With the mocks on, every FEA and CFD run returns at once with plausible, fixed numbers in the real ``Result`` types,
so each cell runs and its tables fill, but the numbers mean nothing:

- Talos: ``StructuralModel.mesh`` / ``solve`` / ``solve_modes`` (``ensure`` and ``solve_models`` go through them, and
  so does ``vegeta.core``), ``assess_fatigue``, ``read_frd``; the plots and animations of results become inert
  placeholders.
- Aeromant: ``OpenFOAMEnvironment.detect`` (a default environment when OpenFOAM is missing), ``CFDCase.run`` and
  ``results``, ``read_case_results`` (``ensure``, ``run_cases``, ``vegeta.core`` go
  through them); ``openfoam_sampler`` is a uniform stream; the field plots, coefficient plots and particle movies
  become placeholders. ``prepare`` and the set-up plots stay real (they only write and draw the case).
- The result cache (``cache=``) is switched off, so no mocked result is ever saved as a real one.

- Mellonia: ``slice_stl`` writes a short synthetic G-code (read back with the real ``read_gcode``) instead of running
  PrusaSlicer.

Nothing else is mocked: geometry, propeller and flight models, Chiron/MuJoCo episodes run as usual.
``mock.uninstall()`` puts the real functions back.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

ENV = "VEGETA_DEBUG"
_saved: list[tuple[object, str, object]] = []
calls: dict[str, int] = {}


def debug(default: bool = False) -> bool:
    """The debug switch: ``VEGETA_DEBUG`` (1/true/on/yes) when it is set, else ``default``."""
    v = os.environ.get(ENV)
    if v is None or v == "":
        return bool(default)
    return v.strip().lower() in ("1", "true", "on", "yes")


def setup(default: bool = False) -> bool:
    """Install the mocks when the debug switch (``debug(default)``) is on; returns the switch."""
    on = debug(default)
    if on:
        install()
    return on


def installed() -> bool:
    return bool(_saved)


class Metrics(dict):
    """Result metrics that answer every key: the known ones with plausible values, any other with 1.0."""

    def __missing__(self, key):
        return 1.0


def _count(name):
    calls[name] = calls.get(name, 0) + 1


def placeholder(*args, **kwargs):
    """What a mocked plot returns: an object that accepts any attribute or call (``plotter.add_text(...)``)."""
    return MagicMock(name="vegeta.mock placeholder")


# ------------------------------------------------------------------------------------------------ Talos
FEA = {"max_von_mises": 12.0, "safety_factor_yield": 4.0, "safety_factor_temperature": 3.0, "max_displacement": 0.2,
       "n_elements": 1000, "n_nodes": 400, "max_temperature": 300.0}
MODES_HZ = [45.0, 110.0, 190.0, 320.0, 480.0, 650.0, 870.0, 1100.0, 1400.0, 1750.0, 2100.0, 2500.0]
N_NODES = 10


def _talos_mesh(self, workdir, progress=False, *, cache=None):
    from vegeta.talos import model as tm
    from vegeta.talos.result import Result
    _count("talos.mesh")
    w = Path(workdir)
    w.mkdir(parents=True, exist_ok=True)
    (w / tm.MESH_FILE).write_text("vegeta.mock: no mesh\n")
    res = Result(kind="talos.mesh", metrics=Metrics(n_nodes=400, n_elements=1000),
                 metadata={"model": self.config(), "mesh_key": self._mesh_key(), "mock": True})
    res.artifacts["mesh"] = w / tm.MESH_FILE
    res.artifacts["mesh_summary"] = res.save_json(w / tm.MESH_SUMMARY)
    return res


def _fea_metrics(model) -> Metrics:
    """The keys and types of a real static summary (talos StructuralModel._summarise), with fixed values; the
    reactions balance the applied point forces."""
    applied = np.zeros(3)
    for load in model.loads:
        if all(hasattr(load, a) for a in ("fx", "fy", "fz")):
            applied += [load.fx or 0.0, load.fy or 0.0, load.fz or 0.0]
    sup = [s.region for s in model.supports]
    share = (-applied / max(len(sup), 1)).tolist()
    m = Metrics(FEA)
    m.update(element_type="C3D10", max_displacement_node=N_NODES, max_displacement_location=[90.0, 0.0, 0.0],
             displacement_min=[0.0, 0.0, -0.2], displacement_max=[0.0, 0.0, 0.2], max_von_mises_location=[0.0, 0.0, 0.0],
             reactions={r: share for r in sup}, reaction_total=(-applied).tolist(), applied_force_total=applied.tolist(),
             safety_factor_location=[0.0, 0.0, 0.0], safety_factor_von_mises=FEA["max_von_mises"])
    ys = getattr(model.material, "yield_strength", None)
    m["safety_factor_yield"] = ys / FEA["max_von_mises"] if ys else None
    return m


def mesh_data(path=None, regions=()):
    """A mocked ``read_mesh``: a ten-node line of tetrahedra, every region on two of its nodes."""
    from vegeta.talos.mesh import MeshData
    ids = np.arange(1, N_NODES + 1)
    coords = np.column_stack([np.linspace(0.0, 90.0, N_NODES), np.zeros(N_NODES), np.zeros(N_NODES)])
    conn = np.array([[i, i + 1, i + 2, i + 3] for i in range(1, N_NODES - 2)])
    names = list(regions) or ["all"]
    return MeshData(ids, coords, "C3D4", np.arange(1, len(conn) + 1), conn,
                    {r: ids[:2] for r in names}, {r: np.array([[1, 2, 3]]) for r in names})


def _talos_solve(self, workdir, *, executable="ccx", threads=1, timeout=None, progress=False, cancel=None, cache=None):
    from vegeta.talos.result import Result
    _count("talos.solve")
    w = Path(workdir)
    w.mkdir(parents=True, exist_ok=True)
    (w / "model.frd").write_text("vegeta.mock: no results\n")
    res = Result(kind="talos.solve", metrics=_fea_metrics(self),
                 metadata={"model": self.config(), "solve_key": self._solve_key(), "mock": True})
    res.artifacts.update(mesh=w / "mesh.msh", frd=w / "model.frd")
    res.artifacts["summary"] = w / "summary.json"
    res.save_json(res.artifacts["summary"])
    return res


def _talos_modes(self, workdir, n_modes=10, *, executable="ccx", threads=1, timeout=None, progress=False, cancel=None,
                 cache=None):
    from vegeta.talos.result import Result
    _count("talos.solve_modes")
    w = Path(workdir)
    w.mkdir(parents=True, exist_ok=True)
    f = [MODES_HZ[i % len(MODES_HZ)] * (1 + i // len(MODES_HZ)) for i in range(int(n_modes))]
    res = Result(kind="talos.modes", metadata={"model": self.config(), "n_modes": n_modes, "mock": True},
                 metrics=Metrics(n_nodes=400, n_elements=1000, n_modes=len(f), frequencies_hz=f, first_frequency_hz=f[0],
                                 effective_modal_mass=[1.0 / len(f)] * len(f), total_effective_mass=1.0,
                                 point_mass_total=float(sum(m.mass for m in self.masses))))
    res.artifacts.update(mesh=w / "mesh.msh", frd=w / "modes.frd")
    res.artifacts["summary"] = w / "modes_summary.json"
    res.save_json(res.artifacts["summary"])
    return res


def frd_fields(path=None):
    """A mocked ``.frd`` read: ``N_NODES`` nodes along x, a uniaxial stress rising to 12 MPa, a small displacement."""
    from vegeta.talos import FieldResults
    ids = np.arange(1, N_NODES + 1)
    coords = np.column_stack([np.linspace(0.0, 90.0, N_NODES), np.zeros(N_NODES), np.zeros(N_NODES)])
    stress = np.zeros((N_NODES, 6))
    stress[:, 0] = np.linspace(1.2, 12.0, N_NODES)
    disp = np.zeros((N_NODES, 3))
    disp[:, 2] = np.linspace(0.0, 0.2, N_NODES)
    return FieldResults(ids, coords, {"STRESS": stress, "DISP": disp, "NDTEMP": np.full((N_NODES, 1), 300.0)})


def _assess_fatigue(unit_cases, spectrum, curve, *, workdir=None):
    from vegeta import talos
    _count("talos.assess_fatigue")
    if isinstance(spectrum, (str, Path)):
        spectrum = json.loads(Path(spectrum).read_text())
    blocks = spectrum.get("blocks", [])
    fr = frd_fields()
    load = sum(abs(b["amplitude"]) * b["cycles"] for b in blocks) * 1e-9 + 1e-12
    damage = load * np.linspace(0.1, 1.0, N_NODES)
    hot = int(np.argmax(damage))
    hours = spectrum.get("duration_s", 3600.0) / 3600.0
    d = float(damage[hot])
    r = talos.Result(kind="talos.fatigue", metadata={"mock": True},
                     metrics=Metrics(damage_per_pass=d, passes_to_failure=1.0 / d, hours_to_failure=hours / d,
                                     spectrum_duration_h=hours, blocks=len(blocks), hotspot_node=int(fr.node_ids[hot]),
                                     hotspot_location=fr.coords[hot].tolist(), nodes_above_1pct=N_NODES))
    if workdir is not None:
        Path(workdir).mkdir(parents=True, exist_ok=True)
    total = sum(abs(b["amplitude"]) * b["cycles"] for b in blocks) or 1.0
    contributions = {b.get("source") or b["pattern"]: d * abs(b["amplitude"]) * b["cycles"] / total for b in blocks}
    return talos.FatigueResult(fr.node_ids, fr.coords, damage, contributions, r)


# ------------------------------------------------------------------------------------------------ Aeromant
CFD = {"Cd": 0.03, "Cl": 0.4, "Cm": 0.01, "drag_force_N": 5.0, "lift_force_N": 20.0, "side_force_N": 0.0,
       "thrust_N": 5.0, "torque_Nm": 0.1, "power_W": 60.0, "efficiency": 0.6, "figure_of_merit": 0.6,
       "mach_number": 0.3, "converged": True, "iterations": 100, "mesh_cells": 1000, "mesh_ok": True}


def _cfd_result(case, kind="aeromant.results"):
    from vegeta.aeromant.result import Result
    w = Path(case.workdir)
    w.mkdir(parents=True, exist_ok=True)
    m = Metrics(CFD)
    p = getattr(case, "user_parameters", {}) or {}
    if "velocity" in p:
        m["velocity"] = p["velocity"]
    if "outlet_pressure" in p and "inlet_total_pressure" in p:      # a compressor speed-line point
        pr = p["outlet_pressure"] / p["inlet_total_pressure"]
        flow = max(0.05, 0.40 - 0.08 * pr)
        m.update(mass_flow_kg_s=flow, corrected_mass_flow_kg_s=flow, pressure_ratio_tt=1.05 * pr, efficiency_tt=0.72,
                 efficiency_from_torque=0.70, work_coefficient=0.6, shaft_power_W=1000.0 * flow * 1.05 * pr * 40.0)
    return Result(kind=kind, metrics=m, artifacts={"case": w}, metadata={"mock": True})


def _cfd_run(self, steps=None, *, progress=False, cancel=None, timeout=None, processors=1, cache=None):
    _count("aeromant.run")
    res = _cfd_result(self, "aeromant.run")
    res.metadata["steps"] = list(steps) if steps is not None else [s.name for s in self.pipeline(processors)]
    res.artifacts["summary"] = res.save_json(Path(self.workdir) / "summary.json")
    return res


def _cfd_results(self, average_window=50):
    _count("aeromant.results")
    return _cfd_result(self)


def _read_case_results(workdir, average_window=50):
    from vegeta.aeromant.result import Result
    _count("aeromant.read_case_results")
    return Result(kind="aeromant.results", metrics=Metrics(CFD), artifacts={"case": Path(workdir)}, metadata={"mock": True})


def _read_results(case, time=None):
    """A mocked solved case as pyvista reads it (``internalMesh`` + ``boundary`` with a ``body`` patch): small real
    datasets with uniform U, p, k and nut, so code that walks the blocks runs."""
    import pyvista as pv
    _count("aeromant.read_results")

    def fields(ds):
        for name, v in (("p", -50.0), ("k", 0.01), ("nut", 1e-5)):
            ds.cell_data[name] = np.full(ds.n_cells, v)
            ds.point_data[name] = np.full(ds.n_points, v)
        ds.cell_data["U"] = np.tile([10.0, 0.0, 0.0], (ds.n_cells, 1))
        ds.point_data["U"] = np.tile([10.0, 0.0, 0.0], (ds.n_points, 1))
        return ds
    internal = fields(pv.ImageData(dimensions=(6, 4, 4), spacing=(0.2, 0.2, 0.2), origin=(-0.5, -0.3, -0.3))
                      .cast_to_unstructured_grid())
    boundary = pv.MultiBlock({"body": fields(pv.Sphere(radius=0.05)), "inlet": fields(pv.Plane())})
    return pv.MultiBlock({"internalMesh": internal, "boundary": boundary})


def _openfoam_sampler(case, time=None):
    """A uniform stream at the case's velocity (else 10 m/s along x), valid everywhere."""
    _count("aeromant.openfoam_sampler")
    p = getattr(case, "user_parameters", {}) or {}
    v = float(p.get("velocity", 10.0) or 10.0)

    def sample(points):
        pts = np.asarray(points, float).reshape(-1, 3)
        u = np.zeros_like(pts)
        u[:, 0] = v
        return u, np.ones(len(pts), bool)
    return sample


def _placeholder_file(path) -> Path:
    """An empty file where a movie would be (so ``IPython.display.Video(path)`` finds a file)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"")
    return p


def _animate(obj, path, *args, **kwargs):
    """A mocked animation (talos ``animate``, aeromant ``animate_particles``): an empty file at ``path``."""
    _count("animate")
    return _placeholder_file(path)


def _make_movie(case, path, **kw):
    _count("aeromant.make_movie")
    return _placeholder_file(path)


def _concat_videos(paths, out, **kw):
    _count("aeromant.concat_videos")
    return _placeholder_file(out)


# ------------------------------------------------------------------------------------------------ Mellonia
def _gcode(settings, n_layers=5, size=20.0) -> str:
    """A short G-code in PrusaSlicer's format: square perimeters, the summary comments, the config block."""
    cfg = dict(settings.merged()) if hasattr(settings, "merged") else {}
    lh = float(str(cfg.get("layer_height", 0.2)).rstrip("%") or 0.2)
    cfg.setdefault("layer_height", lh)
    cfg.setdefault("first_layer_height", lh)
    cfg.setdefault("filament_density", 1.24)
    cfg.setdefault("filament_cost", 25)
    out = ["; generated by vegeta.mock (no slicer was run)", "G21", "G90", "M83"]
    for k in range(n_layers):
        z = lh * (k + 1)
        out += [";LAYER_CHANGE", f";Z:{z:.3f}", f"G1 Z{z:.3f} F600", "G1 X0 Y0"]
        out += [f"G1 X{size} Y0 E1.0", f"G1 X{size} Y{size} E1.0", f"G1 X0 Y{size} E1.0", "G1 X0 Y0 E1.0"]
    out += ["; filament used [mm] = 400.0", "; filament used [cm3] = 1.0", "; filament used [g] = 1.24",
            "; total filament used [g] = 1.24", "; filament cost = 0.03", "; total filament cost = 0.03",
            "; estimated printing time (normal mode) = 12m 30s", "; estimated printing time (silent mode) = 13m 0s",
            "; estimated first layer printing time (normal mode) = 2m 0s", "; prusaslicer_config = begin"]
    out += [f"; {k} = {v}" for k, v in sorted(cfg.items())] + ["; prusaslicer_config = end"]
    return "\n".join(out) + "\n"


def _slice_stl(stl, settings, orientation, workdir, *, executable="prusa-slicer", timeout=None, cancel=None):
    from vegeta.mellonia import read_gcode
    from vegeta.mellonia.result import Result
    _count("mellonia.slice_stl")
    w = Path(workdir)
    w.mkdir(parents=True, exist_ok=True)
    gcode = w / (Path(stl).stem + ".gcode")
    gcode.write_text(_gcode(settings))
    info = read_gcode(gcode)
    eff = w / "effective_config.ini"
    eff.write_text("# full configuration (vegeta.mock: no slicer was run)\n"
                   + "".join(f"{k} = {v}\n" for k, v in sorted(info.config.items())))
    res = Result(kind="mellonia.slice", metrics=Metrics(info.metrics(), slicing_status="ok"), metadata={"mock": True},
                 artifacts={"input_stl": Path(stl), "gcode": gcode, "effective_config": eff})
    res.artifacts["summary"] = w / "summary.json"
    res.save_json(res.artifacts["summary"])
    return res


# ------------------------------------------------------------------------------------------------ install
def _patch(owner, name, new):
    _saved.append((owner, name, owner.__dict__[name] if isinstance(owner, type) and name in owner.__dict__
                   else getattr(owner, name)))
    setattr(owner, name, new)


def _real_or_placeholder_show(real):
    def show(plotter, *args, **kwargs):
        if isinstance(plotter, MagicMock) or plotter is None:
            return None
        return real(plotter, *args, **kwargs)
    return show


def install() -> None:
    """Put the mocks in place (idempotent) and switch the result cache off."""
    if _saved:
        return
    from vegeta import aeromant, talos
    from vegeta.aeromant import movie as amovie
    from vegeta.aeromant import viz as aviz
    from vegeta.talos import viz as tviz

    _patch(talos.StructuralModel, "mesh", _talos_mesh)
    _patch(talos.StructuralModel, "solve", _talos_solve)
    _patch(talos.StructuralModel, "solve_modes", _talos_modes)
    for owner in (talos, talos.fatigue if hasattr(talos, "fatigue") else None):
        if owner is not None and hasattr(owner, "assess_fatigue"):
            _patch(owner, "assess_fatigue", _assess_fatigue)
    for owner in (talos, talos.frd):
        _patch(owner, "read_frd", frd_fields)
    from vegeta.talos import mesh as tmesh
    from vegeta.talos import model as tmodel
    for owner in (tmesh, tviz, tmodel):
        _patch(owner, "read_mesh", mesh_data)
    _patch(tviz, "animate", _animate)
    for name in ("plot_results", "plot_mode", "plot_damage", "plot_section", "export_vtu", "plot_deformed",
                 "plot_along_axis", "plot_von_mises_histogram", "plot_mesh", "plot_problem", "mesh_to_pyvista",
                 "results_to_pyvista"):
        for owner in (tviz, talos):
            if hasattr(owner, name):
                _patch(owner, name, placeholder)
    _patch(tviz, "show", _real_or_placeholder_show(tviz.show))

    real_detect = aeromant.OpenFOAMEnvironment.detect

    def detect(*args, **kwargs):                        # no OpenFOAM here: a default environment (the runs are mocked)
        try:
            return real_detect(*args, **kwargs)
        except RuntimeError:
            return aeromant.OpenFOAMEnvironment()
    _patch(aeromant.OpenFOAMEnvironment, "detect", staticmethod(detect))
    _patch(aeromant.CFDCase, "run", _cfd_run)
    _patch(aeromant.CFDCase, "results", _cfd_results)
    for owner in (aeromant, aeromant.case):
        if hasattr(owner, "read_case_results"):
            _patch(owner, "read_case_results", _read_case_results)
    _patch(amovie, "openfoam_sampler", _openfoam_sampler)
    _patch(amovie, "make_movie", _make_movie)
    _patch(amovie, "concat_videos", _concat_videos)
    _patch(aviz, "read_results", _read_results)
    _patch(aviz, "animate_particles", _animate)
    for name in ("plot_field_slice", "plot_mesh_slice", "plot_section", "plot_streamlines", "plot_surface_pressure"):
        if hasattr(aviz, name):
            _patch(aviz, name, placeholder)
    _patch(aviz, "show", _real_or_placeholder_show(aviz.show))
    for name in ("plot_coefficients", "plot_residuals"):
        if hasattr(aeromant, name):
            _patch(aeromant, name, placeholder)

    from vegeta import mellonia
    from vegeta.mellonia import slicer as mslicer
    for owner in (mellonia, mslicer):
        _patch(owner, "slice_stl", _slice_stl)

    os.environ["VEGETA_CACHE"] = "off"
    os.environ[ENV] = "1"
    print("vegeta.mock: DEBUG run - FEA, CFD and slicing are mocked (fixed numbers), the result cache is off")


def uninstall() -> None:
    """Put the real functions back (the cache switch is left as it is)."""
    while _saved:
        owner, name, orig = _saved.pop()
        setattr(owner, name, orig)
