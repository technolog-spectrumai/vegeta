"""prop_lab (notebook 41): the 9x6 alone — its configurations, the table beside BEMT, the calibration fit on a
synthetic campaign, the export Kestrel reads (the blade section, the drive key, the normal-force factor), and a
failure as a result.

Run: cd /home/user/vegeta && python3 -m pytest -q notebooks/designs/tests/test_prop_lab.py
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import kestrel_systems as ks
import prop_lab as pl


class _R:
    def __init__(self, ok, metrics=None, messages=()):
        self.ok, self.metrics, self.messages = ok, metrics or {}, list(messages)


def _synthetic(specs, cl_scale=1.1, cd_scale=0.8, cn_scale=0.8, fail=("axial, 24 m/s, 8000 rpm",), diverge=("oblique, 8 m/s at 90°, 8000 rpm",)):
    """A campaign whose 'CFD' is BEMT on a blade section with the lift slope x1.1 and the profile drag x0.8 (what the
    fit must recover), the normal force 0.8 x De Young; one case failed, one diverged."""
    from vegeta import boreas
    base = ks.blade_airfoil(calibrated=False)
    af = boreas.Airfoil(cl_alpha=base.cl_alpha * cl_scale, alpha0_deg=base.alpha0_deg, cl_max=base.cl_max, cd0=base.cd0 * cd_scale, k=base.k)
    prop = ks.propulsion().prop
    res = {}
    for name, s in specs.items():
        if name in fail:
            res[name] = _R(False, messages=["step solver failed: simpleFoam exited with 1"]); continue
        a = math.radians(s["inflow_angle"])
        op = pl.bemt(s["rpm"], s["airspeed"] * math.cos(a))
        cfd = boreas.bemt.solve(prop, af, s["rpm"], s["airspeed"] * math.cos(a), pl.RHO)
        T, Q = cfd.thrust, cfd.torque
        if name in diverge:
            T, Q = 3e94, -1e94
        m = {"thrust_N": T, "torque_Nm": Q, "power_W": Q * s["rpm"] * 2 * math.pi / 60, "efficiency": None, "figure_of_merit": 0.6, "iterations": 400,
             "converged": True, "mesh_cells": 300000, "thrust_std_lastN": 0.01}
        if s["kind"] == "oblique":
            q = 0.5 * pl.RHO * s["airspeed"] ** 2
            A = math.pi * (ks.propulsion().prop.diameter / 2) ** 2
            m["normal_force_N"] = cn_scale * pl.deyoung_cn_alpha(op.thrust, s["airspeed"]) * q * A * math.sin(a)
            m["pitching_moment_Nm"] = 0.01
        res[name] = _R(True, m)
    return res


def test_configurations_and_the_bemt_baseline():
    c = pl.configurations()
    kinds = {k: sum(1 for v in c.values() if v["kind"] == k) for k in ("hover", "axial", "oblique")}
    assert kinds == {"hover": 3, "axial": 9, "oblique": 9} and all(v["template"].startswith("rotor_mrf") for v in c.values())
    t = pl.table(None, c)
    assert (t["status"] == "NOT RUN").all() and len(t) == 21
    assert float(t.loc["hover, 5000 rpm", "BEMT T [N]"]) > 1.0 and 0 < float(t.loc["axial, 16 m/s, 8000 rpm", "J"]) < 1.0
    assert np.isfinite(float(t.loc["oblique, 8 m/s at 30°, 8000 rpm", "De Young CNα [1/rad]"]))
    assert pl.summary(t)["NOT RUN"] == 21 and 5500 < pl.rpm_hover() < 8500


def test_cad_geometry_matches_the_components():
    g = pl.geometry()
    assert g["diameter"] == pytest.approx(228.6) and g["pitch"] == pytest.approx(152.4) and g["blades"] == 2 and g["chord_max"] == pytest.approx(21.0)


def test_calibration_fit_export_and_what_kestrel_reads(tmp_path, monkeypatch):
    specs = pl.configurations()
    T = pl.table(_synthetic(specs), specs)
    sm = pl.summary(T)
    assert sm["FAILED"] == 1 and sm["diverged"] == 1 and sm["solved"] == 19 and sm["by kind"]["oblique"] == 8
    assert T.loc["axial, 24 m/s, 8000 rpm", "note"].startswith("step solver failed")
    sec = pl.fit_section(T)
    assert sec["n_cases"] == 11 and sec["rms_log_ratio_after"] < 0.1 * sec["rms_log_ratio_before"]
    assert sec["cl_scale"] == pytest.approx(1.1, abs=0.02) and sec["cd_scale"] == pytest.approx(0.8, abs=0.05)
    nf = pl.fit_normal_force(T)
    assert nf["n_cases"] == 8 and nf["cn_factor"] == pytest.approx(0.8, rel=1e-6)
    path = tmp_path / "prop_lab.json"
    doc = pl.export(T, path, meta={"quality": "screening"})
    assert doc["calibration"]["cn_factor"] == pytest.approx(0.8, rel=1e-6) and len(doc["cases"]) == 21
    cal = pl.calibration(path)
    assert cal["on_file"] and cal["cl_scale"] == pytest.approx(sec["cl_scale"]) and cal["n_solved"] == 19
    # a NOT RUN pass keeps the calibration
    doc2 = pl.export(pl.table(None, specs), path)
    assert doc2["calibration"]["cl_scale"] == pytest.approx(sec["cl_scale"]) and doc2["last_listing"]["summary"]["NOT RUN"] == 21
    assert not pl.calibration(tmp_path / "none.json")["on_file"]
    # Kestrel reads it: the blade section, the drive key, the normal force
    monkeypatch.setattr(ks, "PROP_LAB_JSON", path)
    af0, af1 = ks.blade_airfoil(calibrated=False), ks.blade_airfoil()
    assert af1.cl_alpha == pytest.approx(af0.cl_alpha * sec["cl_scale"]) and af1.cd0 == pytest.approx(af0.cd0 * sec["cd_scale"])
    assert ks._drive_key()["prop_lab"] == {"cl_scale": round(sec["cl_scale"], 4), "cd_scale": round(sec["cd_scale"], 4)}
    import kestrel_flight as kf
    assert kf.prop_cn_alpha(1.0, 10.0) == pytest.approx(0.8 * kf.prop_cn_alpha(1.0, 10.0, calibrated=False))
    assert ks.propulsion().airfoil.cl_alpha == pytest.approx(af1.cl_alpha)
    monkeypatch.setattr(ks, "PROP_LAB_JSON", tmp_path / "none.json")
    assert ks._drive_key()["prop_lab"] is None and kf.prop_cn_alpha(1.0, 10.0) == pytest.approx(kf.prop_cn_alpha(1.0, 10.0, calibrated=False))


def test_flight_points_cruise_and_stall():
    import kestrel_cfd as kcfd
    import kestrel_transition as kt
    veh = kt.make_vehicle()
    pts = kcfd.flight_points(veh)
    assert [p["source"] for p in pts] == ["cruise", "stall"] and pts[0]["V"] == 16.0 and 10.5 < pts[1]["V"] < 12.5
    assert 0 < pts[0]["alpha_deg"] < 8 and pts[1]["alpha_deg"] > pts[0]["alpha_deg"] and pts[0]["T_up"] > veh.drive.T_min
    spec = kcfd.specs(pts, sweep=False)
    assert len(spec) == 2 and all(s["props"] for s in spec.values())
