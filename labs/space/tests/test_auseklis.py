"""Auseklis: physics against hand calculations, budgets that close, capture momentum, CAD that builds, and the prepared
heavy-tool scripts under the mocks. Fast: no real solver runs here."""
import importlib.util
import math
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from auseklis import capture as C, mission as M, physics as ph, provenance as P, rendezvous as R, trade as T, variants as V

EXT = Path(__file__).resolve().parents[1] / "exp" / "auseklis" / "external"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, EXT / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- physics ------------------------------------------------------------------------------------------------------
def test_circular_speed_and_period_at_550km():
    r = ph.radius(550)
    assert ph.v_circ(r) == pytest.approx(7585.1, abs=1.0)
    assert ph.period(r) / 60 == pytest.approx(95.65, abs=0.05)


def test_sun_synchronous_inclination_matches_requirement():
    assert ph.sso_inclination_deg(550) == pytest.approx(97.6, abs=0.05)


def test_hohmann_leo_to_geo_textbook():
    h = ph.hohmann(6578e3, 42164e3)                       # 200 km LEO to GEO, textbook: 2.457 + 1.478 = 3.935 km/s
    assert h["dv"] == pytest.approx(3935, rel=0.005)


def test_rocket_equation_round_trip_and_burn_time():
    mp = ph.propellant_mass(400, 500, 290)
    assert ph.final_mass(400, 500, 290) == pytest.approx(400 - mp)
    assert ph.burn_time(400, 500, 290, 44) == pytest.approx(mp * 290 * ph.G0 / 44)   # constant thrust: t = mp ve / F


def test_hall_thrust():
    assert ph.hall_thrust(800, 1500, 0.5) == pytest.approx(0.0544, abs=1e-3)


def test_eclipse_fraction():
    assert ph.eclipse_fraction(550, 0) == pytest.approx(0.3723, abs=1e-3)
    assert ph.eclipse_fraction(550, 70) == 0.0
    assert ph.eclipse_fraction(550, 20) < ph.eclipse_fraction(550, 0)


def test_phasing_cost_falls_with_time_and_refuses_inside_atmosphere():
    r = ph.radius(550)
    assert ph.phasing_dv(r, 1.0, 30)["dv"] < ph.phasing_dv(r, 1.0, 10)["dv"]
    with pytest.raises(ValueError):
        ph.phasing_dv(r, 3.0, 1)


def test_cw_equilibrium_and_bounded_motion():
    n = ph.mean_motion(ph.radius(550))
    assert np.allclose(ph.cw_propagate([0, -500, 0, 0, 0, 0], n, 3000)[:3], [0, -500, 0])
    x0 = [100.0, 0, 0, 0, -2 * n * 100.0, 0]               # no-drift condition: a closed 2:1 ellipse
    ys = [ph.cw_propagate(x0, n, t)[1] for t in np.linspace(0, 5 * 5740, 50)]
    assert max(abs(np.array(ys))) < 250


def test_density_is_monotonic_and_drag_scales():
    assert ph.density(450) > ph.density(550) > ph.density(650)
    assert ph.density(550, "max") > ph.density(550, "mod") > ph.density(550, "min")
    assert ph.drag_dv_per_year(550, 0.02) == pytest.approx(2 * ph.drag_dv_per_year(550, 0.01))
    assert ph.lifetime_days(450, 0.02) < ph.lifetime_days(550, 0.02)


# --- provenance ---------------------------------------------------------------------------------------------------
def test_provenance_tags():
    x = P.assumed(3.0, "kg", "test", "t_x")
    assert x * 2 == 6.0 and x.tag == P.ASSUMED
    with pytest.raises(ValueError):
        P.verified(1.0, "m", "")
    assert any(r[0] == "t_x" for r in P.register_table(P.ASSUMED))


# --- budgets and mission -------------------------------------------------------------------------------------------
@pytest.mark.parametrize("key", "ABC")
def test_budget_closes(key):
    v = V.VARIANTS[key]
    for kw in ({}, {"wet_target": 400.0}):
        b = V.budget(v, **kw)
        assert b["wet"] == pytest.approx(b["dry_with_margin"] + b["tanks"] + b["propellant"] + b["rcs_propellant"])
        assert b["dv_capability"] >= 500 - 1e-6
    assert V.budget(v, wet_target=400.0)["wet"] == pytest.approx(400.0)
    assert V.budget(v)["dv_capability"] == pytest.approx(500.0)


def test_array_sizing_for_1500w():
    a = V.array_sizing()
    assert 5.0 < a["area"] < 8.0
    assert a["eclipse_min"] == pytest.approx(35.6, abs=0.2)


def test_mission_feasibility_flags():
    assert not M.removal(V.VARIANTS["B"], M.DEBRIS[250])["feasible"]                    # docking-only: no uncooperative targets
    assert not M.removal(V.VARIANTS["C"], M.DEBRIS[250], "reentry")["feasible"]         # low thrust cannot do an impulsive reentry
    r = M.removal(V.VARIANTS["A"], M.DEBRIS[250])
    assert r["feasible"] and r["propellant"] > 0 and r["dv_total"] == pytest.approx(sum(l["dv"] for l in r["legs"]))


def test_removals_fall_when_solar_activity_is_low():
    a = V.VARIANTS["A"]
    assert M.campaign(a, M.DEBRIS[250], activity="min")["removals"] < M.campaign(a, M.DEBRIS[250], activity="max")["removals"]


def test_reentry_costs_more_than_decay():
    a = V.VARIANTS["A"]
    assert M.removal(a, M.DEBRIS[500], "reentry")["dv_total"] > M.removal(a, M.DEBRIS[500])["dv_total"]


def test_trade_matrix():
    t = T.matrix()
    assert sum(T.WEIGHTS.values()) == pytest.approx(1.0)
    assert t["winner"] in "ABC" and set(t["total"]) == set("ABC")
    assert sum(T.weight_sensitivity(n=200).values()) == pytest.approx(1.0)


# --- rendezvous and capture ----------------------------------------------------------------------------------------
def test_monitor_and_pc():
    m = R.SafetyMonitor()
    assert m.check(50, 0.3) == "abort" and m.check(50, 0.05, 0.1) == "ok" and m.check(50, 0.05, 12.0) == "abort"
    assert R.pc_2d(0, 5, 3) > R.pc_2d(10, 5, 3) > R.pc_2d(30, 5, 3)
    assert 0 <= R.pc_2d(0, 0.5, 3) <= 1.0 + 1e-9
    assert R.pc_2d(0, 5, 3) == pytest.approx(1 - math.exp(-9 / 50), rel=0.02)       # centred Gaussian, small disc: closed form


def test_gate_table_has_four_gates():
    g = R.gate_table()
    assert [x["range_m"] for x in g] == [10000, 1000, 100, 10]


def test_capture_conserves_angular_momentum():
    Is, Id = C.box_inertia(400, 1, 1, 1.2), C.cylinder_inertia(500, .8, 3.6)
    w, H, m, I, com = C.capture_rate(400, Is, 500, Id, np.radians([0, 3, 0]), [2.5, 0, 0])
    assert np.allclose(I @ w, H)
    assert m == 900 and com[0] == pytest.approx(500 * 2.5 / 900)
    w0, H0, *_ = C.capture_rate(400, Is, 500, Id, [0, 0, 0], [2.5, 0, 0])
    assert np.allclose(w0, 0) and np.allclose(H0, 0)


def test_contact_peak_force_below_undamped_bound():
    r = C.contact_1d(400, 500, 0.05, 2000.0, 300.0)
    assert 0 < r["peak_force_N"] <= r["analytic_undamped_N"] * 1.001
    assert C.impulsive_capture(400, 500, 0.05)["impulse_Ns"] == pytest.approx(400 * 500 / 900 * 0.05)


# --- CAD, render, STEP ---------------------------------------------------------------------------------------------
cq = pytest.importorskip("cadquery")


@pytest.mark.parametrize("key", "ABC")
def test_cad_builds_and_closes_with_budget(key):
    from auseklis import cad
    d = cad.Auseklis()
    p = d.resolve(variant=key)
    parts = d.parts(p)
    assert "bus_shell" in parts and any(n.startswith("tank") for n in parts) and any(n.startswith(("capture", "docking")) for n in parts)
    bus = parts["bus_shell"].BoundingBox()
    for n, s in parts.items():
        if n.startswith("tank_f") or n.startswith("tank_o") or n == "tank_main":        # tanks stay inside the bus
            bb = s.BoundingBox()
            assert bb.xmin > bus.xmin and bb.xmax < bus.xmax and bb.ymin > bus.ymin and bb.ymax < bus.ymax, n
    mp = cad.mass_properties(variant=key)
    assert mp["mass_kg"] == pytest.approx(mp["budget_wet_kg"], rel=1e-6)
    assert mp["array_area_m2"] >= V.array_sizing()["area"]
    I = mp["inertia_kgm2"]
    assert np.all(np.linalg.eigvalsh(I) > 0)


def test_oversized_tank_is_refused():
    from auseklis import cad
    with pytest.raises(ValueError):
        cad.Auseklis().parts(cad.Auseklis().resolve(variant="B", bus_x=500.0, bus_y=450.0))


def test_step_export_and_render(tmp_path):
    from auseklis import cad, render
    f = cad.export_step(str(tmp_path / "a.step"), variant="C")
    assert Path(f).stat().st_size > 10_000
    png = render.render("C", str(tmp_path / "c.png"), tol=12.0, dpi=40)
    assert Path(png).stat().st_size > 5_000


# --- prepared heavy-tool scripts (mocks only) -----------------------------------------------------------------------
def test_mujoco_scenario_is_written_not_run(tmp_path):
    mod = _load("mujoco_capture")
    xml = mod.build_mjcf("A", 500)
    root = ET.fromstring(xml)
    assert root.find(".//weld").get("active") == "false" and len(root.findall(".//freejoint")) == 2
    assert mod.main(["--out", str(tmp_path / "c.xml")]) == 0 and (tmp_path / "c.xml").exists()
    if os.environ.get("VEGETA_DEBUG") or os.environ.get("VEGETA_NO_SOLVERS"):
        with pytest.raises(RuntimeError):
            mod.run(xml, 0.05, 3.0)                       # the real run is refused in mock mode


def test_bus_fea_plan_and_mocked_run(tmp_path):
    pytest.importorskip("vegeta.talos")
    mod = _load("bus_fea")
    assert mod.main(["--workdir", str(tmp_path / "plan")]) == 0 and (tmp_path / "plan" / "plan.json").exists()
    if not os.environ.get("VEGETA_DEBUG"):
        pytest.skip("the run needs the mocks (VEGETA_DEBUG=1) or real Gmsh + CalculiX")
    assert mod.main(["--workdir", str(tmp_path / "run"), "--run"]) == 0 and (tmp_path / "run" / "results.json").exists()


def test_report_builds():
    from auseklis import report
    text = report.build()
    assert "Mass and Δv budgets" in text and "Assumption register" in text
