"""K4TQ Stage 1: the layout and its transforms, the mass budget, the physics (the hover's momentum limit, the lattice,
the stall rule, the drive's current limit), the corridor's balance, the loads' consistency, the first CAD. Fast, on
the mocks (``python tools/lab.py test aero -- -k k4tq``)."""
import math

import numpy as np
import pytest

import k4tq
import k4tq_sizing as s


@pytest.fixture(scope="module")
def VEH():
    return s.vehicle(k4tq.MAIN)


def test_requirements_marking_and_logging():
    assert k4tq.STANDARD_MARKING == "BE CAREFUL – FRAGILE – TEMPERATURE SENSITIVE LOAD"
    assert "no red cross" in k4tq.REQUIREMENTS["R8 marking"]
    assert "mandatory" in k4tq.REQUIREMENTS["R7 temperature logging"]
    cargo = k4tq.CATALOGUE["cargo"]
    assert "temperature logger" in cargo and "temperature sensors (2)" in cargo          # two channels + an independent logger
    assert k4tq.params()["fuselage_length"] == 1.2 and k4tq.MAIN == "B"                 # the approved baseline


def test_layout_clearances_and_poses():
    for v in k4tq.VARIANTS:
        L = k4tq.layout(k4tq.params(v))
        assert k4tq.checks(L) == [], (v, k4tq.checks(L))
    L = k4tq.layout(k4tq.params())
    for (which, side), pose in L["poses"].items():
        cr, hv = pose["cruise"], pose["hover"]
        assert np.allclose(cr["axis"], [-1, 0, 0]) and np.allclose(hv["axis"], [0, 0, 1], atol=1e-12)    # forward / up
        assert abs(cr["hub"][1] - hv["hub"][1]) < 1e-12 and abs(abs(hv["hub"][1]) - L["p"]["motor_y"]) < 1e-12   # tips stay laterally out
    assert L["clearance"]["hover disc separation along x [m]"] >= 0.10                   # the selection's rotor-interference rule


def test_tilt_is_a_rotation_about_the_spar():
    L = k4tq.layout(k4tq.params())
    w = L["wings"]["front"]
    for tilt in (0.0, 0.4, math.pi / 2):
        spar = k4tq.wing_point(L, "front", 1, 0.5, w["pivot_chord"], tilt)
        assert np.allclose(spar, [w["x_pivot"], 0.5, w["z"]])                          # the spar does not move
        le, te = (k4tq.wing_point(L, "front", 1, 0.5, xi, tilt) for xi in (0.0, 1.0))
        assert abs(np.linalg.norm(te - le) - w["c_at"](0.5)) < 1e-12                    # rigid
    le90 = k4tq.wing_point(L, "front", 1, 0.5, 0.0, math.pi / 2)
    assert le90[2] > w["z"] and abs(le90[0] - w["x_pivot"]) < 1e-12                     # hover: leading edge up
    tip_f = k4tq.wing_point(L, "front", 1, w["y1"], w["pivot_chord"], math.pi / 2, math.pi / 2)
    assert tip_f[0] > w["x_pivot"] + 0.5 and abs(tip_f[1] - w["y_hinge"]) < 0.02         # folded at tilt 90°: swung aft along the side
    f = k4tq.wing_frame(L, "front", 1, math.pi / 2, math.pi / 2)
    inner = f([[0.0, w["y_hinge"] - 0.03, 0.0]])[0]
    assert abs(inner[1] - (w["y_hinge"] - 0.03)) < 1e-12                                 # inboard of the hinge does not fold


def test_mass_budget_and_cg(VEH):
    items = VEH["items"]
    assert abs(sum(i["mass_kg"] for i in items) - VEH["mass"]["mass_kg"]) < 1e-9
    assert not any(i["clipped"] for i in items if i["group"] == "battery")
    assert abs(VEH["mass"]["x_cg"] - VEH["x_cg_target"]) < 1e-3
    assert 4.5 < VEH["mass"]["mass_kg"] < 7.5 and 0.9 < k4tq.tilt_wing_tax(items) < 2.5          # Stage 2: the CAD masses
    assert abs(sum(1 for i in items if i["group"] == "roots") - 4) == 0


def test_lattice_and_stall_rule(VEH):
    a, st = VEH["aero"], VEH["stall"]
    assert 4.0 < a["CLa"] < 6.5 and 0.5 < a["slope_rear"] < 0.9 < a["slope_front"]    # the rear wing in the downwash
    assert a["x_np"] > VEH["mass"]["x_cg"] and abs(st["static margin [MAC]"] - s.STATIC_MARGIN) < 1e-3
    assert st["stalls first"] == "front" and st["nose-down stall"]


def test_hover_momentum_limit_and_drive(VEH):
    h = s.hover(VEH)
    assert h["shaft power [W]"] > h["ideal power [W]"] and 0.4 < h["figure of merit"] < 0.8
    assert h["electrical [W]"] > h["shaft power [W]"]
    assert abs(h["thrust front each [N]"] * 2 + h["thrust rear each [N]"] * 2 - VEH["W"] * s.DOWNLOAD) < 1e-6
    m = VEH["drive"]
    i_max = k4tq.CATALOGUE["motor"][k4tq.params()["motor"]]["I_max_A"]
    t = s.max_thrust(m, 0.0)
    assert m.at(0.0, m.throttle_for_thrust(0.0, t))["current"] <= i_max * 1.02            # the current limit holds
    assert h["most loaded motor / its maximum"] <= 0.75


def test_cruise_power_is_drag_times_speed(VEH):
    c = s.cruise(VEH, n_running=2)
    ok = c[c["feasible"]]
    assert len(ok) > 3
    r = ok.iloc[0]
    assert 0.3 < r["propulsive eff."] < 0.8                                               # D·V / P_electrical
    assert ok.index.min() >= s.CRUISE_MIN_OVER_STALL * s.stall_speed(VEH) - 1.0


def test_corridor_balances(VEH):
    rows = {V: s.corridor_point(VEH, V) for V in (0.0, 8.0, 22.0)}
    for V, r in rows.items():
        assert r["feasible"], r
        assert abs(r["trim residual [N·m]"]) < 0.05 and abs(r.get("X residual [N]", 0.0)) < 0.2
    assert 80.0 < rows[0.0]["tilt [deg]"] < 100.0                    # the hover: about vertical (trimmed against the cambered wings' pull)
    assert rows[22.0]["tilt [deg]"] < 10.0                                               # wing-borne


def test_root_loads_consistency(VEH):
    r = s.root_loads(VEH)
    assert abs(r["root moment ultimate [N·m]"] - s.ULTIMATE * max(r["root moment, cruise limit [N·m]"], r["root moment, hover motor-max limit [N·m]"])) < 1e-9
    assert r["actuator reserve"] >= 1.0 and r["bearing static reserve"] >= 2.0
    assert r["spar tubes"].loc[r["spar chosen"], "reserve (ultimate)"] >= 1.5
    assert r["divergence speed of the all-moving wing [m/s]"] > 2 * 32.0


def test_directional_needs_a_fin(VEH):
    d = s.directional(VEH["L"], VEH, VEH["aero"], fin_area=VEH["p"]["fin_area"])
    assert d["Cnβ without a fin"] < 0 and d["Cnβ with the fin"] >= 0.049


def test_quad_reference_flies_worse_on_range(VEH):
    q = s.quad_reference("Q-best", "17x5.8", k4=VEH)
    assert q["mass [kg]"] < VEH["mass"]["mass_kg"]                                       # no wings, no roots
    assert q["range [km]"] < s.best_speeds(s.cruise(VEH, n_running=2))["range [km]"]


def test_first_cad_builds_in_three_configurations():
    pytest.importorskip("cadquery")
    import k4tq_cad
    p = k4tq.params()
    for tilt, fold in ((0.0, 0.0), (90.0, 0.0), (90.0, p["fold_transport_deg"])):
        parts = k4tq_cad.assembly(p, tilt, fold)
        assert sum(1 for k in parts if k.startswith("wing panel ")) == 4
        assert all(v.solid.isValid() and v.solid.Volume() > 0 for v in parts.values()), (tilt, fold)
