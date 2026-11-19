"""GAVILAN (notebook 38): the CAD's variants, the mass budgets, the reduced flight model against the lab's own
models, the manoeuvres' orderings, the emitter's bookkeeping, the CFD launcher's plumbing.

Run: .venv/bin/python -m pytest -q labs/aero/tests/test_gavilan.py   (about a minute; the CAD tests need CadQuery)
"""
import math
import types

import numpy as np
import pytest

import gavilan as gv
import gavilan_flight as gf


@pytest.fixture(scope="module")
def fleet():
    return gf.fleet()


# --------------------------------------------------------------------------------------------------------------- CAD
@pytest.mark.parametrize("key", list(gv.VARIANTS))
def test_tailsitter_variants_build_as_one_valid_solid_with_clear_propellers(key):
    p = gv.resolve(gv.variant(key))
    g = gv.Gavilan().generate(**gv.variant(key))
    assert g.shape.isValid() and len(g.shape.Solids()) == 1
    L = gv.Gavilan.layout(p)
    assert min(L["tip_to_body"], L["tip_to_wing"], L["tip_to_tip"]) > 10.0          # mm
    assert L["width"] <= 700.0 + 1e-6                                              # a 4 m gap with room to spare
    assert g.dimensions[0] == pytest.approx(p["length"] + 5.0, abs=6.0)            # the lens stands ahead of the nose
    for part in ("wing", "pylon", "nose", "tail"):
        q = gv.Gavilan().generate(**gv.variant(key, part=part))
        assert q.shape.isValid() and len(q.shape.Solids()) == 1, part


def test_camera_tilts_and_propeller_discs_are_optional():
    a = gv.Gavilan().generate(camera_tilt_deg=90.0)
    assert a.shape.isValid() and a.dimensions[0] < gv.Gavilan().generate().dimensions[0]    # the lens no longer leads
    assert gv.Gavilan().generate(show_props=True).volume > gv.Gavilan().generate().volume
    with pytest.raises(Exception):
        gv.Gavilan().generate(motor_z=80.0)                                        # the tips would reach the wing


def test_fixed_wing_is_fixed_wings_layout_with_the_emitter_on_the_axis():
    g = gv.GavilanF().generate()
    q = gv.f_resolve()
    assert g.shape.isValid() and len(g.shape.Solids()) == 1
    assert g.dimensions[0] == pytest.approx(q["fuselage_length"] + q["muzzle_out"] + 6.0, abs=1.0)
    assert all(q[k] == v for k, v in gf.F_PARAMS.items())                          # the flight model reads the same numbers
    assert gv.GavilanF().generate(part="emitter").shape.isValid()


def test_planform_and_outline_match_the_layout():
    p = gv.resolve()
    L = gv.Gavilan.layout(p)
    pf = gv.planform(p)
    area = sum(0.5 * abs(np.cross(q[1] - q[0], q[2] - q[0])[2]) + 0.5 * abs(np.cross(q[2] - q[0], q[3] - q[0])[2]) for q in pf["wing"])
    assert area == pytest.approx(L["S_mm2"] / 1e6, rel=1e-6)
    o = gv.outline(p)
    assert o["span"] == pytest.approx(0.70) and len(o["front"]["props"]) == 4


# -------------------------------------------------------------------------------------------------------------- masses
def test_mass_budgets_add_up_and_the_emitter_is_what_the_table_says():
    assert gf.emitter_kg() == pytest.approx(sum(gf.EMITTER["parts_g"].values()) * gf.EMITTER["margin"] / 1e3)
    armed, clean = gf.mass_budget(emitters=1), gf.mass_budget(emitters=0)
    assert armed.mass - clean.mass == pytest.approx(gf.emitter_kg() + gf.CRADLE_G / 1e3)
    assert armed.mass == pytest.approx(sum(armed.by_group().values()))
    assert all(i.source for i in armed.items)                                      # every line says where it comes from
    assert 1.4 < clean.mass < 2.0 and 1.5 < gf.f_mass_budget().mass < 2.1
    cg = armed.properties().com[0]
    assert abs(cg - gf.geometry()["x_ac"]) < 0.08                                  # near the wing's aerodynamic centre


def test_drag_buildup_carries_emerillons_margin():
    d = gf.drag_buildup()
    assert d["CdA"] == pytest.approx(d["clean"] * gf.ef.DRAG_MARGIN) and d["emitter"] > 0
    assert gf.drag_buildup(emitters=0)["CdA"] < d["CdA"]


# ---------------------------------------------------------------------------------------------------------- the model
def test_wing_lift_slope_sits_between_slender_and_two_dimensional():
    a = gf.wing_aero()
    assert 0.5 * math.pi * a["AR"] < a["CLa"] * 1.6 and a["CLa"] < 2 * math.pi
    assert 0.8 < a["e"] < 1.3                                                      # the pylons act as small end plates


def test_fixed_wings_reproduce_their_own_turn_model(fleet):
    import peregrine_flight as pf
    m = fleet["peregrine"]
    d = pf.make_aircraft("B", {"variant": "B"}, m.mass)
    own = pf.turn(d.af(0.0), pf.drive(), np.array([25.0]), n_struct=m.n_struct)
    mine = gf.turn(m, 25.0)
    assert mine["n_inst"] == pytest.approx(float(own["n_inst"][0]), rel=0.05)
    assert mine["n_sus"] == pytest.approx(float(own["n_sus"][0]), rel=0.12)


def test_emerillon_keeps_its_own_thrust_and_mass(fleet):
    m = fleet["emerillon"]
    assert m.mass == pytest.approx(gf.ef.mass_budget(gf.ef.ER2).mass)
    assert m.t_max(0.0) == pytest.approx(gf.ef.drive(gf.ef.ER2, 0.5)["thrust"], rel=0.02)
    assert fleet["emerillon_armed"].mass - m.mass == pytest.approx(gf.emitter_kg())


def test_what_hovers_and_what_does_not(fleet):
    assert {k for k, m in fleet.items() if m.hovers} == {"xwing", "emerillon", "emerillon_armed", "gavilan_a_unarmed", "gavilan_a", "gavilan_b"}
    for k in ("merlin", "peregrine", "gavilan_f"):
        assert gf.min_speed(fleet[k]) > 10.0 and not gf.brake(fleet[k])["stops"]
    assert gf.brake(fleet["gavilan_a"])["distance_m"] < 25.0


def test_a_wing_is_not_allowed_beyond_its_stall_but_a_tailsitter_is(fleet):
    assert gf.turn(fleet["gavilan_f"], 10.0)["n_inst"] < 1.0                       # below its stall: no level turn
    t = gf.turn(fleet["gavilan_a"], 10.0)
    assert t["n_inst"] > 3.5 and t["theta_inst_deg"] > 60.0                        # it turns on its thrust, the axis across the path


def test_turn_orderings_the_report_states(fleet):
    a, f, x = fleet["gavilan_a"], fleet["gavilan_f"], fleet["xwing"]
    for V in (10.0, 15.0, 25.0):
        assert gf.turn(a, V)["rate_inst_deg_s"] > gf.turn(x, V)["rate_inst_deg_s"] > gf.turn(f, V)["rate_inst_deg_s"]
    assert gf.turn(fleet["gavilan_a_unarmed"], 15.0)["n_inst"] > gf.turn(fleet["emerillon"], 15.0)["n_inst"] > gf.turn(a, 15.0)["n_inst"]
    assert gf.turn(fleet["peregrine"], 25.0)["n_inst"] > gf.turn(a, 25.0)["n_inst"]          # the light wing wins at speed
    assert gf.turn(fleet["gavilan_f_broad"], 15.0)["radius_inst_m"] < gf.turn(f, 15.0)["radius_inst_m"]


def test_level_flight_goes_from_thrust_borne_to_wing_borne(fleet):
    a = fleet["gavilan_a"]
    slow, fast = gf.level(a, 8.0), gf.level(a, 24.0)
    assert slow["theta_deg"] > 50.0 and slow["wing_share"] < 0.4
    assert fast["theta_deg"] < 12.0 and fast["wing_share"] > 0.9 and fast["power_w"] < slow["power_w"]
    assert gf.level(fleet["gavilan_f"], 8.0) is None


def test_gun_pass_a_cradle_holds_a_fixed_gun_passes(fleet):
    assert gf.gun_pass(fleet["gavilan_a"])["holds"]
    p = gf.gun_pass(fleet["gavilan_f"])
    assert not p["holds"] and p["window_s"] == pytest.approx(25.0 / p["v_pass"]) and p["reattack_s"] > 5.0
    assert gf.gun_pass(fleet["gavilan_f"], v_bandit=20.0)["holds"]                 # it can match a bandit at 20 m/s


def test_scorecard_has_a_row_per_aircraft(fleet):
    rows = gf.scorecard({k: fleet[k] for k in ("peregrine", "gavilan_a", "gavilan_f")})
    assert [r["key"] for r in rows] == ["peregrine", "gavilan_a", "gavilan_f"]
    assert all(math.isfinite(r["top speed [m/s]"]) and r["source"] for r in rows)


# ----------------------------------------------------------------------------------------------------------------- CFD
def test_cfd_references_and_rows(tmp_path):
    import gavilan_cfd as gc
    for d in ("A", "F"):
        ref = gc.reference(d)
        assert ref["S"] > 0.1 and 0.15 < ref["c"] < 0.25 and 1.5 < ref["mass"] < 2.6
    assert gc.to_flow(100.0, 0.0, 90.0) == pytest.approx([0.0, 0.0, -0.1], abs=1e-9)
    assert gc.pitch_authority("A") == pytest.approx(0.5 * gf.gavilan("A").t_max(0.0) * 0.12)
    assert gc.row("A", 30.0, 20.0, None)["status"] == "NOT RUN"
    fake = types.SimpleNamespace(ok=True, messages=[], metrics={"Cl": 0.6, "Cd": 0.4, "Cm": -0.1, "lift_force_N": 19.5, "drag_force_N": 13.0})
    r = gc.row("A", 30.0, 20.0, fake)
    assert r["status"] == "solved" and r["pitch_moment_Nm"] < 0 and 0 < r["moment_over_authority"] < 1
    assert gc.export_stl(tmp_path, "A", 30.0).stat().st_size > 10000


# ---------------------------------------------------------------------------------------------------------------- KPIs
def test_kpis_unarmed_cannot_is_none_and_scores_are_scaled_to_the_best():
    rows = gf.kpi_table({k: m for k, m in gf.fleet_unarmed().items() if k in ("peregrine", "emerillon", "gavilan_a", "gavilan_f")})
    by = {r["key"]: r for r in rows}
    assert all(gf.fleet_unarmed()[k].payload_kg == 0.0 for k in by)
    assert by["gavilan_f"]["stop"] is None and by["gavilan_f"]["hover"] is None and by["gavilan_a"]["stop"] < 20.0
    assert by["gavilan_a"]["band"] == 100.0 and by["peregrine"]["band"] < 60.0 and by["gavilan_a"]["repass"] == 0.0
    assert {k["group"] for k in gf.KPIS} == {"bandit", "general"} and all(k["key"] in rows[0] for k in gf.KPIS)
    sc = gf.kpi_scores(rows)
    assert sc["gavilan_f"]["stop"] == 0.0 and max(s["rate15"] for s in sc.values()) == 1.0
    assert all(0.0 <= v <= 1.0 for s in sc.values() for v in s.values())
    assert sc["gavilan_a"]["bandit"] > sc["emerillon"]["bandit"] > sc["gavilan_f"]["bandit"]
    assert sc["gavilan_a"]["rate25"] == 1.0                                         # unarmed, it out-turns the light wings at 25 m/s too


# ----------------------------------------------------------------------------------------------------------------- net
def test_net_launcher_bookkeeping_and_who_holds_a_bandit_in_range():
    assert gf.net_kg(1) == pytest.approx(sum(gf.NET["parts_g"].values()) * gf.NET["margin"] / 1e3)
    assert gf.net_kg(2) > 1.8 * gf.net_kg(1)                                      # one tube, valve and tank per shot
    fl = gf.fleet_net()
    rows = {r["key"]: r for r in gf.net_table({k: fl[k] for k in ("gavilan_a", "gavilan_f", "gavilan_n")})}
    assert fl["gavilan_a"].hovers and rows["gavilan_a"]["holds0"] and rows["gavilan_a"]["repass"] == 0.0
    for k in ("gavilan_f", "gavilan_n"):
        assert not rows[k]["holds0"] and 0.03 < rows[k]["trigger0"] < 0.2 and rows[k]["repass"] > 5.0
    assert rows["gavilan_n"]["vmin"] < rows["gavilan_f"]["vmin"] and rows["gavilan_n"]["trigger10"] > rows["gavilan_f"]["trigger10"]
    p = gf.net_pass(fl["gavilan_n"], 0.0)
    assert p["trigger_s"] == pytest.approx(0.5 * gf.NET["open_span_m"] / p["v_pass"]) and gf.NET["range_m"][0] < p["height_m"] < gf.NET["range_m"][1]
    assert gf.net_pass(fl["gavilan_n"], 20.0)["holds"]                             # it can fly beside a bandit at 20 m/s


# ---------------------------------------------------------------------------------------------------------------- hawk
def test_hawk_cad_tail_halves_sit_in_the_wash_and_the_model_reads_the_same_numbers():
    q = gv.h_resolve()
    assert all(q[k] == v for k, v in gf.H_PARAMS.items())
    g = gv.GavilanH().generate()
    assert g.shape.isValid() and len(g.shape.Solids()) == 1 and g.dimensions[1] == pytest.approx(1000.0, abs=1.0)
    L = gv.GavilanH.layout(q)
    assert L["blown_span_per_half"] == pytest.approx(L["wash_width"]) and 0.3 < L["blown_share"] < 0.7     # the whole wash lands on the tail
    assert gv.GavilanH().generate(taileron_deg=30.0).dimensions[2] > g.dimensions[2]
    assert gv.GavilanH().generate(weapon="emitter").dimensions[0] > g.dimensions[0]


def test_hawk_flies_past_its_stall_and_sits_between_the_fixed_wing_and_the_tailsitter():
    h, f, a = gf.gavilan_h(), gf.gavilan("F", emitters=0, p=gf.F_BROAD), gf.gavilan("A", emitters=0)
    assert h.kind == "wing" and math.degrees(h.theta_max) == pytest.approx(90.0) and h.hovers
    assert 45.0 < h.weight / h.S < 75.0                                             # a hawk's wing loading [N/m2]
    for V in (10.0, 15.0):
        assert gf.turn(f, V)["rate_inst_deg_s"] < gf.turn(h, V)["rate_inst_deg_s"] < gf.turn(a, V)["rate_inst_deg_s"]
    b = gf.brake(h)
    assert b["stops"] and b["distance_m"] > gf.brake(a)["distance_m"]              # it cannot point its thrust backwards
    rows = gf.h_authority()
    assert all(r["margin"] > 1.5 for r in rows) and rows[0]["tail_moment_Nm"] > 2.0     # the tail holds the stalled wing (hand estimate)
    import gavilan_cfd as gc
    assert gc.reference("H")["S"] == pytest.approx(h.S) and gc.pitch_authority("H") > 5.0


def test_hawk_with_the_one_shot_net_cad_balance_and_what_the_load_costs():
    p = gv.h_resolve(gv.H_NET)
    g = gv.GavilanH().generate(**gv.H_NET)
    assert g.shape.isValid() and len(g.shape.Solids()) == 1
    N = gv.GavilanH.net_layout(p)
    assert N["prop_plane_behind_muzzle"] > 100.0                                   # the net leaves ahead of the propellers
    assert p["taileron_gap"] > p["net_pipe_od"] / 2                                # the jet pipe passes between the tail halves
    with pytest.raises(Exception):
        gv.GavilanH().generate(weapon="net")                                       # the default 14 mm gap would foul the pipe
    assert gf.light_net_kg() == pytest.approx(sum(gf.LIGHT_NET["parts_g"].values()) * gf.LIGHT_NET["margin"] / 1e3)
    clean, armed = gf.h_mass_budget(False), gf.h_mass_budget(True)
    assert armed.mass == pytest.approx(gf.gavilan_h(net=True).mass) and clean.mass == pytest.approx(gf.gavilan_h().mass)
    assert not any(i.name.startswith("tail boom") for i in armed.items) and any(i.name.startswith("tail boom") for i in clean.items)
    for net in (False, True):
        sm = gf.h_static_margin(net)
        assert 0.25 < sm["cg_mac"] < 0.40 and sm["static_margin"] > 0.05           # ahead of the (hand-estimated) neutral point
    rows = {r["key"]: r for r in gf.h_net_table()}
    h, hn = rows["gavilan_h"], rows["gavilan_h_net"]
    assert hn["mass_kg"] - h["mass_kg"] < gf.light_net_kg() + 1e-9                 # the jet pipe replaces the boom
    assert hn["static thrust / weight"] > 2.0 and hn["hangs on its propellers"] and hn["wing stall [m/s]"] > h["wing stall [m/s]"]
    assert hn["turn rate @15 [deg/s]"] < h["turn rate @15 [deg/s]"] and hn["tail margin at 15 m/s"] > 1.5


# ------------------------------------------------------------------------------------------------------------------- J
def test_gavilan_j_cad_talons_reach_past_the_propellers_and_fold_away():
    p = gv.resolve(gv.variant("J"))
    out = gv.Gavilan().generate(**gv.variant("J"))
    folded = gv.Gavilan().generate(**gv.variant("J", talon_deg=0.0))
    assert out.shape.isValid() and len(out.shape.Solids()) == 1 and folded.shape.isValid()
    assert folded.dimensions[2] < out.dimensions[2] - 150.0                       # thrown out they stand well below the belly
    T = gv.Gavilan.talon_layout(p)
    assert T["beyond_lower_propellers"] > 100.0 and 400.0 < T["open_width"] < 500.0     # wider than a 0.35 m bandit
    assert gv.Gavilan.talon_layout(gv.resolve(gv.variant("J", talon_deg=0.0)))["open_width"] < 60.0
    assert p["bay_length"] < gv.resolve()["bay_length"] and p["length"] < gv.resolve()["length"]     # one pack shorter
    L = gv.Gavilan.layout(p)
    assert p["motor_z"] - (p["prop_diameter"] / 2 + p["guard_clearance"]) > 0.5 * p["thickness"] * p["root_chord"] + 5.0   # the rings clear the wing


def test_gavilan_j_is_lighter_carries_the_bandit_and_the_sprint_orders_as_reported():
    j, a = gf.gavilan_j(), gf.gavilan("A", emitters=0)
    b = gf.j_mass_budget()
    assert j.mass == pytest.approx(b.mass) and j.mass < a.mass and b.by_group()["battery"] == pytest.approx(0.246)
    assert j.t_max(0.0) / j.weight > 6.0 and j.t_max(0.0) / ((j.mass + gf.BANDIT_KG) * 9.80665) > 3.0
    assert gf.talons_kg() + gf.guards_kg() < 0.25
    near, far = gf.intercept("gavilan_j", 50.0), gf.intercept("gavilan_j", 200.0)
    assert near["catches"] and near["arrival_m_s"] == gf.CONTACT_SPEED and 2.0 < near["time_s"] < 3.5 < far["time_s"] < 7.0
    wing = gf.intercept("gavilan_f_broad", 50.0)
    assert not wing["catches"] and wing["arrival_m_s"] > 30.0                      # it gets there sooner and flies straight past
    assert gf.intercept("gavilan_h", 50.0)["time_s"] > near["time_s"] + 1.0        # the hawk-shaped one stops too slowly


# -------------------------------------------------------------------------------------------------------------- X, XP
def test_gavilan_x_four_identical_panels_that_are_its_feet_and_what_it_gains_on_j():
    p = gv.x_resolve()
    g = gv.GavilanX().generate()
    assert g.shape.isValid() and len(g.shape.Solids()) == 1
    L = gv.GavilanX.layout(p)
    assert L["tip_to_body"] > 50.0 and L["tip_to_tip"] > 50.0 and L["x_le_tip"] + L["c_tip"] == pytest.approx(p["length"])    # the tips are the feet
    assert L["foot_half_width"] > 1.8 * gv.Gavilan.layout(gv.resolve(gv.variant("J")))["foot_half_width"]                       # a steadier perch
    assert gv.GavilanX().generate(part="panel").shape.isValid() and len(gv.x_planform()) == 4
    x, j = gf.gavilan_x(), gf.gavilan_j()
    assert x.mass == pytest.approx(gf.x_mass_budget().mass) and x.mass < j.mass and x.span < j.span and x.length < j.length
    assert len(gf.PRINTED["gavilan_x"]["types"]) < len(gf.PRINTED["gavilan_j"]["types"])
    assert gf.x_wing_aero()["CLa"] < 0.6 * gf.wing_aero(gv.variant("J"))["CLa"]    # an X lifts about half as well per area
    assert gf.x_mass_budget().properties().com[0] < gf.x_geometry()["x_ac"]        # the centre of gravity ahead of the panels' aerodynamic centre
    assert x.t_max(0.0) / ((x.mass + gf.BANDIT_KG) * 9.80665) > 3.0


def test_gavilan_xp_feet_open_shut_fold_and_the_catch_sequence_adds_up():
    q = gv.xp_resolve()
    assert gv.GavilanXP().generate().shape.isValid()
    wide = gv.GavilanXP.foot_layout({**q, "toe_open_deg": 80.0})
    shut = gv.GavilanXP.foot_layout({**q, "toe_open_deg": 20.0})
    assert wide["aperture"] > 150.0 and shut["aperture"] < 10.0                    # spread it takes the bandit's body; shut the talons cross
    assert len(wide["feet"]) == 2 and len(wide["feet"][0]["toes"]) == 4 and wide["past_lower_propellers"] > 50.0
    assert len(gv.GavilanXP.foot_layout({**q, "front_toes": 2})["feet"][0]["toes"]) == 3
    folded = gv.GavilanXP.foot_layout({**q, "leg_deg": 180.0, "toe_open_deg": 20.0})
    assert max(t[1][0] for f in folded["feet"] for t in f["toes"]) < q["length"] and min(t[1][0] for f in folded["feet"] for t in f["toes"]) > -10.0   # inside its own length
    seq = gf.catch_sequence(100.0)
    ts = [e["t"] for e in seq["events"]]
    assert ts == sorted(ts) and 3.0 < seq["time_to_contact_s"] < 5.0
    assert seq["snap_fits_window"] and seq["toe_snap_s"] < 0.05 and seq["trigger_tolerance_s"] < 0.03
    assert 0.03 < seq["sideways_tolerance_m"] < 0.08                               # the pincer forgives centimetres, not decimetres
    assert seq["thrust_over_weight_with_bandit"] > 3.0 and seq["spare_thrust_if_bandit_fights_N"] > 0.0
    tr = seq["trace"]
    assert tr["range_m"][0] == pytest.approx(100.0, abs=3.0) and tr["range_m"][-1] == pytest.approx(0.0, abs=0.05)
    assert tr["speed_m_s"][-1] == pytest.approx(gf.XP_CONTACT_SPEED, abs=0.2) and tr["leg_deg"][0] == 180.0 and tr["leg_deg"][-1] == 100.0


def test_gavilan_x_with_the_one_shot_net_pod_clears_the_propellers_and_stops_short_of_its_net():
    p = gv.x_resolve(gv.X_NET)
    g = gv.GavilanX().generate(**gv.X_NET)
    assert g.shape.isValid() and len(g.shape.Solids()) == 1
    N = gv.GavilanX.net_layout(p)
    assert N["clear_of_lower_discs"] > 20.0 and N["clear_of_camera"] > 10.0 and N["x_muzzle"] < 0.0     # the muzzle ahead of the nose
    m = gf.gavilan_x_net()
    assert m.mass == pytest.approx(gf.gavilan_x().mass - gf.talons_kg() + gf.net_pod_kg()) and m.t_max(0.0) / m.weight > 4.0
    for rng in (4.0, 6.0, 8.0):
        v = gf.fastest_safe_shot(rng, m)
        assert gf.net_shot(v, rng, m)["safe"] and not gf.net_shot(v + 1.0, rng, m)["safe"]
    assert gf.fastest_safe_shot(8.0, m) > gf.fastest_safe_shot(4.0, m)          # further out, it may come in faster
    a = gf.x_net_attack(100.0)
    assert 3.0 < a["time_to_foul_s"] < 5.0 and a["time_to_foul_s"] > a["time_to_shot_s"]
