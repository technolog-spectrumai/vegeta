"""Kestrel-4Q (notebook 38): the zero-stagger layout and its rules, the CAD, the masses and the CG, the hold point (the
allocation's rank for each sense of rotation, the attainable moments, a motor out), the biplane's lattice against
Prandtl's biplane factor, the drag of the bracing, the full-range polar, the slipstream, the transitions both ways
through the stall (and what fails without elevons), the CFD cases with four rotor disks and the CFD fed back into the
transition model, the truss and the ground stance.

Run: cd /home/user/vegeta && python3 -m pytest -q notebooks/designs/tests/test_kestrel.py      (add -m "not slow")
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import kestrel
import kestrel_control as kc
import kestrel_flight as kf
import kestrel_structure as kst
import kestrel_systems as ks
import kestrel_transition as kt


@pytest.fixture(scope="module")
def P():
    return kestrel.resolve()


@pytest.fixture(scope="module")
def L(P):
    return kestrel.Kestrel4Q.layout(P)


@pytest.fixture(scope="module")
def CI():
    return ks.cg_inertia(ks.mass_table())


@pytest.fixture(scope="module")
def DR():
    return ks.drive_table()


@pytest.fixture(scope="module")
def VEH():
    return kt.make_vehicle()


# ------------------------------------------------------------------------------------------------ layout and CAD
def test_zero_stagger_equal_wings_and_the_rules(P, L):
    assert "stagger" not in {p.name for p in kestrel.Kestrel4Q.parameters}             # zero stagger is built in, not a setting
    assert L["S_ref"] == pytest.approx(2 * L["S_wing"]) and L["gap"] == pytest.approx(P["gap"] / 1000)
    assert kestrel.Kestrel4Q.checks(P) == []
    assert L["prop_gap"] * 1000 >= P["prop_clearance"] and L["prop_to_pod"] * 1000 >= P["prop_clearance"]
    zs = sorted(m["z"] for m in L["motors"].values())
    assert zs == pytest.approx([-L["h"], -L["h"], L["h"], L["h"]])                     # two vertically separated pairs
    assert len({m["x"] for m in L["motors"].values()}) == 1                            # all four in one propeller plane
    assert sum(m["y"] for m in L["motors"].values()) == pytest.approx(0.0)             # mirror symmetry


def test_checks_refuse_overlapping_propellers(P):
    bad = dict(P, gap=240.0)
    assert any("clear each other" in m for m in kestrel.Kestrel4Q.checks(bad))
    assert any("pod" in m for m in kestrel.Kestrel4Q.checks(dict(P, motor_y=150.0)))


def test_rotate_matches_the_cad_convention():
    x, y, z = kestrel.rotate((0.0, 5.0, 0.0), 90.0, 200.0)                             # the nose tip, pitched up 90° about x = 200
    assert (x, y, z) == pytest.approx((200.0, 5.0, 200.0))


def test_cad_builds_symmetric_with_both_wings_at_one_station(P):
    pytest.importorskip("cadquery")
    d = kestrel.Kestrel4Q()
    wing = d.generate(part="wing").measure()
    assert wing["bbox_min"][0] == pytest.approx(P["wing_le_x"], abs=0.5)
    assert wing["bbox_min"][2] > 0 and wing["center_of_mass"][2] == pytest.approx(P["gap"] / 2, abs=0.5)
    whole = d.generate(part="airframe").measure()
    assert whole["valid"] and abs(whole["center_of_mass"][1]) < 1.0                    # mirror-symmetric
    assert whole["dimensions"][1] == pytest.approx(P["span"], abs=2.0)
    with pytest.raises(Exception):
        d.generate(gap=200.0)


# ------------------------------------------------------------------------------------------------ masses and drive
def test_mass_table_and_cg(L, CI):
    assert 1.4 < CI["mass_kg"] < 2.0
    # the battery slides along the pod's floor under the bays; at its forward stop the CG lands a millimetre behind the target
    assert CI["x_cg"] == pytest.approx(L["x_le"] + ks.X_CG_TARGET_FRAC * L["chord"], abs=0.002)
    assert abs(CI["y_cg"]) < 1e-9 and kf.neutral_point(x_cg=CI["x_cg"])["static_margin"] > 0.05
    lo, hi = ks.battery_range()
    assert lo <= CI["battery_x"] <= hi
    assert CI["Ixx"] > CI["Iyy"] and CI["Izz"] > CI["Iyy"]                              # the span dominates roll and yaw


def test_drive_table_is_consistent(DR):
    assert DR.T_max[0] > 8.0 and np.all(np.diff(DR.T_max) < 0)                          # full thrust falls with the inflow
    assert 0.008 < DR.kappa[0] < 0.03                                                    # Q/T of a 9-inch propeller [m]
    assert DR.power(4.0, 0.0) > DR.power(2.0, 0.0) > 0.0


@pytest.mark.slow
def test_stored_drive_table_matches_boreas(DR):
    fresh = ks.drive_table(recompute=True)
    assert np.allclose(fresh.T_max, DR.T_max, rtol=0.01) and np.allclose(fresh.kappa, DR.kappa, rtol=0.01)


# ------------------------------------------------------------------------------------------------ the hold point
def test_only_the_diagonal_pairing_controls_the_hover(CI, DR):
    tab = kc.rank_table(kappa=float(DR.kap(0.0)), cg=CI)
    assert tab.loc[kc.BASELINE_SPIN, "rank"] == 4
    for name in kc.SPINS:
        if name != kc.BASELINE_SPIN:
            assert tab.loc[name, "rank"] == 3, name                                     # one axis lost
    assert bool(tab.loc["side columns alike (M1, M3 | M2, M4)", "mirror-symmetric"])    # symmetric, and still uncontrollable


def test_fore_aft_offsets_carry_no_lever(P, CI):
    B = kc.allocation(P, kappa=0.015, cg=CI)
    moved = dict(CI, x_cg=CI["x_cg"] + 0.1)                                              # move the CG along the thrust axis
    assert np.allclose(kc.allocation(P, kappa=0.015, cg=moved), B)


def test_attainable_moments_in_the_hover(CI, DR):
    B = kc.allocation(kappa=float(DR.kap(0.0)), cg=CI)
    att = kc.attainable(B, CI["mass_kg"] * ks.G, DR.T_min, DR.T_max[0])
    for ax in ("L", "M", "N"):
        assert att[ax][0] < 0 < att[ax][1]
    assert abs(att["N"][0]) > abs(att["M"][0]) > 5 * abs(att["L"][0])                   # the heading lever is the weak one
    h = kestrel.layout()["h"]
    assert att["M"][1] <= 4 * h * (DR.T_max[0] - DR.T_min) / 2 + 1e-6


def test_a_motor_out_leaves_an_unbalanced_torque(CI, DR):
    tab = kc.motor_out(mass_kg=CI["mass_kg"], cg=CI)
    assert (tab["unbalanced torque about the axis [N·m]"].abs() > 0.1).all()


# ------------------------------------------------------------------------------------------------ aerodynamics
@pytest.fixture(scope="module")
def A():
    pytest.importorskip("cadquery")                                                     # peregrine_flight's lattice imports the CAD module
    return kf.aero()


def test_biplane_lattice_against_prandtl(A, L):
    # Prandtl's biplane: induced drag factor (1 + σ) / 2 of the monoplane, σ ≈ 0.35 at gap/span 0.28 (tables: 0.33-0.37)
    f = A["factors"]["induced drag vs the monoplane"]
    assert 0.64 < f < 0.74
    assert 0.85 < A["factors"]["lift slope vs two isolated wings"] < 0.95                # the gap's mutual downwash
    assert A["biplane"]["upper share"] == pytest.approx(0.5, abs=0.01)                  # zero stagger, no decalage: equal shares
    assert A["biplane"]["x_np"] == pytest.approx(L["x_ac"], abs=0.01 * L["chord"] + 2e-3)


def test_neutral_point_and_the_propellers(A, CI):
    npd = kf.neutral_point(x_cg=CI["x_cg"], a=A)
    assert npd["Cma_props"] > 0 and npd["Cma_pod"] > 0                                  # both destabilising
    assert 0.05 < npd["static_margin"] < 0.15
    bound = kf.prop_normal_force_slope(16.0, 0.8, bound=True)
    assert bound > 5 * kf.prop_normal_force_slope(16.0, 0.8)                            # the momentum bound is far above the estimate


def test_bracing_drag_and_fairings():
    round_ = kf.drag_buildup()
    faired = kf.drag_buildup(faired=True)
    assert round_["bracing_share"] > 0.5 > faired["bracing_share"]
    assert faired["Cd0"] < 0.6 * round_["Cd0"]


def test_full_range_polar_shape():
    pol = kf.Polar(cl_alpha=4.1, cd0=0.03, k=0.07, cl_max=0.81, cd90=1.24)
    a = np.radians([0.0, 5.0, 45.0, 90.0, 180.0])
    cl, cd, cm = pol.coefficients(a)
    assert cl[0] == pytest.approx(0.0) and cl[1] == pytest.approx(4.1 * math.sin(a[1]) * math.cos(a[1]), rel=0.02)
    assert cd[3] == pytest.approx(0.03 + 1.24, rel=0.01) and abs(cl[3]) < 1e-6           # broadside: a flat plate
    assert cl[2] == pytest.approx(0.62, abs=0.03)                                        # CD90 / 2 at 45°
    assert cm[3] < 0 and cm[0] == pytest.approx(0.0)                                     # mid-chord pressure at 90°: nose down


def test_glauert_induced_velocity():
    A = kestrel.layout()["A_disc"]
    vh = math.sqrt(4.0 / (2 * kf.RHO * A))
    assert kf.induced_velocity(4.0, 0.0, 0.0, A) == pytest.approx(vh, rel=1e-3)
    V = 15.0                                                                             # axial climb: vᵢ = −V/2 + √((V/2)² + vₕ²)
    assert kf.induced_velocity(4.0, V, 0.0, A) == pytest.approx(-V / 2 + math.sqrt((V / 2) ** 2 + vh ** 2), rel=1e-3)
    vi = kf.induced_velocity(4.0, 10.0, math.radians(90.0), A)                            # edgewise: vᵢ √(V² + vᵢ²) = vₕ²
    assert vi * math.sqrt(100.0 + vi ** 2) == pytest.approx(vh ** 2, rel=1e-3)


# ------------------------------------------------------------------------------------------------ the transitions
def test_slipstream_lowers_the_blown_angle(VEH):
    V, a = 4.0, math.radians(40.0)
    s = kt.slipstream(VEH, 5.0, V * math.cos(a), V * math.sin(a))
    blown = math.atan2(V * math.sin(a), V * math.cos(a) + s["dv"])
    assert math.degrees(blown) < 15.0 and 0.0 < s["blown_span"] < VEH.L["span"]


def test_forward_transition_passes_through_the_stall(VEH):
    ep = kt.simulate(VEH, "forward", t_tr=3.0)
    m = ep.metrics
    assert m["pass"], m
    assert m["time stalled, free stream [s]"] > 0.3                                     # it does cross the stall...
    assert m["α blown max while stalled [deg]"] < math.degrees(VEH.pol.alpha_stall) + 5  # ...with the blown span attached
    assert m["speed at the end [m/s]"] > 14.0


def test_back_transition_reaches_the_hover(VEH):
    ep = kt.simulate(VEH, "back", t_tr=3.0)
    m = ep.metrics
    assert m["pass"], m
    assert m["hover reached at [s]"] < ep.log["t"][-1] and m["speed at the end [m/s]"] < 0.5


def test_without_elevons_the_roll_check_fails(VEH):
    bare = kt.make_vehicle(elevons=False)
    ep = kt.simulate(bare, "back", t_tr=3.0)
    assert ep.metrics["roll demand / rotors' authority"] > 1.0 and not ep.metrics["roll holds"]


# ------------------------------------------------------------------------------------------------ CFD
def test_cfd_points_and_cases_prepare(VEH, tmp_path):
    pytest.importorskip("cadquery")
    import kestrel_cfd as kcfd
    from vegeta import aeromant
    eps = [kt.simulate(VEH, k, t_tr=3.0) for k in ("forward", "back")]
    pts = kcfd.points(eps, mass_kg=VEH.mass)
    assert {pt["kind"] for pt in pts} == {"forward", "back", "hover"}
    assert len({(pt["source"], round(pt["t"], 4)) for pt in pts}) == len(pts)              # no sample twice
    spec = kcfd.specs(pts)
    assert (kcfd.not_run_table(spec)["status"] == "NOT RUN").all()
    env = aeromant.OpenFOAMEnvironment(env={"PATH": str(tmp_path)})
    name = next(n for n, s in spec.items() if s["kind"] == "back")
    s = spec[name]
    c = kcfd.case(tmp_path / "cfd", {k: v for k, v in s.items() if k != "props"}, x_cg_m=VEH.x_cg, env=env)
    assert c.prepare().ok
    src = (Path(c.workdir) / "system/fvOptions").read_text()
    assert src.count("type            rotorDisk;") == 4
    ds = kcfd.disk_setup(s, VEH.x_cg)
    assert ds["rpm1"] < ds["rpm3"]                                                      # back transition: the lower pair pushes harder (nose up)
    assert np.linalg.norm(ds["disk_axis"]) == pytest.approx(1.0)


def test_cfd_correction_feeds_the_model(VEH):
    import pandas as pd
    import kestrel_cfd as kcfd
    pt = {"alpha_deg": 30.0, "V": 8.0, "T_up": 3.0, "T_lo": 3.5}
    mc = kcfd.model_coefficients(VEH, pt)
    rows = {}
    for da in (0.0, 15.0, 30.0):                                                        # a "CFD" 0.1 above the model in CL everywhere
        q = dict(pt, alpha_deg=pt["alpha_deg"] + da)
        m_ = kcfd.model_coefficients(VEH, q)
        rows[str(da)] = {"status": "solved", "rotor disks": True, "α [deg]": q["alpha_deg"], "V [m/s]": q["V"], "T upper [N]": q["T_up"],
                         "T lower [N]": q["T_lo"], "CL": m_["CL"] + 0.1, "CD": m_["CD"], "Cm (CG, chord)": m_["Cm"]}
    corr = kcfd.CFDAirframe.from_results(pd.DataFrame(rows).T, VEH)
    assert corr.correction(math.radians(30.0), mc["mu"])[0] == pytest.approx(0.1, abs=1e-6)
    veh2 = kt.make_vehicle(airframe=corr)
    a = math.radians(30.0)
    f0 = kt.airframe_forces(VEH, 8 * math.cos(a), 8 * math.sin(a), 3.0, 3.5)
    f1 = kt.airframe_forces(veh2, 8 * math.cos(a), 8 * math.sin(a), 3.0, 3.5)
    lift = lambda f: -f["Fx"] * math.sin(a) + f["Fz"] * math.cos(a)
    assert lift(f1) - lift(f0) == pytest.approx(0.1 * 0.5 * kf.RHO * 64 * VEH.L["S_ref"], rel=1e-6)


# ------------------------------------------------------------------------------------------------ structure and ground
def test_truss_stiffens_the_wing_pair(CI):
    tab = kst.compare_bracing(mass_kg=CI["mass_kg"])
    braced, bare = tab.loc["posts + X diagonals"], tab.loc["no truss (two cantilevers)"]
    assert braced["tip deflection, upper [mm]"] < 0.2 * bare["tip deflection, upper [mm]"]
    assert braced["spar stress [MPa]"] < bare["spar stress [MPa]"] and bare["spar reserve"] > 2.0
    uneven = kst.frame(mass_kg=CI["mass_kg"], bracing="posts", upper_share=0.75)
    assert abs(uneven["bars"][0]["force [N]"]) > 1.0                                    # the post works when the wings load unevenly


def test_cantilever_against_beam_theory():
    """No truss, no inertia, a uniform load: the frame's spar against w = q l⁴ / 8 E I and M = q l² / 2 at the root."""
    L = kestrel.layout()
    _, I_s = kst._section(*kst.TUBES["spar"])
    E = kst.CARBON["E"]
    r = kst.frame(mass_kg=1.0, n=1.0, bracing="none", distribution="uniform", inertia=False)
    l = L["span"] / 2
    q = 1.0 * kf.G / 2 / 2 / l                                                           # per wing, per metre of the half span
    assert r["tip deflection, upper [mm]"] / 1e3 == pytest.approx(q * l ** 4 / (8 * E * I_s), rel=0.03)
    assert r["spar peak moment [N·m]"] == pytest.approx(q * l ** 2 / 2, rel=0.03)


def test_ground_stance(CI):
    g = kst.ground_stance(mass_kg=CI["mass_kg"], x_cg=CI["x_cg"], z_cg=CI["z_cg"])
    assert g["least tip-over angle [deg]"] > 30.0
    assert g["wind that tips it, broadside [m/s]"] < g["wind that tips it, edge-on [m/s]"]
    assert g["foot rod margin, two feet first"] > 2.0


def test_fairing_section_and_the_faired_truss(P):
    c, pts = kestrel.fairing_section(8.0, 0.33, 0.8)
    xs = [x for x, _ in pts]
    ts = [t for _, t in pts]
    assert c == pytest.approx(9.6 / 0.33)
    assert min(xs) == pytest.approx(-0.3 * c, abs=0.05) and max(xs) == pytest.approx(0.7 * c, abs=0.05)   # the tube at 30 % chord
    assert max(ts) - min(ts) == pytest.approx(9.6, rel=0.02)                                  # 0.8 mm outside the Ø8 tube
    pytest.importorskip("cadquery")
    d = kestrel.Kestrel4Q()
    bare, faired = d.generate(part="truss").measure(), d.generate(part="truss", strut_fairing=True).measure()
    assert faired["valid"] and faired["volume"] > 3 * bare["volume"]
    assert faired["dimensions"][0] == pytest.approx(c, abs=0.5)                                # the chords along x
    assert faired["dimensions"][1] == pytest.approx(bare["dimensions"][1], abs=3.0)           # same footprint seen from the front


# ------------------------------------------------------------------------------------------------ Part II: the options and the X-wing
def test_part_ii_cad_options_build_and_weigh(P):
    pytest.importorskip("cadquery")
    d = kestrel.Kestrel4Q()
    base = d.generate(part="airframe").measure()["volume"]
    el = d.generate(part="airframe", elevons=True, elevon_deg=15.0).measure()
    assert el["valid"] and abs(el["volume"] - base) < 0.02 * base                         # the elevons are cut from the wing, not added
    e = d.generate(part="elevon", elevons=True).measure()
    assert e["bbox_min"][1] == pytest.approx(P["elevon_inner_y"], abs=0.5) and e["bbox_max"][1] == pytest.approx(P["span"] / 2, abs=0.5)
    assert e["bbox_min"][0] > P["wing_le_x"] + (1 - P["elevon_chord"]) * P["chord"] - 1.0
    for br in ("posts", "none"):
        assert d.generate(part="airframe", bracing=br).measure()["valid"]
    with pytest.raises(Exception):
        d.generate(part="truss", bracing="none")
    with pytest.raises(Exception):
        d.generate(part="elevon")
    L = kestrel.layout(dict(elevons=True, bracing="posts"))
    assert L["diagonals"] == 0 and L["posts"] == 2 and L["S_elevon"] == pytest.approx(4 * 0.225 * 0.28 * 0.14)
    m0 = ks.cg_inertia(ks.mass_table(kestrel.resolve()))["mass_kg"]
    assert ks.cg_inertia(ks.mass_table(kestrel.resolve(dict(elevons=True))))["mass_kg"] == pytest.approx(m0 + 4 * ks.SERVO_KG)
    assert ks.cg_inertia(ks.mass_table(kestrel.resolve(dict(bracing="none"))))["mass_kg"] < ks.cg_inertia(ks.mass_table(kestrel.resolve(dict(bracing="posts"))))["mass_kg"] < m0


def test_panel_model_reproduces_the_biplane(VEH):
    """The generalised panel model: four flat panels, the same numbers as Part I."""
    assert len(VEH.panels) == 4 and all(pn["phi"] == 0.0 for pn in VEH.panels) and VEH.z_rotor == pytest.approx(VEH.L["h"])
    f = kt.airframe_forces(VEH, 8.0, 2.0, 3.0, 3.5)
    assert len(f["panels"]) == 4 and f["S_blown_up"] == pytest.approx(2 * f["panels"][0]["S_blown"])
    m = kt.simulate(VEH, "forward", t_tr=3.0).metrics
    assert m["pass"] and m["height loss [m]"] < 0.5


def test_xwing_layout_mass_and_levers():
    import kestrel_xwing as kx
    L = kx.layout()
    assert kx.checks(L) == [] and L["S_ref"] == pytest.approx(0.28, abs=1e-9)
    ci = ks.cg_inertia(kx.mass_table(L))
    assert 1.4 < ci["mass_kg"] < 1.8 and not ci["battery_clipped"]
    assert ci["mass_kg"] < ks.cg_inertia(ks.mass_table(kestrel.resolve()))["mass_kg"]       # no truss, no fin plate
    B = kc.allocation(kappa=0.015, cg=ci, L=L)
    assert np.linalg.matrix_rank(B) == 4
    assert abs(B[2]).max() == pytest.approx(abs(B[3]).max(), abs=abs(ci["z_cg"]) + 1e-9)       # pitch and yaw levers equal at 45° (± the CG's height)
    c = kx.cantilever(L, mass_kg=ci["mass_kg"])
    assert c["first bending, with the tip motor [Hz]"] >= kx.F1_MIN - 1e-6 and c["spar reserve"] > 1.5


def test_xwing_lift_slope_is_about_half_and_it_flies():
    pytest.importorskip("cadquery")
    import kestrel_xwing as kx
    L = kx.layout()
    a = kx.aero(L)
    bi = kf.aero()["biplane"]["CLa"]
    assert 0.4 < a["CLa"] / bi < 0.55                                                           # cos² 45° of the flat pair, less the X's interference
    veh = kx.make_vehicle()
    assert veh.lift_slope_vertical == pytest.approx(veh.pol.cl_alpha * 0.5, rel=1e-6)
    assert all(pn["phi"] == pytest.approx(math.radians(45.0)) for pn in veh.panels)
    assert kt.simulate(veh, "forward", t_tr=3.0, V_cruise=20.0).metrics["pass"]
    cr = kx.cruise(veh)
    assert bool(cr.loc[16.0, "feasible"]) and cr.loc[16.0, "α [deg]"] > 9.0                     # the same-area X cruises near the stall at 16 m/s


def test_stiffness_and_modes_of_the_bracings():
    df = kst.stiffness_and_modes()
    k = df["k symmetric [N/mm]"]
    assert k["posts + X diagonals"] > 50 * k["no truss (two cantilevers)"]
    assert k["posts only"] == pytest.approx(k["no truss (two cantilevers)"], rel=1e-3)          # pinned posts carry nothing symmetric
    assert df.loc["posts only", "k antisymmetric [N/mm]"] > 100 * df.loc["no truss (two cantilevers)", "k antisymmetric [N/mm]"]
    f = df.loc["no truss (two cantilevers)", "first mode, symmetric [Hz]"]
    assert 5.0 < f < 40.0 and f > 3 * df.attrs["references"]["pitch loop bandwidth [Hz]"]


def test_elevon_hinge_and_authority(CI):
    h = kc.elevon_hinge(dict(elevons=True), V=16.0)
    assert h["servo margin, cruise"] > 1.5 and h["servo margin, hover"] > 1.5
    df = kc.elevon_authority(dict(elevons=True), mass_kg=CI["mass_kg"], cg=CI)
    assert df.loc["hover (slipstream only)", "elevons / rotors, about the axis"] > 1.0        # the elevons beat the reaction torque for the heading
    assert df.loc["cruise 16 m/s", "elevons / rotors, about the axis"] > 3.0


def test_section_study_on_zephyr(tmp_path):
    """Part II § 13 runs on vegeta.zephyr (branch dev_zephyr); skipped where it is not merged yet."""
    pytest.importorskip("vegeta.zephyr")
    import kestrel_section as ksec
    s = ksec.section("NACA 23112", 0.02)
    assert s.name.startswith("NACA 23112") and s.thickness == pytest.approx(0.12)
    specs = ksec.cfd_specs([("NACA 0012", None), ("NACA 23112", 0.02)], alphas=(0.0, 8.0))
    assert len(specs) == 4
    import kestrel_cfd as kcfd
    tab = kcfd.section_not_run_table(specs)
    assert (tab["status"] == "NOT RUN").all() and tab["Re (chord)"].astype(float).min() > 1e5
    pytest.importorskip("cadquery")
    from vegeta import aeromant
    env = aeromant.OpenFOAMEnvironment(env={"PATH": str(tmp_path)})
    c = kcfd.section_case(tmp_path / "cfd", next(iter(specs.values())), env=env)
    assert c.prepare().ok
    if __import__("vegeta.zephyr").zephyr.backends()["neuralfoil"]:
        t = ksec.table(recompute=False) if ksec.SECTIONS_JSON.exists() else {ksec.key("NACA 0012"): ksec.study_one("NACA 0012")}
        k0 = ksec.key("NACA 0012")
        assert abs(t[k0]["cruise"]["cm0"]) < 0.01 and 0.25 < t[k0]["elevon"]["tau"] < 0.5


# ------------------------------------------------------------------------------------------------ the nose, the speed range, the manoeuvrability
def test_pod_profile_is_an_ogive_with_a_monotone_tail(P):
    prof = kestrel.Kestrel4Q.pod_profile(P)
    xs, ws, hs = zip(*prof)
    i_full = next(i for i, x in enumerate(xs) if x >= P["nose_length"] - 1e-9)
    assert all(b >= a for a, b in zip(hs[:i_full], hs[1:i_full + 1])) and hs[i_full] == pytest.approx(0.98 * P["gap"])   # the nose grows to the full section
    assert hs[0] >= 0.9 * P["camera_diameter"] and hs[0] < 0.4 * P["gap"]                                               # the tip: the camera's flat
    tail = [h for x, _, h in prof if x >= P["pod_length"] - P["tail_length"]]
    assert all(b <= a for a, b in zip(tail, tail[1:])) and tail[-1] < 0.6 * tail[0]                                     # the tail shrinks monotonically
    # the ellipsoid: half-way along the nose the section is √(1 − 0.25) = 0.87 of the full one
    h_mid = float(np.interp(P["nose_length"] / 2, xs, hs))
    assert h_mid == pytest.approx(math.sqrt(0.75) * 0.98 * P["gap"], rel=0.03)
    assert any("mid-body" in m for m in kestrel.Kestrel4Q.checks(dict(P, nose_length=250.0, tail_length=200.0)))


def test_pod_builds_monotone(P):
    pytest.importorskip("cadquery")
    import cadquery as cq
    shape = kestrel.Kestrel4Q().generate(part="pod").shape
    solid = cq.Workplane("XY").add(shape)
    # a true section at each station (the pod cut by a 1 mm slab): its height must grow to the mid-body and shrink after it
    zs = []
    x_end = P["pod_length"]
    for x, _, h in kestrel.Kestrel4Q.pod_profile(P):
        xc = min(x + 0.5, x_end - 0.5)                                                        # the slab inside the pod at its last station
        cut = solid.intersect(cq.Workplane("XY").box(1.0, 400.0, 400.0).translate((xc, 0, 0)))
        if not cut.vals():
            continue
        zs.append((x, cut.val().BoundingBox().zmax, h / 2))
    assert len(zs) >= 10
    for x, z, h2 in zs:
        assert z == pytest.approx(h2, abs=2.5), (x, z, h2)                                   # the built section is the profile's
    hs = [z for _, z, _ in zs]
    i_max = int(np.argmax(hs))
    assert all(b >= a - 0.5 for a, b in zip(hs[:i_max], hs[1:i_max + 1])) and all(b <= a + 0.5 for a, b in zip(hs[i_max:], hs[i_max + 1:]))


def test_drag_vs_speed_and_the_maximum_speed(VEH):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import kestrel_drawings as kd
    fig, axs = plt.subplots(1, 3)
    T = kd.drag_vs_speed({"4Q": VEH}, speeds=np.arange(8.0, 40.5, 0.5), axs=axs)
    plt.close(fig)
    r = T.loc["4Q"]
    assert r["V stall [m/s]"] == pytest.approx(kf.stall_speed(mass_kg=VEH.mass, pol=VEH.pol), rel=1e-6)
    assert 25.0 < r["V max, thrust = drag [m/s]"] < 36.0                                     # the 9x6 runs out of thrust near its pitch speed
    assert r["power at V max [W]"] <= 4 * VEH.drive.P_cap + 13.0 + 1e-6
    assert r["drag at 16 m/s [N]"] == pytest.approx(r["parasite at 16 [N]"] + r["induced at 16 [N]"])
    assert VEH.drive.power(50.0, 36.0) == VEH.drive.P_cap                                     # the cap holds where the table runs out of thrust


def test_cruise_manoeuvrability(VEH):
    df = kc.cruise_manoeuvrability(VEH)
    r16 = df.loc[16.0]
    assert r16["pitch: elevons [rad/s²]"] > r16["pitch: rotors [rad/s²]"]                     # in cruise the elevons, not the thrust split, pitch it
    assert 0 < r16["ΔT available [N]"] < 2.0 and r16["steady roll rate [deg/s]"] > 60.0
    assert r16["load factor available"] < 4.0 and df.loc[25.0, "load factor available"] == pytest.approx(4.0)
    assert df["turn rate [deg/s]"].max() > 60.0
    bare = kc.cruise_manoeuvrability(kt.make_vehicle(elevons=False))
    assert bare.loc[16.0, "pitch: elevons [rad/s²]"] == 0.0 and bare.loc[16.0, "steady roll rate [deg/s]"] < r16["steady roll rate [deg/s]"]


# ----------------------------------------------------------------------------------------------- failure is a result
class _FakeResult:
    def __init__(self, ok, metrics=None, messages=()):
        self.ok, self.metrics, self.messages = ok, metrics or {}, list(messages)


def _fake_campaign():
    """A campaign like the first real one: the bare sweep solved, two blown cases ran into 1e94, the rest failed."""
    import kestrel_cfd as kcfd
    pts = [{"source": "forward 3 s", "kind": "forward", "alpha_deg": a, "V": v, "T_up": 4.6, "T_lo": 4.7} for a, v in ((53.0, 1.5), (20.0, 5.8), (10.0, 9.3))]
    spec = kcfd.specs(pts, sweep=True)
    res = {}
    for name, sp in spec.items():
        if not sp["props"]:
            a = sp["alpha_deg"]
            res[name] = _FakeResult(True, {"Cl": 0.07 * min(a, 10.0) - 0.05 + (0.03 if a > 10 else 0.0), "Cd": 0.05 + 0.0003 * a * a, "Cm": -0.004 * a,
                                           "iterations": 400, "converged": False, "mesh_cells": 280000})
        elif sp["alpha_deg"] > 40:
            res[name] = _FakeResult(True, {"Cl": 1.5e94, "Cd": -4.9e94, "Cm": -6.1e94, "iterations": 400, "converged": False, "mesh_cells": 498000})
        elif sp["alpha_deg"] > 15:
            res[name] = _FakeResult(False, messages=["step solver failed: simpleFoam exited with 1 (sigFpe)"])
        else:
            res[name] = _FakeResult(True, {"Cl": 0.9, "Cd": 0.2, "Cm": -0.1, "iterations": 400, "converged": True, "mesh_cells": 490000})
    return kcfd.table(res, spec), spec


def test_cfd_table_accepts_failure_and_divergence_as_results(VEH):
    import kestrel_cfd as kcfd
    T, spec = _fake_campaign()
    st = T["status"].astype(str)
    assert st.value_counts().to_dict() == {"solved": len(kcfd.SWEEP_ALPHAS) + 1, "diverged": 1, "FAILED": 1}
    assert T.loc[st == "FAILED", "note"].iloc[0].startswith("step solver failed")
    assert T.loc[st == "diverged", "note"].iloc[0].startswith("diverged")
    sm = kcfd.summary(T)
    assert sm["solved with rotor disks"] == 1 and sm["solved bare"] == len(kcfd.SWEEP_ALPHAS) and "1 diverged, 1 failed" in sm["line"]
    # the correction from the blown cases uses the one sane row only; the diverged one never reaches the model
    corr = kcfd.CFDAirframe.from_results(T, VEH)
    assert corr is not None and len(corr.samples) == 1 and all(abs(corr.samples[0][k]) < 5 for k in ("dCL", "dCD", "dCm"))
    # drop that row too: no correction, not an exception
    T2 = T.copy()
    T2.loc[(T2["rotor disks"].astype(bool)) & (st == "solved"), "status"] = "FAILED"
    assert kcfd.CFDAirframe.from_results(T2, VEH) is None
    # the bare sweep against the model, and a correction in α from it
    cmp = kcfd.sweep_comparison(T, VEH)
    assert len(cmp) == len(kcfd.SWEEP_ALPHAS) and np.allclose(cmp["ΔCL"].astype(float), cmp["CL CFD"].astype(float) - cmp["CL model"].astype(float))
    nums = kcfd.sweep_numbers(cmp)
    assert 3.0 < nums["CFD"]["CLα [1/rad]"] < 5.0 and 2.5 < nums["model"]["CLα [1/rad]"] < 6.0
    sw = kcfd.CFDAirframe.from_results(T, VEH, source="sweep")
    assert sw is not None and sw.source == "sweep" and len(sw.samples) == len(kcfd.SWEEP_ALPHAS)
    d0 = sw.correction(math.radians(5.0), 0.3)                      # μ is ignored: the sweep knows only α
    assert np.allclose(d0, sw.correction(math.radians(5.0), 1.0)) and np.all(np.isfinite(d0))


def test_cfd_records_roundtrip_and_the_plug_in(tmp_path, VEH):
    import kestrel_cfd as kcfd
    T, _ = _fake_campaign()
    path = tmp_path / "kestrel_cfd_results.json"
    recs = kcfd.save_records(T, path, key="4Q-1", meta={"quality": "screening"})
    doc = kcfd.load_records(path, "4Q-1")
    assert doc["meta"]["quality"] == "screening" and len(doc["records"]) == len(recs) == len(kcfd.SWEEP_ALPHAS) + 1
    assert doc["summary"]["FAILED"] == 1 and doc["summary"]["diverged"] == 1 and len(doc["statuses"]) == len(T)
    assert kcfd.load_records(tmp_path / "missing.json") is None and kcfd.load_records(path, "4Q-2") is None
    corr = kcfd.CFDAirframe.from_records(doc["records"], VEH)
    assert corr is not None and len(corr.samples) == 1
    sw = kcfd.CFDAirframe.from_records(doc["records"], VEH, source="sweep")
    assert sw is not None and len(sw.samples) == len(kcfd.SWEEP_ALPHAS)
    # a second campaign in the same file; a NOT RUN pass never erases a result
    T2 = T.copy(); T2["status"] = "NOT RUN"
    assert len(kcfd.save_records(T2, path, key="4Q-1")) == len(recs) and len(kcfd.load_records(path, "4Q-1")["records"]) == len(recs)
    kcfd.save_records(T2, path, key="4Q-2", keep_previous=False)
    assert kcfd.load_records(path, "4Q-2")["records"] == [] and sorted(kcfd.load_records(path)["campaigns"]) == ["4Q-1", "4Q-2"]
    # the plug-in: the blown correction first, the sweep when none solved blown, nothing when nothing solved
    pi = kcfd.plug_in(VEH, path, "4Q-1")
    assert pi["source"] == "disks" and pi["airframe"] is not None
    assert kcfd.plug_in(VEH, path, "4Q-2")["airframe"] is None and kcfd.plug_in(VEH, tmp_path / "none.json")["campaign"] is None
    rt = kcfd.records_table(doc["records"])
    assert len(kcfd.sweep_comparison(rt, VEH)) == len(kcfd.SWEEP_ALPHAS)


def test_a_diverging_airframe_is_a_failed_transition_not_a_crash(P):
    import kestrel_cfd as kcfd
    bad = kcfd.CFDAirframe([{"alpha": 0.5, "mu": 0.5, "dCL": 1.5e94, "dCD": -4.9e94, "dCm": -6.1e94}])
    veh = kt.make_vehicle(P, airframe=bad, name="garbage")
    ep = kt.simulate(veh, "forward", t_tr=3.0, t_after=1.0)
    m = ep.metrics
    assert np.isfinite(m["diverged at [s]"]) and not m["pass"]
    df, _ = kt.matrix({"garbage": veh}, kinds=("forward",), durations=(3.0,), headwinds=(0.0,), t_after=1.0)
    v = kt.verdict(df)["garbage"]
    assert v["passes"] == 0 and "diverged" in v["failures"][0]


# ----------------------------------------------------------------------------------------------- the design loop
def test_envelope_checks_and_the_iteration_log(CI):
    import kestrel_loop as kl
    e = kl.envelope(mass_kg=2.0)
    assert e["wing area [m²]"] == pytest.approx(0.28, rel=1e-6) and 0.4 < e["cruise CL"] < 0.5 and 10.5 < e["stall speed [m/s]"] < 12.0
    assert 150 < e["hover power, momentum theory [W]"] < 300 and 15 < e["endurance at the cruise [min]"] < 30 and 0.08 < e["static margin [chords]"] < 0.14
    # the envelope lands within a third of the tools on the numbers both compute
    assert e["take-off mass [kg]"] == pytest.approx(CI["mass_kg"], rel=0.35)
    c = [kl.check("take-off mass [kg]", 1.9), kl.check("static margin [chords]", 0.2), kl.check("own", 3.0, "≥", 2.0, unit="N", note="x"), kl.check("flag", True, "==", True)]
    assert [r["holds"] for r in c] == [True, False, True, True] and c[2]["value"] == "3 N" and c[1]["target"] == "0.05 – 0.15"
    t = kl.check_table(c, "t")
    assert t.attrs["passes"] == 3 and t.attrs["of"] == 4 and list(t["holds"]) == ["✔", "✘", "✔", "✔"] and kl.failing(c) == ["static margin [chords]"]
    assert kl.check("nan", float("nan"), "≤", 1.0)["holds"] is False
    log = kl.IterationLog()
    log.add("0", changed="—", checks=c, **{"take-off mass [kg]": 2.0, "Cd0": 0.067})
    log.add("1", changed="battery", **{"take-off mass [kg]": 1.69, "Cd0": 0.067, "transitions passing": "11 of 12"})
    f = log.frame()
    assert list(f.index) == ["0", "1"] and f.loc["0", "checks holding"] == "3 / 4" and f.loc["1", "transitions passing"] == "11 of 12"
    assert isinstance(log.to_dict()[0]["take-off mass [kg]"], float)
