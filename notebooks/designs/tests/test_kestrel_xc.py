"""Kestrel-X-C (notebook 48): the layout rules, the blended fuselage, the drag build-up's thickness and chord trades,
the vehicle in cruise, the CFD specs and the FEA case plan, the workflow adapter's fingerprint."""
import math

import pytest

import kestrel_xc as xc
import kestrel_xc_campaigns as xcc


@pytest.fixture(scope="module")
def LAY():
    return {v: xc.layout(v) for v in xc.VARIANTS}


def test_layout_rules_hold_and_the_variants_differ_as_named(LAY):
    for v, L in LAY.items():
        assert xc.checks(L) == [], (v, xc.checks(L))
        assert L["prop_beyond_tip"] < 0.03 and L["prop_to_pod"] >= 0.025                 # the disc may overhang the tip pod by a little
        assert abs(L["root_chord"] + L["tip_chord"] - 2 * L["chord"]) < 1e-9             # the taper keeps the area
        assert L["x_te"] < xc.POD_LENGTH
    assert LAY["X-C45"]["phi_deg"] == 45.0 and LAY["X-C"]["phi_deg"] == 30.0
    assert LAY["X-CX"]["chord"] > LAY["X-C"]["chord"] and LAY["X-CX"]["span_projected"] == LAY["X-C"]["span_projected"]
    assert LAY["X-C"]["h"] < LAY["X-C45"]["h"]                                          # the shallower X's hover lever


def test_fuselage_geometry_and_the_build_up():
    f = xc.fuselage_geometry()
    assert 4.0 < f["fineness"] < 5.0 and 0.30 < f["wetted_m2"] < 0.40
    assert f["volume_m3"] > xc.MODULE[0] * xc.MODULE[1] * xc.MODULE[2] * 1.5            # the module and the battery fit inside
    L = xc.layout("X-C")
    d = xc.drag_buildup(L)
    assert set(d["parts_m2"]) >= {"panels (4)", "fuselage with the module inside", "camera window (flush)", "wing-pod junctions (4, filleted at 30°)"}
    assert "pod" not in d["parts_m2"] and "camera dome" not in d["parts_m2"]
    assert 0.020 < d["Cd0"] < 0.030
    # thicker panels: more drag, monotonically; the chord trade: more area, a lower aspect ratio, a higher k
    cd = [xc.drag_buildup(L, t=t)["CdA_m2"] for t in (0.10, 0.12, 0.15, 0.18)]
    assert cd == sorted(cd) and cd[-1] / cd[1] < 1.10
    a, ax = xc.aero(L), xc.aero(xc.layout("X-CX"))
    assert ax["k"] > a["k"] and ax["AR_projected"] < a["AR_projected"]


def test_thickness_buys_spar_depth_only():
    L = xc.layout("X-C")
    thin, thick = xc.cantilever(L, mass_kg=1.8, t=0.12), xc.cantilever(L, mass_kg=1.8, t=0.18)
    assert thin["sized by"].startswith("section depth")
    assert thick["spar OD [mm]"] >= thin["spar OD [mm]"] and thick["first bending, with the tip motor [Hz]"] >= thin["first bending, with the tip motor [Hz]"]


def test_vehicle_cruises_and_the_chord_trade_shows():
    out = {}
    for v in ("X-C", "X-CX"):
        veh = xc.make_vehicle(v)
        cr = xc.cruise(veh)
        ok = cr[cr["feasible"]]
        assert len(ok) > 5 and ok["endurance [min]"].max() > 30
        out[v] = (float(cr.loc[16.0, "L/D"]), float(ok["endurance [min]"].max()), veh.pol.cl_max * math.cos(veh.L["phi"]) * veh.L["S_ref"])
    assert out["X-C"][0] > out["X-CX"][0] and out["X-C"][1] > out["X-CX"][1]            # the aspect ratio wins the cruise
    assert out["X-CX"][2] > out["X-C"][2]                                              # the area wins the stall


def test_cfd_specs_and_fea_plan():
    sw, st = xc.cfd_specs("X-C", "sweep"), xc.cfd_specs("X-C", "stall")
    assert len(sw) == len(xc.SWEEP_ALPHAS) and len(st) == len(xc.STALL_ALPHAS)
    assert all(s["V"] == xc.SWEEP_V for s in sw.values())
    L = xc.layout("X-C")
    br = xc.panel_breaks(L, 12.0)
    assert br[0] == pytest.approx(L["r0"] * 1e3) and br[-1] == pytest.approx(L["r1"] * 1e3) and len(br) == xc.N_SEG + 3


def test_adapter_fingerprint_and_compare():
    fp = xcc.fingerprint("X-C")
    assert fp["book"] == "kestrel-x" and fp["variant"] == "X-C"
    assert xcc.compare(fp, fp)["level"] == "compatible"
    other = xcc.fingerprint("X-CX")
    assert xcc.compare(fp, other)["level"] != "compatible"
    with pytest.raises(ValueError):
        xcc.params("X-Z")


@pytest.mark.slow
def test_cad_builds_every_part():
    from vegeta import dedalus
    design = dedalus.load_design("designs/kestrel_xc.py:KestrelXC")
    for part in ("pod", "panel", "module"):
        g = design.generate(**xc.cad_params("X-C"), part=part)
        assert g.shape.isValid()
    bb = design.generate(**xc.cad_params("X-C"), part="pod").shape.BoundingBox()
    assert abs(bb.xlen - xc.POD_LENGTH * 1e3) < 2 and abs(bb.zlen - xc.POD_HEIGHT * 1e3) < 2
