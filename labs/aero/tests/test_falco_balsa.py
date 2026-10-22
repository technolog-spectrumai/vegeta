"""FALCO's built-up balsa wing and tail (notebook 36): the stations, the ribs against the design's section and tube
positions, the sheets and stock, the mass against the foam wing, the torsion without the foam, the cut jobs."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "designs"))

import falco
import falco_balsa as fb
import nisus_plus_balsa as nb
from fixed_wing import naca4
from vegeta import atropos
from vegeta.atropos.examples import D1_PRO_BALSA_2MM, D1_PRO_PLY_2MM


@pytest.fixture(scope="module")
def P():
    return falco.resolve()


@pytest.fixture(scope="module")
def L(P):
    return falco.Falco.layout(P)


@pytest.fixture(scope="module")
def stations(P):
    return fb.rib_stations(P)


@pytest.fixture(scope="module")
def wing(P):
    return fb.wing_parts(P)


@pytest.fixture(scope="module")
def tail(P):
    return fb.tail_parts(P)


def test_stations_order_forced_and_chords(P, L, stations):
    ys = [s["y"] for s in stations]
    assert ys == sorted(ys) and ys[0] == L["yc"] and ys[-1] == L["b2"] - 2.0
    kinds = {s["kind"] for s in stations}
    assert {"root rib", "flap end", "joint rib (centre section)", "joint rib (outer panel)", "aileron end", "tip rib", "rib"} <= kinds
    assert [s["y"] for s in stations if s["kind"] == "flap end"] == [P["flap_y0"], P["flap_y1"]]
    gaps = np.diff(ys)
    assert gaps.min() > 0 and gaps.max() <= nb.BALSA["rib_pitch"] + 1e-9
    joint = [s for s in stations if "joint rib" in s["kind"]]
    assert all(s["material"] == "ply" for s in joint) and abs(joint[0]["chord"] - L["chord_joint"]) < 0.2
    for s in stations:
        assert s["chord"] == pytest.approx(P["root_chord"] + (P["tip_chord"] - P["root_chord"]) * max(0.0, (s["y"] - L["yc"]) / (L["b2"] - L["yc"])))
        names = [n for _, _, n in s["tubes"]]
        assert ("spar" in names) == (s["y"] <= P["spar_half_length"]) and ("tip rod" in names) == (s["y"] > P["spar_half_length"])
        assert ("rear tube" in names) == (s["y"] <= P["rear_spar_half_length"])
        if P["flap_y0"] <= s["y"] <= P["flap_y1"]:
            assert s["hinge_frac"] == 1 - P["flap_chord_frac"]
        elif s["y"] >= L["y_aileron0"]:
            assert s["hinge_frac"] == 1 - P["aileron_chord_frac"]
        else:
            assert s["hinge_frac"] is None


def test_rib_holes_sit_on_the_designs_tube_line(P, L, stations):
    """The hole centres (rib coordinates) are the design's ``tube_point`` with the incidence and the dihedral undone."""
    d = falco.Falco()
    a = math.radians(P["wing_incidence_deg"])
    xq = 0.25 * P["root_chord"]
    for st in stations[::4]:
        parts = atropos.make_rib(fb.nb.rib_spec(st, P, "t"))
        f_span = max(0.0, (st["y"] - L["yc"]) / (L["b2"] - L["yc"]))
        x_le = L["le_sweep_tip"] * f_span
        zc = max(0.0, st["y"] - L["yc"]) * math.tan(math.radians(P["dihedral_deg"]))
        for f, dia, name in st["tubes"]:
            X, Z = d.tube_point(P, st["y"], f)
            x_ = xq + (X - xq) * math.cos(a) - Z * math.sin(a)
            z_ = (X - xq) * math.sin(a) + Z * math.cos(a)
            hole = next(h for q in parts for h in q.holes if abs(h.centroid()[0] - f * st["chord"]) < 1e-6)
            cx, cz = hole.centroid()
            assert cx + x_le == pytest.approx(x_, abs=1e-6) and cz + zc == pytest.approx(z_, abs=1e-6)
            assert abs(hole.area()) == pytest.approx(math.pi / 4 * (dia + nb.BALSA["tube_clearance"]) ** 2, rel=2e-3)


def test_wing_parts_both_sides_fit_the_strip(P, stations, wing):
    n_pieces = sum(2 if s["hinge_frac"] else 1 for s in stations)
    assert len(wing) == 2 * n_pieces
    assert {q.meta["side"] for q in wing} == {"R", "L"} and all(q.meta["grain"] == "x" for q in wing)
    assert sum(1 for q in wing if q.meta["material"] == "ply") == 2 * 3           # the root rib and the two joint ribs, both sides
    for q in wing:
        w, h = q.size()
        assert h <= 100.0 - 2 * 4.0 and w <= 400.0 - 2 * 4.0                      # the 400 x 100 strip
        assert q.area() > 0 and q.outer.area() > 0
    up, lo = naca4(P["camber"], P["camber_pos"], P["thickness"], n=60)
    root = next(q for q in wing if q.name == "W01R")
    full = atropos.section_area(up, lo, P["root_chord"])
    assert 0.55 * full < root.area() < 0.95 * full                                # the holes, the skin let in, the stock cuts
    fwd = [q for q in wing if q.name.endswith("-fwd")]
    aft = [q for q in wing if q.name.endswith("-aft")]
    assert len(fwd) == len(aft) and all(q.meta["role"] == "surface" for q in aft)


def test_tail_parts(P, tail):
    stab = [q for q in tail if q.meta["surface"] == "stabiliser"]
    fin = [q for q in tail if q.meta["surface"] == "fin"]
    assert stab and fin and all(q.name.endswith(("-fwd", "-aft")) for q in tail)
    assert sum(1 for q in stab if q.name.endswith("-fwd") and q.meta["side"] == "R") == len({q.meta["station_y"] for q in stab})
    with_spar = [q for q in stab if q.holes and q.name.endswith("-fwd")]
    assert len(with_spar) == 2 * len({q.meta["station_y"] for q in stab})       # the spar runs the whole span
    assert abs(with_spar[0].holes[0].area()) == pytest.approx(math.pi / 4 * (fb.STAB_SPAR["od"] + nb.BALSA["tube_clearance"]) ** 2, rel=2e-3)
    rod = [q for q in fin if q.name.endswith("-fwd") and any(abs(abs(h.area()) - math.pi / 4 * (fb.FIN_ROD["od"] + 0.3) ** 2) < 1 for h in q.holes)]
    assert 0 < len(rod) < len([q for q in fin if q.name.endswith("-fwd")])      # the rod stops short of the tip
    for q in tail:
        assert q.size()[1] <= 92.0


def test_sheeting_stock_and_mass(P, wing, tail):
    sh = fb.sheeting_blanks(P)
    assert set(sh["sheet"]) >= {"D-box", "trailing edge"} and (sh["mass [g]"] > 0).all()
    dbox_root = sh[(sh["piece"] == "centre section") & (sh["sheet"] == "D-box") & (sh["surface"] == "upper")].iloc[0]
    assert dbox_root["width root [mm]"] > 0.33 * P["root_chord"]              # the arc is longer than the chord it covers
    assert dbox_root["width root [mm]"] > dbox_root["width tip [mm]"]
    st = fb.stock_list(P)
    assert (st["mass [g]"] > 0).all() and st["item"].str.startswith("LE stock").sum() == 3
    m = fb.mass_estimate(P, wing=wing, tail=tail)
    assert m.loc["total", "mass [g]"] == pytest.approx(m.drop("total")["mass [g]"].sum())
    assert (m["mass [g]"] > 0).all()
    cmp = fb.mass_comparison(P, m)
    assert cmp.attrs["foam_g"] > 0 and cmp.attrs["balsa_g"] == m.loc["total", "mass [g]"]
    # a built-up balsa wing is lighter than the foam core with its D-box, but not by an order of magnitude
    assert 0.3 < cmp.attrs["balsa_g"] / cmp.attrs["foam_g"] < 1.5


def test_torsion_without_the_foam(P):
    t = fb.torsion_check(P)
    assert t.attrs["T"] > 0 and t.attrs["GJ_box"] > 0 and t.attrs["GJ_tube"] > 0
    assert float(t.loc["balsa D-box margin (shear 1.5 MPa assumed x 0.7)", "value"]) > 0
    assert float(t.loc["spar torsion margin (in-plane shear 40 MPa x 0.8)", "value"]) > 0
    # the 2 mm D-box and the tube without the foam: a little more twist than the foam wing, not a different wing
    assert 0 < t.attrs["twist_foam_deg"] < t.attrs["twist_deg"] < 1.3 * t.attrs["twist_foam_deg"]


def test_cut_plan_writes_sheets(tmp_path, wing, tail):
    res = fb.cut_plan(wing + tail, {"balsa": D1_PRO_BALSA_2MM, "ply": D1_PRO_PLY_2MM}, tmp_path, name="falco")
    assert set(res) == {"balsa", "ply"} and all(r.ok for r in res.values())
    assert res["ply"].metrics["parts"] == 6 and res["ply"].metrics["sheets"] <= 3      # two 300 mm ribs per 100 mm strip
    assert res["balsa"].metrics["parts"] == len(wing) + len(tail) - 6
    assert res["balsa"].metrics["sheets"] >= 5
    for r in res.values():
        for k, v in r.artifacts.items():
            assert v.is_file(), k
        d = atropos.read_gcode(r.artifacts["gcode_01"])
        assert d["max_s"] == 1000 and d["bbox"][2] <= 432 and d["bbox"][3] <= 406
    s = fb.cut_summary(res)
    assert s.loc["balsa", "time [min]"] > 10 and s.loc["balsa", "parts mass [g]"] > 0
