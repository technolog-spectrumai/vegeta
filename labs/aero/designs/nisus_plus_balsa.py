"""NISUS+'s wing and tail built up in balsa for a laser cutter (Atropos), on the aircraft's own carbon tubes.

The foam-core wing becomes ribs on the main spar and the rear carry-through tube, with the D-box sheeting from the
leading edge to just behind the spar, cap strips in the open bay, a sheeted trailing edge, leading- and trailing-edge
stock; the flaps and ailerons are the ribs' aft pieces, split at the hinge line with a bevelled nose. The stabiliser
and the fin get ribs on their spar and rod the same way. Every dimension of the build is in ``BALSA`` (stated
assumptions); the planform, the section, the tube positions and the hinge lines come from the design through the
``design`` seam (``NisusPlus()`` by default, FALCO's instance from ``falco_balsa``).

What the module gives: the rib stations (``rib_stations``), the ribs as Atropos parts (``wing_parts``,
``tail_parts``), the sheeting blanks and the stock as tables (``sheeting_blanks``, ``stock_list``), the balsa mass
against the foam wing's (``mass_estimate``, ``compare_with_foam``), the D-box torsion without the foam
(``torsion_check``) and the cut jobs per material (``cut_plan``).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

import nisus_plus
import nisus_plus_flight as ff
from fixed_wing import naca4
from nisus_structure import ULTIMATE, CARBON
from nisus_systems import MATERIALS, TAIL_SPAR, TIP_RODS

from vegeta import atropos

from ogma.units import RHO0  # noqa: E402  ISA sea-level density: Ogma

#: the build (mm unless said): stated assumptions, not sized here
BALSA = dict(
    rib_t=2.0,                 # wing and tail ribs: 2 mm medium balsa, grain along the chord
    joint_rib="ply",           # the root ribs and the panel-joint ribs: 2 mm birch plywood (the joiner's bearing, the bolts)
    dbox_t=2.0,                # D-box sheeting (the design's torsion check assumes a 2 mm D-box: without the foam, keep it)
    sheet_t=1.5,               # trailing-edge and control-surface sheeting, the tail's sheeting
    dbox_end=0.03,             # the D-box sheet runs to spar_x_frac + this (behind the spar, over a shear web)
    te_sheet_from=0.85,        # the trailing-edge sheeting starts here on unsplit ribs (a split rib's aft piece is sheeted all over)
    cap=(6.0, 2.0),            # cap strips (width, thickness) on the open bay, upper and lower
    cap_fracs=(0.42, 0.63),    # where the cap strips sit (chord fractions)
    le_stock=10.0,             # leading-edge stock: a 10 x 10 mm triangle sanded to the section; the rib's nose is cut back by le_cut
    le_cut=6.0,
    te_stock=(20.0, 6.0),      # trailing-edge stock: 20 x 6 mm tapered strip; the rib's tail is cut back by te_cut
    te_cut=12.0,
    tube_clearance=0.3,        # on the diameter: the glue gap around the tubes
    min_wall=1.0,              # rib left around a tube hole
    split_gap=1.5,             # at the hinge line
    bevel_deg=12.0,            # the control surface's nose wedge, either side
    lightening=dict(fracs=(0.45, 0.60, 0.80), d_max=40.0, wall=5.0),
    rib_pitch=60.0,            # wing rib stations
    tail_rib_pitch=40.0,       # stabiliser and fin
    tail_thickness=0.10,       # the built-up tail's section (NACA 0010): the 12/10 spar does not fit a NACA 0006 stabiliser at 190 mm chord
                               # (11.4 mm thick: the foam plate hides it) nor the 10/8 rod the fin at its 0.45 c with the sheeting let in;
                               # 0010 leaves 1.9 mm of rib around the stabiliser's bore under the sheeting
    glue_share=0.10,           # glue and finishing sanding: 10 % of the wood
    tip_rod_od=TIP_RODS["od"],  # the rods from the spar's end to the tip
)

MATERIAL = {
    "balsa": MATERIALS["balsa"],
    "ply": dict(rho=0.68, E=9000.0, strength=40.0, source="birch aircraft plywood 2 mm, 680 kg/m³ (catalogue class)"),
}


def _design(design=None):
    return nisus_plus._design(design)


def _p(p, design):
    return nisus_plus.resolve(p, design)


def _L(p, design):
    return type(_design(design)).layout(p)


def chord(p, y, L=None):
    L = L or _L(p, None)
    f = max(0.0, (abs(y) - L["yc"]) / (L["b2"] - L["yc"]))
    return p["root_chord"] + (p["tip_chord"] - p["root_chord"]) * f


# ================================================================================================= stations
def rib_stations(p=None, *, pitch=None, design=None, b: dict | None = None) -> list[dict]:
    """The rib stations of one half wing, root to tip: a regular pitch from the centre strip's edge (``yc``), with
    ribs forced at the flap's ends, the panel joint (one rib on each side of it), the aileron's start and the tip; a
    regular station within a third of the pitch of a forced one is dropped. Each station: ``y``, ``chord``,
    ``piece`` (centre section / outer panel), ``hinge_frac`` (None on a plain rib), ``tubes`` (chord fraction,
    diameter, name), ``material``, ``kind``."""
    b = {**BALSA, **(b or {})}
    p = _p(p, design)
    L = _L(p, design)
    pitch = pitch or b["rib_pitch"]
    yc, b2, yj = L["yc"], L["b2"], p["wing_joint_y"]
    y_tip = b2 - 2.0
    forced = [(yc, "root rib", b["joint_rib"]), (p["flap_y0"], "flap end", "balsa"), (p["flap_y1"], "flap end", "balsa"),
              (yj - 0.5, "joint rib (centre section)", b["joint_rib"]), (yj + 0.5, "joint rib (outer panel)", b["joint_rib"]),
              (L["y_aileron0"], "aileron end", "balsa"), (y_tip, "tip rib", "balsa")]
    regular = []
    y = yc + pitch
    while y < y_tip - pitch / 3:
        if all(abs(y - yf) > pitch / 3 for yf, _, _ in forced):
            regular.append((y, "rib", "balsa"))
        y += pitch
    out = []
    for y, kind, material in sorted(forced + regular):
        c = chord(p, y, L)
        hinge = None
        if p["flap_y0"] - 1e-6 <= y <= p["flap_y1"] + 1e-6:
            hinge = 1 - p["flap_chord_frac"]
        elif y >= L["y_aileron0"] - 1e-6:
            hinge = 1 - p["aileron_chord_frac"]
        tubes = []
        if y <= p["spar_half_length"] + 1e-6:
            tubes.append((p["spar_x_frac"], p["spar_od"], "spar"))
        else:
            tubes.append((p["spar_x_frac"], b["tip_rod_od"], "tip rod"))
        if y <= p["rear_spar_half_length"] + 1e-6:
            tubes.append((p["rear_spar_x_frac"], p["rear_spar_od"], "rear tube"))
        out.append(dict(y=float(y), chord=float(c), piece="centre section" if y < yj else "outer panel", hinge_frac=hinge,
                        tubes=tubes, material=material, kind=kind))
    return out


def rib_spec(station: dict, p: dict, name: str, b: dict | None = None, side: str = "R") -> atropos.RibSpec:
    """The Atropos rib of one station: the design's section, the holes on the camber line, the sheeting, the slots,
    the stock cuts, the hinge split."""
    b = {**BALSA, **(b or {})}
    up, lo = naca4(p["camber"], p["camber_pos"], p["thickness"], n=60)
    c = station["chord"]
    hinge = station["hinge_frac"]
    sheeting = [atropos.Sheeting(0.0, p["spar_x_frac"] + b["dbox_end"], b["dbox_t"]),
                atropos.Sheeting(hinge if hinge else b["te_sheet_from"], 1.0, b["sheet_t"])]
    tubes = [atropos.TubeHole(f, d, b["tube_clearance"], z_mm=atropos.camber_z(p["camber"], p["camber_pos"], f) * c,
                              min_wall_mm=b["min_wall"], name=n) for f, d, n in station["tubes"]]
    w, t = b["cap"]
    notches = [atropos.Notch(f, s, w, t) for f in b["cap_fracs"] for s in ("upper", "lower")
               if p["spar_x_frac"] + b["dbox_end"] + 0.02 < f < (hinge or b["te_sheet_from"]) - 0.02]
    lt = b["lightening"]
    lightening = [atropos.Lightening(f, lt["d_max"], lt["wall"]) for f in lt["fracs"]]
    return atropos.RibSpec(name, c, up, lo, sheeting, tubes, lightening, notches, le_cut_mm=b["le_cut"], te_cut_mm=b["te_cut"],
                           split_at_frac=hinge, split_gap_mm=b["split_gap"] if hinge else 0.0, bevel_deg=b["bevel_deg"] if hinge else 0.0,
                           hinge_z_mm=atropos.camber_z(p["camber"], p["camber_pos"], hinge) * c if hinge else None,
                           meta=dict(station_y=station["y"], piece=station["piece"], side=side, kind=station["kind"],
                                     material=station["material"], grain="x", surface="wing"))


def wing_parts(p=None, *, design=None, b: dict | None = None, sides=("R", "L")) -> list[atropos.Part]:
    """Every wing rib piece of both half wings as Atropos parts (a left rib is a right rib turned over)."""
    p = _p(p, design)
    parts = []
    for i, st in enumerate(rib_stations(p, design=design, b=b)):
        for side in sides:
            parts += atropos.make_rib(rib_spec(st, p, f"W{i + 1:02d}{side}", b, side))
    return parts


def station_table(p=None, *, design=None, b: dict | None = None) -> pd.DataFrame:
    """The stations with the rib's size, holes and the thinnest wall around the spar hole."""
    b = {**BALSA, **(b or {})}
    p = _p(p, design)
    up, lo = naca4(p["camber"], p["camber_pos"], p["thickness"], n=60)
    rows = []
    for i, st in enumerate(rib_stations(p, design=design, b=b)):
        parts = atropos.make_rib(rib_spec(st, p, f"W{i + 1:02d}", b))
        f, d, _ = st["tubes"][0]
        zu, zl = atropos.thickness_at(up, lo, st["chord"], f)
        wall = 0.5 * ((zu - zl) - 2 * b["dbox_t"] - (d + b["tube_clearance"]))
        rows.append({"rib": f"W{i + 1:02d}", "y [mm]": st["y"], "chord [mm]": st["chord"], "piece": st["piece"], "kind": st["kind"],
                     "material": st["material"], "hinge": st["hinge_frac"] if st["hinge_frac"] else "",
                     "tubes": ", ".join(f"{n} {d_:g}" for _, d_, n in st["tubes"]), "spar wall [mm]": wall,
                     "pieces": len(parts), "area [mm²]": sum(q.area() for q in parts),
                     "cut [mm]": sum(q.cut_length() for q in parts), "holes": sum(len(q.holes) for q in parts)})
    return pd.DataFrame(rows).set_index("rib")


# ================================================================================================= the tail
def tail_parts(p=None, *, design=None, b: dict | None = None, stab_spar: dict | None = None, fin_rod: dict | None = None,
               sides=("R", "L")) -> list[atropos.Part]:
    """The stabiliser's ribs (both halves, split at the elevator) on its spar and the fin's ribs (split at the
    rudder) on its rod. ``stab_spar``: ``dict(od, x_frac, half_length)``; ``fin_rod``: ``dict(od, x_frac, length)``
    or None (no rod: NISUS+'s boom fins). Both sections are ``BALSA['tail_thickness']`` (see there)."""
    b = {**BALSA, **(b or {})}
    p = _p(p, design)
    L = _L(p, design)
    stab_spar = stab_spar or dict(od=TAIL_SPAR["od"], x_frac=0.30, half_length=TAIL_SPAR["length"] / 2)
    pitch = b["tail_rib_pitch"]
    parts = []
    up, lo = naca4(0.0, 0.4, b["tail_thickness"], n=50)
    c = p["tail_chord"]
    hinge = 1 - p["elevator_frac"]
    ys = list(np.arange(0.0, p["tail_span"] / 2 - pitch / 3, pitch)) + [p["tail_span"] / 2 - 2.0]
    for i, y in enumerate(ys):
        tubes = [atropos.TubeHole(stab_spar["x_frac"], stab_spar["od"], b["tube_clearance"], min_wall_mm=b["min_wall"], name="stab spar")] \
            if y <= stab_spar["half_length"] + 1e-6 else []
        spec = atropos.RibSpec("", c, up, lo, [atropos.Sheeting(0.0, stab_spar["x_frac"] + b["dbox_end"], b["sheet_t"]), atropos.Sheeting(hinge, 1.0, b["sheet_t"])],
                               tubes, [atropos.Lightening(0.48, 20.0, 4.0)], [], le_cut_mm=b["le_cut"] * 0.6, te_cut_mm=b["te_cut"] * 0.6,
                               split_at_frac=hinge, split_gap_mm=b["split_gap"], bevel_deg=b["bevel_deg"],
                               meta=dict(station_y=float(y), piece="stabiliser", kind="rib", material="balsa", grain="x", surface="stabiliser"))
        for side in sides:
            spec.name = f"S{i + 1:02d}{side}"
            spec.meta = dict(spec.meta, side=side)
            parts += atropos.make_rib(spec)
    c = p["fin_chord"]
    hinge = 1 - p["rudder_frac"]
    zs = list(np.arange(0.0, p["fin_height"] - pitch / 3, pitch)) + [p["fin_height"] - 2.0]
    for k in range(int(L.get("n_fins", 1))):
        for i, z in enumerate(zs):
            tubes = [atropos.TubeHole(fin_rod["x_frac"], fin_rod["od"], b["tube_clearance"], min_wall_mm=b["min_wall"], name="fin rod")] \
                if fin_rod and z <= fin_rod["length"] + 1e-6 else []
            spec = atropos.RibSpec(f"F{i + 1:02d}" + (chr(ord("A") + k) if L.get("n_fins", 1) > 1 else ""), c, up, lo,
                                   [atropos.Sheeting(0.0, (fin_rod["x_frac"] if fin_rod else 0.30) + b["dbox_end"], b["sheet_t"]), atropos.Sheeting(hinge, 1.0, b["sheet_t"])],
                                   tubes, [atropos.Lightening(0.25, 20.0, 4.0), atropos.Lightening(0.55, 20.0, 4.0)], [],
                                   le_cut_mm=b["le_cut"] * 0.6, te_cut_mm=b["te_cut"] * 0.6,
                                   split_at_frac=hinge, split_gap_mm=b["split_gap"], bevel_deg=b["bevel_deg"],
                                   meta=dict(station_z=float(z), piece="fin", kind="rib", material="balsa", grain="x", surface="fin", side=""))
            parts += atropos.make_rib(spec)
    return parts


# ================================================================================================= sheets and stock
def sheeting_blanks(p=None, *, design=None, b: dict | None = None) -> pd.DataFrame:
    """The sheeting as flat blanks per wing piece and surface: the D-box (upper and lower, from the leading edge to
    its end) and the trailing edge (from ``te_sheet_from`` or the hinge), each a trapezoid whose widths are the arc
    lengths at the piece's ends and whose length is the piece's span. Mass from area x thickness x density."""
    b = {**BALSA, **(b or {})}
    p = _p(p, design)
    L = _L(p, design)
    up, lo = naca4(p["camber"], p["camber_pos"], p["thickness"], n=80)
    rho = MATERIAL["balsa"]["rho"]
    x_db = p["spar_x_frac"] + b["dbox_end"]
    spans = [("centre section", L["yc"], p["wing_joint_y"]), ("outer panel", p["wing_joint_y"], L["b2"])]
    rows = []
    for piece, y0, y1 in spans:
        c0, c1 = chord(p, y0, L), chord(p, y1, L)
        length = (y1 - y0) / math.cos(math.radians(p["dihedral_deg"]))
        for surf, curve in (("upper", up), ("lower", lo)):
            for what, f0, f1, t in (("D-box", 0.0, x_db, b["dbox_t"]), ("trailing edge", b["te_sheet_from"], 1.0, b["sheet_t"])):
                w0, w1 = atropos.arc_length(curve, c0, f0, f1), atropos.arc_length(curve, c1, f0, f1)
                area = 0.5 * (w0 + w1) * length
                rows.append({"piece": piece, "sheet": what, "surface": surf, "y0 [mm]": y0, "y1 [mm]": y1, "length [mm]": length,
                             "width root [mm]": w0, "width tip [mm]": w1, "thickness [mm]": t, "area [cm²]": area / 100,
                             "mass [g]": 2 * area * t * 1e-3 * rho, "count": 2})
    out = pd.DataFrame(rows)
    a_db = (atropos.arc_length(up, p["root_chord"], 0, x_db) + atropos.arc_length(lo, p["root_chord"], 0, x_db)) * 2 * L["yc"]
    a_te = (atropos.arc_length(up, p["root_chord"], b["te_sheet_from"], 1) + atropos.arc_length(lo, p["root_chord"], b["te_sheet_from"], 1)) * 2 * L["yc"]
    out.loc[len(out)] = {"piece": "centre strip", "sheet": "D-box + TE, both surfaces", "surface": "both", "y0 [mm]": -L["yc"], "y1 [mm]": L["yc"],
                         "length [mm]": 2 * L["yc"], "width root [mm]": np.nan, "width tip [mm]": np.nan, "thickness [mm]": b["dbox_t"],
                         "area [cm²]": (a_db + a_te) / 100, "mass [g]": (a_db * b["dbox_t"] + a_te * b["sheet_t"]) * 1e-3 * rho, "count": 1}
    return out


def stock_list(p=None, *, design=None, b: dict | None = None) -> pd.DataFrame:
    """Leading- and trailing-edge stock and the cap strips per wing piece (both sides), with their mass."""
    b = {**BALSA, **(b or {})}
    p = _p(p, design)
    L = _L(p, design)
    rho = MATERIAL["balsa"]["rho"]
    rows = []
    cos_d = math.cos(math.radians(p["dihedral_deg"]))
    spans = [("centre strip", -L["yc"], L["yc"], 1.0), ("centre section", L["yc"], p["wing_joint_y"], cos_d), ("outer panel", p["wing_joint_y"], L["b2"], cos_d)]
    for piece, y0, y1, cs in spans:
        length = (y1 - y0) / cs
        n = 1 if piece == "centre strip" else 2
        a_le = 0.5 * b["le_stock"] ** 2
        rows.append({"piece": piece, "item": f"LE stock {b['le_stock']:g} x {b['le_stock']:g} triangle", "length [mm]": length, "count": n,
                     "section [mm²]": a_le, "mass [g]": n * a_le * length * 1e-3 * rho})
        w, t = b["te_stock"]
        a_te = 0.5 * w * t
        rows.append({"piece": piece, "item": f"TE stock {w:g} x {t:g} tapered", "length [mm]": length, "count": n,
                     "section [mm²]": a_te, "mass [g]": n * a_te * length * 1e-3 * rho})
        if piece != "centre strip":
            w, t = b["cap"]
            n_caps = n * 2 * len(b["cap_fracs"])
            rows.append({"piece": piece, "item": f"cap strips {w:g} x {t:g}", "length [mm]": length, "count": n_caps,
                         "section [mm²]": w * t, "mass [g]": n_caps * w * t * length * 1e-3 * rho})
            web_h = 0.8 * p["thickness"] * 0.5 * (chord(p, y0, L) + chord(p, y1, L))
            rows.append({"piece": piece, "item": f"shear web behind the spar, {b['rib_t']:g} mm, grain vertical", "length [mm]": length, "count": n,
                         "section [mm²]": web_h * b["rib_t"], "mass [g]": n * web_h * b["rib_t"] * length * 1e-3 * rho})
    return pd.DataFrame(rows)


# ================================================================================================= masses
def mass_estimate(p=None, *, design=None, b: dict | None = None, wing: list | None = None, tail: list | None = None) -> pd.DataFrame:
    """The built-up wing's and tail's wood by item: ribs (from the parts' areas), sheeting, stock, glue."""
    b = {**BALSA, **(b or {})}
    p = _p(p, design)
    wing = wing if wing is not None else wing_parts(p, design=design, b=b)
    tail = tail if tail is not None else tail_parts(p, design=design, b=b)
    rows = []
    for mat in ("balsa", "ply"):
        a = sum(q.area() for q in wing if q.meta.get("material") == mat)
        if a:
            rows.append(("wing ribs (%s %g mm)" % (mat, b["rib_t"]), sum(1 for q in wing if q.meta.get("material") == mat),
                         a * b["rib_t"] * 1e-3 * MATERIAL[mat]["rho"]))
    a_tail = sum(q.area() for q in tail)
    rows.append((f"tail ribs (balsa {b['rib_t']:g} mm)", len(tail), a_tail * b["rib_t"] * 1e-3 * MATERIAL["balsa"]["rho"]))
    sh = sheeting_blanks(p, design=design, b=b)
    rows.append((f"wing sheeting (D-box {b['dbox_t']:g} mm, trailing edge {b['sheet_t']:g} mm)", int(sh["count"].sum()), float(sh["mass [g]"].sum())))
    st = stock_list(p, design=design, b=b)
    for item, g in st.groupby(st["item"].str.split(",").str[0])["mass [g]"].sum().items():
        rows.append((f"wing {item}", int(st[st["item"].str.startswith(item.split(" ")[0])]["count"].sum()), float(g)))
    # the tail's sheeting and stock: the same recipe on its areas (D-box + TE over 65 % of the surfaces' area)
    L = _L(p, design)
    tail_area = 2 * (L["S_h"] + L["S_v"]) * 1e6 * 0.65
    rows.append((f"tail sheeting {b['sheet_t']:g} mm and stock", 1, tail_area * b["sheet_t"] * 1e-3 * MATERIAL["balsa"]["rho"] * 1.2))
    df = pd.DataFrame(rows, columns=["item", "count", "mass [g]"])
    wood = df["mass [g]"].sum()
    df.loc[len(df)] = ("glue, sanding (%g %% of the wood)" % (100 * b["glue_share"]), 1, wood * b["glue_share"])
    df.loc[len(df)] = ("total", int(df["count"].sum()), df["mass [g]"].sum())
    return df.set_index("item")


def compare_with_foam(foam_rows: dict[str, float], balsa: pd.DataFrame) -> pd.DataFrame:
    """The foam wing's rows that the built-up wing replaces (name -> g) against the balsa estimate."""
    foam = sum(foam_rows.values())
    built = float(balsa.loc["total", "mass [g]"])
    rows = [(k, v, "foam wing") for k, v in foam_rows.items()] + [("built-up wing and tail (balsa, ply, glue)", built, "balsa wing")]
    df = pd.DataFrame(rows, columns=["item", "mass [g]", "wing"]).set_index("item")
    df.attrs.update(foam_g=foam, balsa_g=built, delta_g=built - foam)
    return df


# ================================================================================================= torsion
def torsion_check(p=None, cases: pd.DataFrame = None, *, design=None, b: dict | None = None) -> pd.DataFrame:
    """The wing's torsion at V_NE without the foam core: the sheeted D-box (``dbox_t``) and the spar tube carry the
    torque that ``nisus_plus_structure.wing_hand`` shares with the foam; the balsa's shear, the tube's, the tip twist,
    against the foam wing's twist from the same formula (its 2 mm D-box, the lightened foam, the tube).
    ``cases``: the load cases (for ``V_NE``); FALCO's from ``falco_structure.load_cases``."""
    import nisus_plus_structure as st
    b = {**BALSA, **(b or {})}
    p = _p(p, design)
    L = _L(p, design)
    c = cases if cases is not None else st.load_cases(p, design=design)
    q_ne = 0.5 * RHO0 * c.attrs["V_NE"] ** 2
    cmac = L["mac"] / 1000
    T = q_ne * L["S_ref"] / 2 * cmac * (abs(ff.AIRFOIL["cm_ac"]) + 0.15 / 3) * ULTIMATE
    c_root = p["root_chord"] / 1000
    t_c = p["thickness"] * c_root
    x_db = p["spar_x_frac"] + b["dbox_end"]
    A_box = x_db * c_root * t_c * 0.70
    perim = 2 * (x_db + 0.01) * c_root + t_c
    t_balsa = b["dbox_t"] * 1e-3
    G_balsa, G_tube, G_foam = 150e6, 5e9, 8e6
    GJ_box = G_balsa * 4 * A_box ** 2 * t_balsa / perim
    do, di = p["spar_od"], p["spar_id"]
    J_tube = 2 * math.pi / 64 * ((do / 1000) ** 4 - (di / 1000) ** 4)
    GJ_tube = G_tube * J_tube
    GJ = GJ_box + GJ_tube
    T_box, T_tube = T * GJ_box / GJ, T * GJ_tube / GJ
    tau_balsa = T_box / (2 * A_box * t_balsa) / 1e6
    tau_tube = T_tube * (do / 2000) / J_tube / 1e6
    twist = math.degrees(T / ULTIMATE * (L["b2"] / 1000) / (2 * GJ))
    # the foam wing by the same formula: its 0.30 c D-box of 2 mm, the lightened core, the tube
    A_f, per_f = 0.30 * c_root * t_c * 0.70, 2 * 0.31 * c_root + t_c
    a_e, b_e = c_root / 2, t_c / 2
    GJ_foam_wing = (G_balsa * 4 * A_f ** 2 * 2e-3 / per_f + G_foam * math.pi * a_e ** 3 * b_e ** 3 / (a_e ** 2 + b_e ** 2) * (1 - st.FOAM_LIGHTENING_SHARE)
                    + GJ_tube)
    twist_foam = math.degrees(T / ULTIMATE * (L["b2"] / 1000) / (2 * GJ_foam_wing))
    rows = {"torque per wing at V_NE, ultimate [N m]": T,
            "torsional stiffness shares (D-box / spar), no foam": f"{GJ_box / GJ:.2f} / {GJ_tube / GJ:.2f}",
            f"balsa D-box shear, {b['dbox_t']:g} mm sheet [MPa]": tau_balsa,
            "balsa D-box margin (shear 1.5 MPa assumed x 0.7)": 1.5 * 0.7 / tau_balsa - 1,
            "spar tube torsion shear [MPa]": tau_tube, "spar torsion margin (in-plane shear 40 MPa x 0.8)": 40 * 0.8 / tau_tube - 1,
            "wing tip twist at V_NE, limit [deg]": twist,
            "the foam wing's twist by the same formula [deg]": twist_foam,
            "twist verdict": ("as stiff as the foam wing" if twist <= 1.1 * twist_foam else
                              f"{twist / twist_foam:.2f} x the foam wing's twist: stiffen the D-box (sheet thicker or further aft)")}
    df = pd.DataFrame({"value": rows}).rename_axis("check")
    df.attrs.update(T=T, GJ_box=GJ_box, GJ_tube=GJ_tube, twist_deg=twist, twist_foam_deg=twist_foam, tau_balsa=tau_balsa)
    return df


# ================================================================================================= cut jobs
def cut_plan(parts: list, settings_by_material: dict[str, atropos.CutSettings], out_dir: str | Path, *, name: str = "wing",
             gcode: bool = True, png: bool = True) -> dict[str, atropos.Result]:
    """One Atropos job per material (``Part.meta['material']``): the ribs on their sheets, SVG and G-code per sheet."""
    out = Path(out_dir)
    results = {}
    for mat, settings in settings_by_material.items():
        sel = [q for q in parts if q.meta.get("material", "balsa") == mat]
        if sel:
            results[mat] = atropos.cut_job(sel, settings, out / mat, name=f"{name} {mat}", gcode=gcode, png=png)
    return results


def cut_summary(results: dict[str, atropos.Result]) -> pd.DataFrame:
    rows = []
    for mat, r in results.items():
        m = r.metrics
        rows.append({"material": mat, "status": r.status, "parts": m.get("parts"), "sheets": m.get("sheets"),
                     "cut length [m]": m.get("cut_length_mm", 0) / 1000, "travel [m]": m.get("travel_length_mm", 0) / 1000,
                     "time [min]": m.get("time_s", 0) / 60, "utilisation": m.get("utilisation"),
                     "parts mass [g]": m.get("parts_mass_g"), "sheets mass [g]": m.get("sheets_mass_g")})
    return pd.DataFrame(rows).set_index("material")
