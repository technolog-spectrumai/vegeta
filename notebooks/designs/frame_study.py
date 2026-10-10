"""Tail frames for the mountain aircraft: NISUS+'s twin booms against a single tail — a structural comparison study,
with each variant's aerodynamic efficiency and its speed and acceleration marked beside the structure (notebook 34).

The question. FALCO, the tractor redesign, wants a large propeller and a simple tail. Before drawing it: how does
NISUS+'s twin-boom frame compare with a single tail — in stress, stiffness and the first modes, in what the tail puts
into the wing, in mass and in drag — and could NISUS+ itself carry the large propeller as a pusher between booms moved
apart (and if that makes no sense, say so)?

The variants, all on NISUS+'s wing, pod and tail arm, with the tail volume coefficients kept (the same stability):

- ``twin 15"``: NISUS+ as built — booms 500 mm apart, 20/18 pultruded tubes, the APC 15x8 between them.
- ``twin 18"``, ``twin 20"``, ``twin 22"``: the booms moved apart for the larger pusher at NISUS+'s propeller clearance;
  the stabiliser spans the booms (its chord shrinks to keep the area), the rear carry-through tube reaches the
  fittings, the flaps start outboard of the fittings (crow loses span), the rear tube is resized for the longer arm.
- ``single 20"``: one roll-wrapped tail tube out of the pod's tail cone to a conventional tail of the same volumes —
  the tractor's frame. A pusher cannot have it: its propeller sits where the tube goes.

Structure. Each frame's carbon members in Talos — the booms or the tube, the stabiliser's spar tying them (NISUS+'s
6/5 across the booms; a 12/10 cantilever spar on the single tube: its stabiliser is a cantilever, the twin's a beam on
two supports), a rod carrying each fin's side load at the fin's aerodynamic centre — under two ultimate cases: the
symmetric pull-out with the fins' side load, and the asymmetric one (the stabiliser's lift on one half only: a rolling
gust), then the first modes with the tail's masses lumped. The FEA material is isotropic (G = E / 2.6 = 46 GPa): the
bending results hold, the torsion of a unidirectional tube does not (G ~5 GPa pultruded, ~20 GPa roll-wrapped), so the
twist comes from hand formulas with those moduli and the FEA's torsional numbers are marked optimistic. What the tail
puts into the wing: NISUS+'s hand checks of the spar and the rear tube at the boom fittings (the tail case) at each
boom spacing; the single tail puts nothing into the spars (its tube is anchored in the pod's keel at the saddle).

Aerodynamics, marked for each variant: the frame's wetted area and parasite drag (NISUS+'s build-up formulas at the
cruise Reynolds numbers), the aircraft's Cd0 with it, L/D, the cruise power, the mission's energy and feasible survey
time (NISUS+'s mission), the propeller's efficiency at cruise and in the climb. Speed and acceleration: the level top
speed, the acceleration at the launch and cruise speeds, the time from 15 to 25 m/s, the best climb — each propeller
on the AT4125 rewound for it (the same copper: R scales with 1/KV², KV chosen for the rated full-throttle current).

    import frame_study as fst
    F = fst.frames()                       # the five variants
    fst.hand_table(F); fst.wing_loading(F); fst.aero_table(F); fst.performance_table(F)
    fst.fea_models(F, workdir)             # Talos cases (static A/B + modes) for the frames that get an FEA
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

import nisus_plus as npl
import nisus_plus_flight as ff
import nisus_plus_structure as st
import nisus_plus_systems as fs
import nisus_systems as ns

__all__ = ["Frame", "frames", "solve_cases", "boom_y_for_prop", "twin_frame", "single_frame", "hand_table", "wing_loading", "aero_table",
           "performance_table", "drive_for_prop", "propeller", "motor_for_prop", "fea_models", "fea_table", "modes_table", "verdict", "sketch"]

G = fs.G
RHO0 = fs.RHO0
DATA = Path(__file__).parent / "data"
G_TUBE = {"pultruded": 5000.0, "roll-wrapped": 20000.0}          # MPa: shear modulus of the tube's wall (catalogue ranges; assumption)
PROPS = {15.0: 8.0, 18.0: 10.0, 20.0: 13.0, 22.0: 12.0}          # APC thin-electric sizes [in]: diameter -> pitch (a pitch speed above the cruise)
STANDARD_TUBES = [(18.0, 14.0), (20.0, 16.0), (22.0, 18.0), (25.0, 21.0)]   # rear carry-through candidates (2 mm wall)
RHO_CARBON = ns.MATERIALS["carbon tube"]["rho"]                   # g/cm³
FOAM = ns.MATERIALS["XPS foam 30"]["rho"]
FILM = ns.MATERIALS["covering film"]["areal_g_m2"]
PETG = ns.MATERIALS["PETG printed"]["rho"]


@dataclass
class Frame:
    """One tail frame on NISUS+'s wing: the propeller it clears, its tubes and its tail; ``p`` the parameter overrides
    that give NISUS+'s layout the frame's stations (the twin variants only: the single tail is not in NISUS+'s CAD)."""
    name: str
    kind: str                                   # 'twin' | 'single'
    prop_in: float
    pitch_in: float
    boom_y: float                               # twin: half the boom spacing [mm]; single: 0
    tube_od: float
    tube_id: float
    tube_kind: str                              # 'pultruded' | 'roll-wrapped'
    x_root: float                               # the socket's exit: the cantilever starts here [mm]
    x_end: float                                # the tube's end [mm]
    tail_span: float
    tail_chord: float
    fin_height: float                           # the fin plate's whole height (dorsal + ventral) [mm]
    fin_chord: float
    n_fins: int
    tie_od: float                               # the stabiliser's spar [mm]
    tie_id: float
    flap_y0: float
    rear_spar_half_length: float
    rear_spar_od: float
    rear_spar_id: float
    feasible: bool = True
    note: str = ""
    p: dict = field(default_factory=dict)
    fin_ventral: float = 0.0                    # the fin's part below the tube [mm] (NISUS+'s fins straddle the boom)

    @property
    def x_tie(self):
        return self.x_end - self.tail_chord + 0.30 * self.tail_chord           # the stabiliser's spar at 30 % of its chord

    @property
    def x_fin(self):
        """The fin's rod: at 50 % of the fin's chord, clear of the stabiliser's spar at 30 % (the torque arm is the rod's height)."""
        return self.x_end - 0.5 * self.fin_chord

    @property
    def x_ac_tail(self):
        return self.x_end - self.tail_chord + 0.25 * self.tail_chord

    @property
    def cantilever(self):
        return self.x_ac_tail - self.x_root

    @property
    def h_fin_ac(self):
        """The fins' side-load centroid above the tube's axis [mm]: a plate fin's centre of pressure at 45 % of its
        height, the dorsal and the ventral part weighted by their areas (NISUS+'s fins straddle the boom: the two
        nearly cancel; the single tail's fin is all dorsal)."""
        d, v = self.fin_height - self.fin_ventral, self.fin_ventral
        return (0.45 * d * d - 0.45 * v * v) / (d + v) + (self.tube_od / 2 if v == 0 else 0.0)

    @property
    def n_tubes(self):
        return 2 if self.kind == "twin" else 1

    def section(self):
        do, di = self.tube_od, self.tube_id
        return {"I": math.pi / 64 * (do ** 4 - di ** 4), "J": math.pi / 32 * (do ** 4 - di ** 4), "A": math.pi / 4 * (do ** 2 - di ** 2)}


# ------------------------------------------------------------------------------------------------------------ variants
def boom_y_for_prop(p, d_in: float, clearance: float | None = None) -> float:
    """The boom spacing/2 that keeps NISUS+'s propeller-to-boom clearance (``clearance``, default the built one) with
    a propeller of ``d_in`` inches: the clearance is measured from the boom's surface to the disc in the propeller's
    plane (the booms sit ``motor_z - boom_z`` below the axis)."""
    p = npl.resolve(p)
    L = npl.NisusPlus.layout(p)
    dz = p["motor_z"] - p["boom_z"]
    c = L["prop_clearance_boom"] if clearance is None else clearance
    R = d_in * 25.4 / 2
    return math.sqrt((R + p["boom_od"] / 2 + c) ** 2 - dz ** 2)


def _tail_areas(p):
    L = npl.NisusPlus.layout(p)
    return L["S_h"] * 1e6, L["S_v_each"] * 1e6            # mm²


def _rear_tube_for(p_over: dict, base_p) -> tuple:
    """The smallest standard rear carry-through tube with a hand margin >= 0.15 in the tail case at this boom spacing."""
    for od, id_ in STANDARD_TUBES:
        p = npl.resolve({**p_over, "rear_spar_od": od, "rear_spar_id": id_})
        h = st.joints_hand(p)
        m = float(h.iloc[:, 0]["rear tube: margin (500 MPa x 0.8)"])
        if m >= 0.15:
            return od, id_, m
    return STANDARD_TUBES[-1][0], STANDARD_TUBES[-1][1], m


def twin_frame(p=None, d_in: float = 15.0, pitch_in: float | None = None, *, fit_rear_tube: bool = True) -> Frame:
    """NISUS+'s twin booms moved apart for a ``d_in`` propeller: the stabiliser spans the booms at the same area, the
    fins keep their area at the stabiliser's chord, the rear tube reaches 50 mm past the fittings, the flaps start
    10 mm outboard of the fittings; the rear tube is resized by hand for the longer arm."""
    p0 = npl.resolve(p)
    pitch = PROPS.get(d_in, 8.0) if pitch_in is None else pitch_in
    S_h, S_v1 = _tail_areas(p0)
    by = p0["boom_y"] if abs(d_in - p0["prop_diameter"] / 25.4) < 1e-6 else boom_y_for_prop(p0, d_in)
    overhang = (p0["tail_span"] - 2 * p0["boom_y"]) / 2
    span = 2 * by + 2 * overhang
    chord = S_h / span
    flap_y0 = by + p0["fitting_width"] / 2 + 10.0
    k_fin = p0["tail_chord"] / chord                                     # the fins keep their area at the stabiliser's chord
    over = {"boom_y": by, "tail_span": span, "tail_chord": chord, "fin_chord": chord, "fin_height": p0["fin_height"] * k_fin, "fin_ventral": p0["fin_ventral"] * k_fin,
            "rear_spar_half_length": by + 50.0, "flap_y0": flap_y0, "prop_diameter": d_in * 25.4}
    note = []
    feasible = True
    if flap_y0 >= p0["flap_y1"] - 60.0:
        feasible = False
        note.append(f"the flap would be {p0['flap_y1'] - flap_y0:.0f} mm long: no crow")
    if over["rear_spar_half_length"] > p0["wing_joint_y"] - 20.0:
        feasible = False
        note.append("the rear tube would cross the panel joint")
    if fit_rear_tube and feasible:
        od, id_, m = _rear_tube_for(over, p0)
        over.update(rear_spar_od=od, rear_spar_id=id_)
        if m < 0.15:
            feasible = False
            note.append(f"no standard rear tube reaches margin 0.15 ({m:+.2f} with {od:g}/{id_:g})")
        m_spar = float(st.joints_hand(npl.resolve(over)).iloc[:, 0]["main spar: margin in the tail case (500 MPa x 0.8)"])
        if m_spar < 0.0:
            feasible = False
            note.append(f"the main spar's tail-case margin {m_spar:+.2f}: the wing's spar would have to grow for the booms")
    else:
        over.update(rear_spar_od=p0["rear_spar_od"], rear_spar_id=p0["rear_spar_id"])
    pv = npl.resolve(over)
    L = npl.NisusPlus.layout(pv)
    s = npl.NisusPlus().fitting_stations(pv)
    return Frame(f'twin {d_in:g}"', "twin", d_in, pitch, by, p0["boom_od"], p0["boom_id"], "pultruded", s["x_te"], L["boom_x1"], span, chord,
                 over["fin_height"] + over["fin_ventral"], chord, 2, fs.TAIL_SPAR["od"], fs.TAIL_SPAR["id_"], flap_y0, over["rear_spar_half_length"],
                 over["rear_spar_od"], over["rear_spar_id"], feasible, "; ".join(note), over, fin_ventral=over["fin_ventral"])


def single_frame(p=None, d_in: float = 20.0, pitch_in: float | None = None, *, tube=(30.0, 28.0), tube_kind: str = "roll-wrapped",
                 tie=(12.0, 10.0)) -> Frame:
    """One tail tube from the pod's tail cone (anchored in the keel from the saddle to the pod's end) to a conventional
    tail of NISUS+'s areas and arm: the stabiliser across the tube, one fin of the two fins' area on top of it."""
    p0 = npl.resolve(p)
    pitch = PROPS.get(d_in, 10.0) if pitch_in is None else pitch_in
    L = npl.NisusPlus.layout(p0)
    S_h, S_v1 = _tail_areas(p0)
    chord = p0["tail_chord"]
    fin_chord = 1.5 * chord                                              # one dorsal fin of the two fins' area, deeper (a belly landing: nothing below)
    return Frame(f'single {d_in:g}"', "single", d_in, pitch, 0.0, tube[0], tube[1], tube_kind, L["x_pod_end"], L["boom_x1"], p0["tail_span"], chord,
                 2 * S_v1 / fin_chord, fin_chord, 1, tie[0], tie[1], p0["flap_y0"], p0["rear_spar_half_length"], p0["rear_spar_od"], p0["rear_spar_id"],
                 True, "the tractor's frame: the tube leaves the pod where the pusher's propeller is", {})


def frames(p=None) -> list:
    """The five variants of the study."""
    return [twin_frame(p, 15.0), twin_frame(p, 18.0), twin_frame(p, 20.0), twin_frame(p, 22.0), single_frame(p, 20.0)]


# ------------------------------------------------------------------------------------------------------------- masses
def frame_mass(f: Frame, p=None) -> dict:
    """The tail-carrying structure [g]: the tubes from the socket's exit to the end, the sockets' share inside the wing
    or the pod, the fittings, the stabiliser (foam, film, spar), the fins (foam, film)."""
    p = npl.resolve(p)
    sec = f.section()
    L_tube = (f.x_end - (p["boom_x0"] if f.kind == "twin" else f.x_root - 200.0))      # the single tube is anchored 200 mm into the keel
    tubes = f.n_tubes * sec["A"] * L_tube * 1e-3 * RHO_CARBON
    if f.kind == "twin":
        fittings = 2 * fs.CAD["boom_fitting"] * 1e-3 * PETG * fs.PRINT_FILL["boom_fitting"] + 2 * fs.CAD["tail_fitting"] * 1e-3 * PETG * fs.PRINT_FILL["tail_fitting"]
    else:
        fittings = fs.CAD["tail_fitting"] * 1e-3 * PETG * fs.PRINT_FILL["tail_fitting"] + 35.0      # one tail fitting + the keel socket (PETG insert)
    t_plate = p["tail_thickness"] * f.tail_chord                                                # the plates' thickness [mm]
    stab = f.tail_span * f.tail_chord * t_plate * 1e-3 * FOAM + FILM * 2 * f.tail_span * f.tail_chord * 1e-6 * 1.02
    tie = math.pi / 4 * (f.tie_od ** 2 - f.tie_id ** 2) * f.tail_span * 1e-3 * RHO_CARBON
    fins = f.n_fins * (f.fin_height * f.fin_chord * t_plate * 1e-3 * FOAM + FILM * 2 * f.fin_height * f.fin_chord * 1e-6 * 1.02)
    total = tubes + fittings + stab + tie + fins
    return {"tubes [g]": tubes, "fittings [g]": fittings, "stabiliser [g]": stab + tie, "fins [g]": fins, "frame [g]": total}


# ---------------------------------------------------------------------------------------------------------- hand checks
def _loads(p=None):
    a = st.load_cases(p).attrs
    return {"F_tail": st.ULTIMATE * a["tail"], "F_fin": st.ULTIMATE * a["fin"], "W": a["W"], "mass": a["mass"]}


def hand_table(F: list, p=None) -> pd.DataFrame:
    """The frames by hand at the ultimate loads: the tube's bending stress and margin, the tail's incidence change and
    the tip deflection under the pull-out, the fin's twist under its side load, the tail's roll under the asymmetric
    case (the twin: differential bending; the single: torsion), the first bending frequency with the tail mass at the
    tip, the frame's mass. Torsion with the tube's own shear modulus (``G_TUBE``)."""
    p = npl.resolve(p)
    ld = _loads(p)
    E = st.CARBON["E"]
    kd = st.CARBON["strength"] * st.KNOCKDOWN["carbon tube"]
    rows = {}
    for f in F:
        s = f.section()
        Lc = f.cantilever
        F_v = ld["F_tail"] / f.n_tubes                       # per tube
        F_s = 2 * ld["F_fin"] / f.n_tubes                   # NISUS+'s two fins' side load, per tube (the single fin carries both)
        M = math.hypot(F_v, F_s) * Lc
        sig = M * f.tube_od / 2 / s["I"]
        defl = F_v * Lc ** 3 / (3 * E * s["I"])
        slope = math.degrees(F_v * Lc ** 2 / (2 * E * s["I"]))
        Gt = G_TUBE[f.tube_kind]
        T_fin = F_s * f.h_fin_ac                              # the fin's torque on its tube
        twist_fin = math.degrees(T_fin * Lc / (Gt * s["J"]))
        if f.kind == "twin":
            dz = (ld["F_tail"] / 2) * Lc ** 3 / (3 * E * s["I"])       # one boom carries the half-stabiliser's lift alone
            roll = math.degrees(dz / (2 * f.boom_y))
            roll_how = "differential bending"
        else:
            T_roll = (ld["F_tail"] / 2) * (f.tail_span / 4)             # the half-stabiliser's lift at its own centroid
            roll = math.degrees((T_roll + T_fin) * Lc / (Gt * s["J"]))
            roll_how = f"torsion (G {Gt / 1000:g} GPa {f.tube_kind})"
        m = frame_mass(f, p)
        m_tip = (m["stabiliser [g]"] + m["fins [g]"] + (m["fittings [g]"] if f.kind == "single" else 2 * fs.CAD["tail_fitting"] * 1e-3 * PETG * fs.PRINT_FILL["tail_fitting"])) * 1e-3 / f.n_tubes
        m_beam = s["A"] * Lc * 1e-3 * RHO_CARBON * 1e-3
        f1 = math.sqrt(3 * E * s["I"] / (Lc ** 3 * (m_tip + 0.24 * m_beam))) / (2 * math.pi) * math.sqrt(1e3)   # N/mm / kg -> 1/s²: x 1000
        rows[f.name] = {"feasible": f.feasible, "propeller [in]": f.prop_in, "booms apart [mm]": 2 * f.boom_y if f.kind == "twin" else np.nan,
                        "tube": f"{f.n_tubes} x {f.tube_od:g}/{f.tube_id:g} {f.tube_kind}", "cantilever [mm]": Lc, "tail span x chord [mm]": f"{f.tail_span:.0f} x {f.tail_chord:.0f}",
                        "fins": f"{f.n_fins} x {f.fin_height:.0f} high", "bending stress, ultimate [MPa]": sig, "margin (500 MPa x 0.8)": kd / sig - 1,
                        "tip deflection at ultimate [mm]": defl, "tail incidence change at ultimate [deg]": slope,
                        "fin twist under its side load [deg]": twist_fin, "tail roll, asymmetric case [deg]": roll, "roll carried by": roll_how,
                        "first bending mode, hand [Hz]": f1, "frame mass [g]": m["frame [g]"], "note": f.note}
    return pd.DataFrame(rows).T


def wing_loading(F: list, p=None) -> pd.DataFrame:
    """What the tail puts into the wing: for the twin frames NISUS+'s hand checks at the boom fittings in the tail case
    (the main spar and the rear tube bent by the boom's reactions at the pod's side, their margins, the rear tube the
    spacing needs, the flap span left for crow); for the single tail the tube's root moment into the pod's keel socket
    and the saddle instead (the spars see only the lift)."""
    p = npl.resolve(p)
    ld = _loads(p)
    rows = {}
    for f in F:
        if f.kind == "twin":
            pv = npl.resolve(f.p)
            h = st.joints_hand(pv).iloc[:, 0]
            rows[f.name] = {"booms apart [mm]": 2 * f.boom_y, "arm from the pod's side [mm]": f.boom_y - npl.NisusPlus.layout(pv)["yc"],
                            "rear tube": f"{f.rear_spar_od:g}/{f.rear_spar_id:g} x {2 * f.rear_spar_half_length:.0f}",
                            "main spar: bending from the boom, tail case [MPa]": h["main spar: bending from the boom (tail case) [MPa]"],
                            "main spar margin": h["main spar: margin in the tail case (500 MPa x 0.8)"],
                            "rear tube: bending at the pod side [MPa]": h["rear tube: bending stress at the pod side, ultimate [MPa]"],
                            "rear tube margin": h["rear tube: margin (500 MPa x 0.8)"],
                            "boom socket: bearing at the exit [N]": h["boom socket: bearing reaction at the exit, ultimate [N]"],
                            "flap span left [mm]": pv["flap_y1"] - f.flap_y0, "crow drag kept [%]": 100 * (pv["flap_y1"] - f.flap_y0) / (p["flap_y1"] - p["flap_y0"]),
                            "into the pod": "nothing: the tail's loads go through the spars"}
        else:
            M = math.hypot(ld["F_tail"], ld["F_fin"] * f.n_fins) * f.cantilever              # N mm at the socket's exit
            Ls = 200.0
            R = M / Ls
            bear = R / (f.tube_od * 25.0)                                                     # on a 25 mm long PETG saddle at each end of the socket
            rows[f.name] = {"booms apart [mm]": np.nan, "arm from the pod's side [mm]": np.nan, "rear tube": f"{f.rear_spar_od:g}/{f.rear_spar_id:g} (NISUS+'s: no tail load on it)",
                            "main spar: bending from the boom, tail case [MPa]": 0.0, "main spar margin": np.nan,
                            "rear tube: bending at the pod side [MPa]": 0.0, "rear tube margin": np.nan,
                            "boom socket: bearing at the exit [N]": R, "flap span left [mm]": p["flap_y1"] - f.flap_y0, "crow drag kept [%]": 100.0,
                            "into the pod": f"{M / 1000:.0f} N m at the keel socket ({Ls:.0f} mm): bearing {bear:.1f} MPa on PETG (45 x 0.7 x 0.5: margin {45 * 0.7 * 0.5 / bear - 1:+.1f}); the saddle's bolts take the couple"}
    return pd.DataFrame(rows).T


# ------------------------------------------------------------------------------------------------------- aerodynamics
def _cf(length_m, V, nu):
    re = V * length_m / nu
    return 0.455 / math.log10(max(re, 1e4)) ** 2.58


def frame_drag(f: Frame, p=None, h_m: float = 3000.0) -> dict:
    """The frame's wetted areas and its parasite drag area [m²] by NISUS+'s build-up (flat plate x form factor, + 10 %
    interference) at the cruise speed and the altitude's viscosity: the tubes outside the wing and the tail, the
    stabiliser, the fins, the fittings as 1.3 x the tubes' local area."""
    p = npl.resolve(p)
    atm = fs.atmosphere(h_m)
    V = ff.V_CRUISE_EAS / math.sqrt(atm["sigma"])
    nu = atm["nu"]
    if f.kind == "twin":
        L_free = (f.x_end - f.tail_chord - (p["root_chord"])) * 1e-3          # from the wing's trailing edge to the stabiliser
    else:
        L_free = (f.x_end - f.tail_chord - f.x_root) * 1e-3                  # from the pod's end to the stabiliser
    d = f.tube_od * 1e-3
    S_tubes = f.n_tubes * math.pi * d * L_free
    fin = L_free / d
    S_tail = 2 * f.tail_span * f.tail_chord * 1e-6 * 1.02
    S_fins = f.n_fins * 2 * f.fin_height * f.fin_chord * 1e-6 * 1.02
    tt = p["tail_thickness"]
    ff_plate = 1 + 2 * tt + 60 * tt ** 4
    parts = {"tubes": _cf(L_free, V, nu) * (1 + 60 / fin ** 3 + fin / 400) * S_tubes,
             "stabiliser": _cf(f.tail_chord * 1e-3, V, nu) * ff_plate * S_tail,
             "fins": _cf(f.fin_chord * 1e-3, V, nu) * ff_plate * S_fins,
             "fittings": 0.3 * _cf(0.06, V, nu) * 1.5 * (f.n_tubes * 2 * math.pi * d * 0.06)}
    total = 1.10 * sum(parts.values())
    return {"wetted tubes [m²]": S_tubes, "wetted tail [m²]": S_tail + S_fins, "wetted frame [m²]": S_tubes + S_tail + S_fins, "parts": parts,
            "cd_area_m2": total, "V": V, "nu": nu}


def _base_frame_cd_area(p, h_m):
    """NISUS+'s own frame share of its build-up (the same formulas): removed before a variant's share is added."""
    return frame_drag(twin_frame(p, 15.0, fit_rear_tube=False), p, h_m)["cd_area_m2"]


def aero_table(F: list, p=None, h_m: float = 3000.0, battery_key: str = fs.DEFAULT_PACK, drives: dict | None = None) -> pd.DataFrame:
    """The aerodynamic efficiency of each variant, marked: the frame's wetted area and drag area, the aircraft's Cd0
    with it, the mass, L/D max, the cruise power at ``V_CRUISE_EAS`` and ``h_m``, the propeller's efficiency at cruise
    and in the full-throttle climb (with each frame's propeller: ``drives``, from ``drive_for_prop``), the mission's
    energy and feasible survey time (NISUS+'s mission plan)."""
    p = npl.resolve(p)
    af0 = ff.airframe(battery_key, p, h_m=h_m)
    S = af0.wing_area_m2
    base_frame = _base_frame_cd_area(p, h_m) / S
    m_base_frame = frame_mass(twin_frame(p, 15.0, fit_rear_tube=False), p)["frame [g]"]
    atm = fs.atmosphere(h_m)
    V = ff.V_CRUISE_EAS / math.sqrt(atm["sigma"])
    pk = fs.pack(battery_key)
    rows = {}
    for f in F:
        d = frame_drag(f, p, h_m)
        cd0 = af0.cd0 - base_frame + d["cd_area_m2"] / S
        m = af0.mass_kg + (frame_mass(f, p)["frame [g]"] - m_base_frame) * 1e-3
        af = replace(af0, cd0=cd0, mass_kg=m)
        afd = ff.airframe_dict(af)
        dr = (drives or {}).get(f.prop_in) or fs.drive()
        env = ff.envelope(af, dr, h_m)
        P_cr, T_cr = fs.level_power(dr, m, cd0, af.aspect_ratio, af.oswald, S, V, h_m)
        eta_cr, eta_cl = _prop_eta(dr, V, T_cr, atm["rho"]), _prop_eta(dr, 16.0 / math.sqrt(atm["sigma"]), None, atm["rho"])
        me = fs.mission_energy(pk, dr, afd)
        rows[f.name] = {"feasible": f.feasible, "propeller": f"{f.prop_in:g}x{f.pitch_in:g}", "frame wetted area [m²]": d["wetted frame [m²]"],
                        "frame drag area [m²]": d["cd_area_m2"], "aircraft Cd0": cd0, "mass [kg]": m, "L/D max": env["L/D_max"],
                        f"cruise power at {ff.V_CRUISE_EAS:g} EAS, {h_m:.0f} m [W]": P_cr, "propeller η at cruise": eta_cr, "propeller η in the climb (full throttle)": eta_cl,
                        "mission energy [Wh]": me["E_used_wh"], "feasible survey [min]": me["t_survey_feasible_s"] / 60, "reserve kept": me["reserve_kept"]}
    return pd.DataFrame(rows).T


def _prop_eta(dr: fs.NisusPlusDrive, V, T, rho):
    """The propeller's efficiency T V / (Q ω) at airspeed V: at thrust T (None: full throttle)."""
    thr = 1.0 if T is None else dr.throttle_for_thrust(V, T, rho)
    r = dr.at(V, thr, rho)
    rows = dr.rows(V, rho)
    Q = float(np.interp(r["rpm"], rows["rpm"], rows["torque"]))
    w = r["rpm"] * math.pi / 30
    return r["thrust"] * V / (Q * w) if Q * w > 0 else np.nan


# ------------------------------------------------------------------------------------------------- propellers, drives
def propeller(d_in: float, pitch_in: float):
    """NISUS+'s generic planform (the 9x6 shape) scaled to a ``d_in`` x ``pitch_in`` thin-electric propeller."""
    from vegeta import boreas
    d, pitch = boreas.inches(d_in, pitch_in)
    k = d_in / 15.0
    return boreas.Propeller.from_pitch(f"APC {d_in:g}x{pitch_in:g}E (generic planform)", d, pitch, blades=2, chord_root_m=0.023 * k, chord_max_m=0.036 * k,
                                       chord_tip_m=0.010 * k, mass_kg=0.046 * k ** 2.5, rotor_mass_kg=0.16 * k ** 2,
                                       notes=f"generic planform scaled x {k:.2f} from the 15x8's; the real blade is not in the model")


def motor_for_prop(prop, *, V=None, I=None):
    """A motor of the AT4125's power class wound for ``prop``: its resistance, no-load current and current limit kept
    (the same 1.55 kW electrical at full throttle, the same copper loss — in practice a larger stator: the 41xx itself
    rewound to a low KV would have more resistance), KV chosen so the static full-throttle point draws the published
    current at the published voltage. Returns (motor, fit) as ``motor_model`` does, with the 15x8's calibration ratios
    kept (the planform's, assumed to carry over)."""
    from vegeta import boreas
    P = fs.PUBLISHED_STATIC
    V = P["voltage"] if V is None else V
    I_t = P["current"] if I is None else I
    m0, fit0 = fs.motor_model()
    af = fs.blade_section()
    kQ = fit0["torque_correction"]

    def static(kv):
        R = m0.resistance_ohm
        kt = 60 / (2 * math.pi * kv)
        lo, hi = 300.0, kv * V
        for _ in range(60):                                              # the rpm where the motor's torque meets the propeller's
            n = 0.5 * (lo + hi)
            Im = (V - n / kv) / R
            Qm = kt * (Im - m0.no_load_current_a)
            Qp = boreas.solve(prop, af, n, 0.0, RHO0).torque * kQ
            lo, hi = (n, hi) if Qm > Qp else (lo, n)
        return n, (V - n / kv) / R, R, kt

    lo, hi = 150.0, P["kv"]
    for _ in range(40):
        kv = 0.5 * (lo + hi)
        n, Im, R, kt = static(kv)
        lo, hi = (kv, hi) if Im < I_t else (lo, kv)
    motor = boreas.Motor(f"AT4125-class KV{kv:.0f}", kv_rpm_per_volt=kv, resistance_ohm=R, no_load_current_a=m0.no_load_current_a,
                         max_current_a=m0.max_current_a, mass_kg=m0.mass_kg, source=f"a motor of the AT4125's power class for the {prop.name}: its R, I0 and current limit, KV for {I_t:.1f} A at {V:.2f} V static (assumption; a larger stator, ~50-100 g more)")
    op = boreas.solve(prop, af, n, 0.0, RHO0)
    fit = {**fit0, "rpm_at_point": n, "resistance_ohm": R, "kt": kt, "kv": kv, "bemt_thrust_n": op.thrust, "static_thrust_n": op.thrust * fit0["thrust_correction"],
           "shaft_power_w": kt * (Im - m0.no_load_current_a) * n * math.pi / 30, "pitch_speed_m_s": n / 60 * prop.describe().get("pitch_m", 0.0) if isinstance(prop.describe(), dict) else np.nan,
           "note": "the 15x8's calibration ratios kept (the generic planform's); the motor of the same power class is an assumption"}
    return motor, fit


def drive_for_prop(d_in: float, pitch_in: float | None = None, *, rebuild: bool = False, **kw) -> fs.NisusPlusDrive:
    """The drive map for a ``d_in`` propeller on the rewound AT4125 (``data/frame_study_drive_<d>x<p>.json``, built
    once: ~400 BEMT solves); the 15x8 is NISUS+'s own map."""
    pitch = PROPS.get(d_in, 8.0) if pitch_in is None else pitch_in
    if abs(d_in - 15.0) < 1e-6 and abs(pitch - 8.0) < 1e-6:
        return fs.drive()
    path = DATA / f"frame_study_drive_{d_in:g}x{pitch:g}.json"
    if path.exists() and not rebuild:
        return fs.NisusPlusDrive.from_json(json.loads(path.read_text()))
    prop = propeller(d_in, pitch)
    motor, fit = motor_for_prop(prop)
    dr = fs.build_drive(prop=prop, motor=motor, fit=fit, **kw)
    path.write_text(json.dumps(dr.to_json()))
    return dr


# ------------------------------------------------------------------------------------------------ speed, acceleration
def performance_table(F: list, p=None, heights=(1200.0, 3000.0), battery_key: str = fs.DEFAULT_PACK, drives: dict | None = None,
                      v_launch: float = 17.0, v_cruise_eas: float = ff.V_CRUISE_EAS) -> pd.DataFrame:
    """Speed and acceleration of each variant (its Cd0 and mass from the frame, its propeller's map) at the heights:
    the level top speed at full throttle (and whether V_NE caps it), the acceleration at the launch speed and at the
    cruise speed in level flight at full throttle, the time from 15 to 25 m/s TAS, the best climb rate."""
    p = npl.resolve(p)
    af0 = ff.airframe(battery_key, p, h_m=heights[0])
    S = af0.wing_area_m2
    base_frame = _base_frame_cd_area(p, heights[0]) / S
    m_base_frame = frame_mass(twin_frame(p, 15.0, fit_rear_tube=False), p)["frame [g]"]
    rows = {}
    for f in F:
        cd0 = af0.cd0 - base_frame + frame_drag(f, p, heights[0])["cd_area_m2"] / S
        m = af0.mass_kg + (frame_mass(f, p)["frame [g]"] - m_base_frame) * 1e-3
        dr = (drives or {}).get(f.prop_in) or fs.drive()
        row = {"feasible": f.feasible, "propeller": f"{f.prop_in:g}x{f.pitch_in:g}", "static thrust, sea level [N]": dr.max_thrust(0.0, RHO0)}
        for h in heights:
            atm = fs.atmosphere(h)
            rho, sig = atm["rho"], atm["sigma"]
            drag = lambda V: fs._drag(m, cd0, af0.aspect_ratio, af0.oswald, S, V, rho)
            excess = lambda V: dr.max_thrust(V, rho) - drag(V)
            lo, hi = 12.0, 60.0
            for _ in range(50):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if excess(mid) > 0 else (lo, mid)
            v_max = lo
            v_ne = ff.V_NE_EAS / math.sqrt(sig)
            Vc = v_cruise_eas / math.sqrt(sig)
            t, V = 0.0, 15.0
            while V < 25.0 and t < 60.0:
                a = excess(V) / m
                if a <= 0.01:
                    t = math.inf
                    break
                V += a * 0.05
                t += 0.05
            Vs = np.linspace(12.0, min(v_max, v_ne), 40)
            roc = max(fs.max_roc(dr, m, cd0, af0.aspect_ratio, af0.oswald, S, float(v), h) for v in Vs)
            tag = f"{h:.0f} m"
            row.update({f"top speed, level [m/s TAS] @{tag}": min(v_max, v_ne), f"top speed capped by V_NE @{tag}": v_max > v_ne,
                        f"acceleration at launch {v_launch:g} m/s [m/s²] @{tag}": excess(v_launch) / m, f"acceleration at cruise {Vc:.0f} m/s TAS [m/s²] @{tag}": excess(Vc) / m,
                        f"15 → 25 m/s [s] @{tag}": t, f"best climb [m/s] @{tag}": roc})
        rows[f.name] = row
    return pd.DataFrame(rows).T


# ---------------------------------------------------------------------------------------------------------------- FEA
def _zt(f: Frame, p) -> float:
    """The stabiliser spar's axis: its top 0.5 mm proud of the tube's top, the rest through the wall and the bore
    (a spar seated tangent on the tube grazes it and leaves sliver elements in the mesh)."""
    return p["boom_z"] + f.tube_od / 2 - f.tie_od / 2 + 0.5


def _frame_cad(f: Frame, p=None):
    """The frame's carbon members as one solid for Talos: the tube(s) from the socket to the end, the stabiliser's spar
    across them at 30 % of its chord (two halves fused without cleaning: each half is its own load region), a rod per
    fin from the tube up to the fin's aerodynamic centre."""
    import cadquery as cq
    p = npl.resolve(p)
    z = p["boom_z"]
    x0 = p["boom_x0"] if f.kind == "twin" else f.x_root - 200.0
    ys = [-f.boom_y, f.boom_y] if f.kind == "twin" else [0.0]
    solid = None
    for y in ys:                                                         # two pieces fused without cleaning: the socket's own faces take the clamp
        for xa, xb in ((x0, f.x_root), (f.x_root, f.x_end)):
            tube = cq.Workplane("YZ").workplane(offset=xa).center(y, z).circle(f.tube_od / 2).circle(f.tube_id / 2).extrude(xb - xa)
            solid = tube if solid is None else solid.union(tube, clean=False)
    zt = _zt(f, p)                                                           # the spar seated through the tubes' wall (no grazing contact: no sliver elements)
    half = f.tail_span / 2
    def tie(length):                                                     # a wall under 1 mm cannot be meshed at this size: a solid rod of the
        w = cq.Workplane("XZ").workplane(offset=0.0).center(f.x_tie, zt).circle(f.tie_od / 2)       # same diameter (2 x the 6/5's stiffness: negligible)
        return (w if f.tie_od - f.tie_id < 2.0 else w.circle(f.tie_id / 2)).extrude(length)
    left, right = tie(half), tie(-half)                                  # XZ's normal is -Y: +length extrudes toward -y
    solid = solid.union(left, clean=False).union(right, clean=False)
    for y in ys:
        post = cq.Workplane("XY").workplane(offset=z).center(f.x_fin, y).circle(8.0).extrude(f.h_fin_ac)    # a stiff rod: it only carries the torque in
        solid = solid.union(post, clean=False)
    return solid


def fea_models(F: list, workdir, p=None, *, element_size: float = 2.5) -> list:
    """Talos cases per frame: ``A`` the symmetric pull-out (the stabiliser's ultimate lift on both halves, each fin's
    side load at its aerodynamic centre), ``B`` the asymmetric one (the lift on the left half only, the fins' side
    load), both with the socket fixed; and a modal model with the tail's masses lumped (the stabiliser on the spar, the
    fins on their rods, the end fittings on the tubes' ends). Returns ``FEACase`` objects (``case.kind`` 'static' or
    'modes'); the STEPs are written in ``workdir/cad``."""
    from vegeta import talos
    p = npl.resolve(p)
    ld = _loads(p)
    workdir = Path(workdir)
    (workdir / "cad").mkdir(parents=True, exist_ok=True)
    out = []
    for f in F:
        if f.kind == "twin" and f.prop_in not in (15.0, 20.0):
            continue                                                             # the 18"/22" booms are the same tubes at another spacing: hand only
        slug = f.name.replace('"', "in").replace(" ", "_")
        step = workdir / "cad" / f"frame_{slug}.step"
        if not step.exists():
            _frame_cad(f, p).val().exportStep(str(step))
        z = p["boom_z"]
        r = f.tube_od / 2 + 0.5
        ys = [-f.boom_y, f.boom_y] if f.kind == "twin" else [0.0]
        x_sock0 = (p["boom_x0"] if f.kind == "twin" else f.x_root - 200.0) - 0.5
        zt = _zt(f, p)
        rt = f.tie_od / 2 + 0.3
        x_fin = f.x_fin
        regions = [talos.SurfacesInBox("socket", (x_sock0, -f.boom_y - r, z - r, f.x_root + 0.3, f.boom_y + r, z + r)),
                   talos.SurfacesInBox("tie_left", (f.x_tie - rt, -f.tail_span / 2 - 0.5, zt - rt, f.x_tie + rt, 0.5, zt + rt)),
                   talos.SurfacesInBox("tie_right", (f.x_tie - rt, -0.5, zt - rt, f.x_tie + rt, f.tail_span / 2 + 0.5, zt + rt)),
                   talos.SurfacesInBox("ends", (f.x_end - 0.3, -f.boom_y - r, z - r, f.x_end + 0.3, f.boom_y + r, z + r))]
        for i, y in enumerate(ys):
            regions.append(talos.SurfacesInBox(f"post_{i}", (x_fin - 8.5, y - 8.5, z + f.h_fin_ac - 0.3, x_fin + 8.5, y + 8.5, z + f.h_fin_ac + 0.3)))
        F_half = ld["F_tail"] / 2
        F_fin = 2 * ld["F_fin"] / len(ys)                                        # NISUS+'s two fins' side load, per rod (the single fin carries both)
        fins = [talos.Force(f"post_{i}", fy=F_fin) for i in range(len(ys))]
        mesh = talos.MeshSettings(element_size=element_size, order=2)
        mat = st.talos_material(st.CARBON)
        m = frame_mass(f, p)
        masses = [talos.PointMass("tie_left", 0.5 * m["stabiliser [g]"] * 1e-6), talos.PointMass("tie_right", 0.5 * m["stabiliser [g]"] * 1e-6),
                  talos.PointMass("ends", (2 if f.kind == "twin" else 1) * fs.CAD["tail_fitting"] * 1e-3 * PETG * fs.PRINT_FILL["tail_fitting"] * 1e-6)]
        masses += [talos.PointMass(f"post_{i}", m["fins [g]"] / len(ys) * 1e-6) for i in range(len(ys))]
        cases = {"A": [talos.Force("tie_left", fz=F_half), talos.Force("tie_right", fz=F_half)] + fins,
                 "B": [talos.Force("tie_left", fz=F_half)] + fins}
        for key, loads in cases.items():
            desc = {"A": f"symmetric pull-out: stabiliser {ld['F_tail']:.0f} N up, fins {F_fin:.0f} N side each (ultimate)",
                    "B": f"asymmetric: the left half's {F_half:.0f} N only, fins {F_fin:.0f} N side (ultimate)"}[key]
            case = st.FEACase(f"{slug}_{key}", "frame", f"{f.name}: {desc}",
                              talos.StructuralModel(step, "mm-N-MPa", mat, regions, [talos.FixedSupport("socket")], loads, mesh, name=f"{slug}_{key}"),
                              None, st.CARBON["strength"])
            case.exclude = lambda xyz, f=f, x_fin=x_fin: ((xyz[:, 0] < f.x_root + f.tube_od) | (np.abs(xyz[:, 0] - f.x_tie) < 15.0) | (np.abs(xyz[:, 0] - x_fin) < 15.0)
                                                          | (xyz[:, 2] > p["boom_z"] + f.tube_od / 2 + 3.0))
            case.kind, case.frame = "static", f
            out.append(case)
        modal = st.FEACase(f"{slug}_modes", "frame", f"{f.name}: the first modes with the tail's masses",
                           talos.StructuralModel(step, "mm-N-MPa", mat, regions, [talos.FixedSupport("socket")], [], mesh, masses=masses, name=f"{slug}_modes"),
                           None, st.CARBON["strength"])
        modal.kind, modal.frame = "modes", f
        out.append(modal)
    return out


def _tail_motion(case, result, f: Frame, p=None):
    """From a solved static case: the stabiliser's vertical deflection at the tube(s) (mean of the spar's two ends:
    the tail's heave), its roll (the difference / the span), the fin rods' lateral deflection (the fin's yaw)."""
    from vegeta import talos
    p = npl.resolve(p)
    fr = talos.read_frd(result.artifacts["frd"])
    c, u = fr.coords, fr.displacement
    zt = _zt(f, p)
    near = lambda y: (np.abs(c[:, 0] - f.x_tie) < f.tie_od) & (np.abs(c[:, 1] - y) < 6.0) & (np.abs(c[:, 2] - zt) < f.tie_od)
    # the spar where it leaves the tube(s): the twin's roll is the booms' differential heave, the single's the tube's twist
    yl, yr = (-f.boom_y - f.tube_od, f.boom_y + f.tube_od) if f.kind == "twin" else (-f.tube_od - 6.0, f.tube_od + 6.0)
    zl, zr = float(u[near(yl), 2].mean()), float(u[near(yr), 2].mean())
    x_fin = f.x_fin
    top = c[:, 2] > p["boom_z"] + f.h_fin_ac - 3.0
    axis = (np.abs(c[:, 0] - x_fin) < 6.0) & (np.abs(c[:, 2] - p["boom_z"]) < f.tube_od / 2 + 0.5)
    yaw = math.degrees((float(u[top, 1].mean()) - float(u[axis, 1].mean())) / f.h_fin_ac) if top.any() and axis.any() else np.nan
    return {"tail heave at the tube(s) [mm]": 0.5 * (zl + zr), "tail roll [deg]": math.degrees((zr - zl) / (yr - yl)),
            "fin yaw twist at its AC [deg] (isotropic G)": yaw}


def _unsound(case, result) -> str:
    """Why a solved result cannot be trusted (a sliver element or a loose piece in the mesh): the support's reaction
    off the applied load, a peak stress far above the 99.5th percentile, a mode near 0 Hz. '' when it is sound."""
    if result is None or not getattr(result, "ok", False):
        return ""
    m = result.metrics or {}
    if case.kind == "static":
        applied, reaction = m.get("applied_force_magnitude"), m.get("reaction_force_magnitude")
        try:
            from vegeta import talos
            rx = talos.read_dat_reactions(result.artifacts["dat"])
            reaction = float(np.linalg.norm(np.sum([np.asarray(v[:3], float) for v in rx.values()], axis=0))) if rx else reaction
        except Exception:
            pass
        row = st.summary_row(case, result)
        applied, reaction = row.get("applied [N]", applied), row.get("reaction [N]", reaction)
        if applied and reaction and abs(reaction - applied) > 0.001 * applied:
            return f"reaction {reaction:.1f} N against {applied:.1f} N applied"
        if row.get("peak von Mises [MPa]", 0) > 10 * max(row.get("99.5th percentile von Mises [MPa]", 1.0), 1.0):
            return f"peak {row['peak von Mises [MPa]']:.0f} MPa against a 99.5th percentile of {row['99.5th percentile von Mises [MPa]']:.0f}"
        return ""
    freqs = m.get("frequencies_hz") or []
    return f"a mode at {freqs[0]:.2f} Hz" if freqs and freqs[0] < 1.0 else ""


def solve_cases(cases: list, workdir, *, run: bool = True, threads: int = 4, n_modes: int = 6, retries: int = 2, p=None,
                progress: bool = False, label: str = "frames FEA") -> dict:
    """Mesh and solve each case in ``workdir/<name>`` (static: ``ensure``; modes: mesh then ``solve_modes``); ``run``
    False reads back what exists and marks the rest NOT RUN. A frame whose result is unsound (``_unsound``: a mesh
    accident) is rebuilt at a smaller element size in ``workdir/retry<n>`` and solved again, up to ``retries`` times;
    the cases list is updated in place with the retried cases. ``progress``: one tqdm bar over the cases, its postfix
    the case and Talos' stage (mesh, solve) with that stage's fraction."""
    out = {}
    bar = None
    if progress:
        from tqdm.auto import tqdm
        bar = tqdm(total=len(cases), desc=label, bar_format="{desc}: {n_fmt}/{total_fmt} |{bar}| {elapsed} {postfix}")

    def stage_of(name):
        if bar is None:
            return False
        def cb(stage, fraction=None, message=""):
            bar.set_postfix_str(f"{name}: {stage}" + (f" {fraction:.0%}" if fraction is not None else ""))
        return cb

    for c in cases:
        wd = Path(workdir) / c.name
        if c.kind == "static":
            out[c.name] = c.model.ensure(wd, run=run, threads=threads, progress=stage_of(c.name))
        else:
            if not run and not (wd / "modes.dat").exists():
                out[c.name] = None
                if bar: bar.update(1)
                continue
            if not (wd / "mesh.msh").exists():
                c.model.mesh(wd, progress=stage_of(c.name))
            out[c.name] = c.model.solve_modes(wd, n_modes=n_modes, threads=threads, progress=stage_of(c.name))
        if bar:
            bar.update(1)
    if bar:
        bar.set_postfix_str("done"); bar.close()
    if not run or retries <= 0:
        return out
    bad = {c.frame.name for c in cases if _unsound(c, out[c.name])}
    if not bad:
        return out
    for name in sorted(bad):
        print(f"{name}: unsound mesh — " + "; ".join(f"{c.name}: {_unsound(c, out[c.name])}" for c in cases if c.frame.name == name and _unsound(c, out[c.name])))
    frames_ = [c.frame for c in cases if c.frame.name in bad and c.kind == "modes"]
    size = cases[0].model.mesh_settings.element_size if hasattr(cases[0].model, "mesh_settings") else 2.5
    n = sum(1 for d in Path(workdir).glob("retry*")) + 1
    redo = fea_models(frames_, Path(workdir) / f"retry{n}", p, element_size=size * 0.88)
    out2 = solve_cases(redo, Path(workdir) / f"retry{n}", run=run, threads=threads, n_modes=n_modes, retries=retries - 1, p=p, progress=progress,
                       label=f"{label} (retry {n})")
    for c in redo:
        i = next(k for k, c0 in enumerate(cases) if c0.name == c.name)
        cases[i] = c
        out[c.name] = out2[c.name]
    return out


def fea_table(cases: list, results: dict, p=None) -> pd.DataFrame:
    """One row per static case: NISUS+'s summary (stress, margin) plus the tail's motion."""
    rows = []
    for c in cases:
        if c.kind != "static":
            continue
        r = results.get(c.name)
        row = st.summary_row(c, r)
        row["case"] = c.name
        if r is not None and getattr(r, "ok", False):
            row.update(_tail_motion(c, r, c.frame, p))
        rows.append(row)
    return pd.DataFrame(rows).set_index("case")


def modes_table(cases: list, results: dict, n: int = 5) -> pd.DataFrame:
    """The first ``n`` frequencies of each frame's modal model (the FEA's isotropic G makes a torsional mode of the
    single tube ~2-3 x too stiff: marked)."""
    rows = {}
    for c in cases:
        if c.kind != "modes":
            continue
        r = results.get(c.name)
        if r is None or not getattr(r, "ok", False):
            rows[c.frame.name] = {"status": "NOT RUN"}
            continue
        freqs = (r.metrics or {}).get("frequencies_hz") or []
        rows[c.frame.name] = {"status": "solved", **{f"mode {i + 1} [Hz]": freqs[i] if i < len(freqs) else np.nan for i in range(n)},
                              "torsion note": "isotropic G 46 GPa: a torsional mode is optimistic x %.1f" % math.sqrt(46000 / G_TUBE[c.frame.tube_kind])}
    return pd.DataFrame(rows).T


# -------------------------------------------------------------------------------------------------------------- verdict
def verdict(hand: pd.DataFrame, wing: pd.DataFrame, aero: pd.DataFrame, perf: pd.DataFrame) -> dict:
    """The study's conclusions in numbers: whether moving NISUS+'s booms apart for the large propeller makes sense,
    and the single tail's standing against the twin frame."""
    twins = [i for i in hand.index if i.startswith("twin")]
    base = twins[0]
    out = {"widening": {}, "single vs twin": {}}
    for i in twins[1:]:
        out["widening"][i] = {"feasible": bool(hand.loc[i, "feasible"]), "booms apart [mm]": float(hand.loc[i, "booms apart [mm]"]),
                              "rear tube": wing.loc[i, "rear tube"], "crow drag kept [%]": float(wing.loc[i, "crow drag kept [%]"]),
                              "frame mass [g]": float(hand.loc[i, "frame mass [g]"]), "Cd0": float(aero.loc[i, "aircraft Cd0"]),
                              "mission energy [Wh]": float(aero.loc[i, "mission energy [Wh]"]), "note": hand.loc[i, "note"]}
    s = [i for i in hand.index if i.startswith("single")][0]
    t20 = [i for i in twins if "20" in i]
    ref = t20[0] if t20 else base
    out["single vs twin"] = {"against": ref, "frame mass [g]": (float(hand.loc[s, "frame mass [g]"]), float(hand.loc[ref, "frame mass [g]"])),
                             "frame drag area [m²]": (float(aero.loc[s, "frame drag area [m²]"]), float(aero.loc[ref, "frame drag area [m²]"])),
                             "L/D max": (float(aero.loc[s, "L/D max"]), float(aero.loc[ref, "L/D max"])),
                             "mission energy [Wh]": (float(aero.loc[s, "mission energy [Wh]"]), float(aero.loc[ref, "mission energy [Wh]"])),
                             "tail roll, asymmetric [deg]": (float(hand.loc[s, "tail roll, asymmetric case [deg]"]), float(hand.loc[ref, "tail roll, asymmetric case [deg]"])),
                             "fin twist [deg]": (float(hand.loc[s, "fin twist under its side load [deg]"]), float(hand.loc[ref, "fin twist under its side load [deg]"])),
                             "first bending mode [Hz]": (float(hand.loc[s, "first bending mode, hand [Hz]"]), float(hand.loc[ref, "first bending mode, hand [Hz]"])),
                             "spar tail-case stress [MPa]": (0.0, float(wing.loc[ref, "main spar: bending from the boom, tail case [MPa]"]))}
    return out


# --------------------------------------------------------------------------------------------------------------- sketch
def sketch(F: list, p=None, figsize=(16, 5)):
    """Top views of the frames: the wing's planform, the pod, the propeller disc, the booms or the tube, the tail."""
    import matplotlib.pyplot as plt
    p = npl.resolve(p)
    L = npl.NisusPlus.layout(p)
    pl = npl.planform_split(p) if hasattr(npl, "planform_split") else None
    fig, axes = plt.subplots(1, len(F), figsize=figsize, sharey=True)
    for ax, f in zip(np.atleast_1d(axes), F):
        b2 = p["span"] / 2
        xs = [0, p["root_chord"], p["tip_chord"] + (p["root_chord"] - p["tip_chord"]) * 0.25, 0.25 * (p["root_chord"] - p["tip_chord"])]
        ax.fill([0, p["root_chord"], p["root_chord"] - 0.75 * (p["root_chord"] - p["tip_chord"]), 0.25 * (p["root_chord"] - p["tip_chord"])],
                [-b2, -b2, b2, b2], color="#cfe0f3", ec="k", lw=0.6)
        ax.fill([L["x_nose"], L["x_pod_end"], L["x_pod_end"], L["x_nose"]], [-p["pod_width"] / 2] * 2 + [p["pod_width"] / 2] * 2, color="#e8e8e8", ec="k", lw=0.6)
        R = f.prop_in * 25.4 / 2
        if f.kind == "twin":
            for y in (-f.boom_y, f.boom_y):
                ax.fill([p["boom_x0"], f.x_end, f.x_end, p["boom_x0"]], [y - f.tube_od / 2, y - f.tube_od / 2, y + f.tube_od / 2, y + f.tube_od / 2], color="#444")
            ax.fill([f.x_end - f.tail_chord, f.x_end, f.x_end, f.x_end - f.tail_chord], [-f.tail_span / 2, -f.tail_span / 2, f.tail_span / 2, f.tail_span / 2], color="#cfe0f3", ec="k", lw=0.6)
            ax.add_patch(plt.Circle((L["prop_x"], 0), R, fill=False, ls="--", color="r"))
            ax.plot([0.7 * p["root_chord"]] * 2, [f.flap_y0, p["flap_y1"]], color="C1", lw=3, label="flap")
            ax.plot([0.7 * p["root_chord"]] * 2, [-p["flap_y1"], -f.flap_y0], color="C1", lw=3)
        else:
            ax.fill([f.x_root - 200, f.x_end, f.x_end, f.x_root - 200], [-f.tube_od / 2, -f.tube_od / 2, f.tube_od / 2, f.tube_od / 2], color="#444")
            ax.fill([f.x_end - f.tail_chord, f.x_end, f.x_end, f.x_end - f.tail_chord], [-f.tail_span / 2, -f.tail_span / 2, f.tail_span / 2, f.tail_span / 2], color="#cfe0f3", ec="k", lw=0.6)
            ax.add_patch(plt.Circle((L["x_nose"] - 30, 0), R, fill=False, ls="--", color="r"))
            ax.plot([0.7 * p["root_chord"]] * 2, [f.flap_y0, p["flap_y1"]], color="C1", lw=3, label="flap")
            ax.plot([0.7 * p["root_chord"]] * 2, [-p["flap_y1"], -f.flap_y0], color="C1", lw=3)
        ax.set_aspect("equal"); ax.set_xlim(-650, 1100); ax.set_ylim(-1250, 1250)
        ax.set_title(f"{f.name}" + ("" if f.feasible else "\n(rejected)"), color="k" if f.feasible else "C3", fontsize=10)
        ax.grid(alpha=0.2); ax.set_xlabel("x [mm]")
    axes[0].set_ylabel("y [mm]")
    fig.suptitle("the tail frames on NISUS+'s wing (top view; the propeller disc red dashed, the flaps orange)")
    fig.tight_layout()
    return fig
