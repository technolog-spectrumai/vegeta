"""Kestrel-4Q — the truss, the wings' bending, the propellers' clearance under load, and the ground stance (notebook 38).

- ``frame``: the front view (y–z plane) as a plane frame. The spars are beams through the pod (the pod a rigid post at
  y = 0, clamped there: the half-model's symmetry and the flight's inertia relief at the CG), the posts and the
  diagonals are pin-ended bars. The load: a pull-up at load factor n, the lift spread elliptically on both wings
  (the lattice's upper/lower share), less the inertia of the wings and of the motors at their stations.
  ``compare_bracing`` solves it braced (posts + X), posts only, and with no truss (two cantilevers): the spar stress,
  the tip and motor-station deflections, the change of the upper-to-lower propeller gap, the bars' forces against
  their Euler buckling loads. A hand model in the spirit of the notebooks' first estimates: Talos FEA of the joints
  is the next step (joint peel, the tubes' local buckling, the balsa skin's share are not in it).
- ``ground_stance``: standing nose-up on its six feet (the fin tips and the four foot rods), the tip-over angles over
  each edge of the footprint, the steady wind that tips it over (broadside to the wings, and edge-on), and the
  touchdown: the feet's load at a sink rate, the foot rods' buckling margin.

Units SI. Materials: the tubes' and rods' axial modulus and allowables are assumptions for pultruded / roll-wrapped
carbon (``CARBON``).
"""
from __future__ import annotations

import math

import numpy as np

import kestrel
import kestrel_systems as ks

G = ks.G
CARBON = {"E": 80e9, "sigma_allow": 400e6, "rho": 1550.0}       # tube modulus [Pa], allowable (compression, with knock-down) [Pa]
TUBES = {"spar": (0.010, 0.008), "post": (0.008, 0.006), "diagonal": (0.006, 0.004), "foot rod": (0.006, 0.0)}   # (OD, ID) [m]


def _section(od, idd):
    A = math.pi / 4 * (od ** 2 - idd ** 2)
    I = math.pi / 64 * (od ** 4 - idd ** 4)
    return A, I


# ================================================================================================= the plane frame
def _model(p, bracing: str, n_seg: int = 6):
    """Nodes (y, z), beam elements (spars) and bar elements (posts, diagonals) of the right half (y ≥ 0)."""
    L = kestrel.layout(p)
    h, b2, ym, yin = L["h"], L["span"] / 2, L["motor_y"], L["pod_width"] / 2
    stations = sorted(set(np.round(np.r_[np.linspace(0.0, b2, n_seg + 1), ym, yin], 6)))
    nodes, idx = [], {}
    for wing, z in (("upper", h), ("lower", -h)):
        for y in stations:
            idx[(wing, y)] = len(nodes)
            nodes.append((y, z))
    beams = []
    for wing in ("upper", "lower"):
        for a, b in zip(stations[:-1], stations[1:]):
            beams.append((idx[(wing, a)], idx[(wing, b)], "spar"))
    bars = []
    if bracing in ("posts", "braced"):
        bars.append((idx[("upper", ym)], idx[("lower", ym)], "post"))
    if bracing == "braced":
        bars.append((idx[("upper", yin)], idx[("lower", ym)], "diagonal"))
        bars.append((idx[("lower", yin)], idx[("upper", ym)], "diagonal"))
    return np.array(nodes), beams, bars, idx, stations, L


def frame(p=None, *, n: float = 4.0, mass_kg: float, bracing: str = "braced", upper_share: float = 0.5,
          distribution: str = "elliptic", inertia: bool = True) -> dict:
    """Solve the half frame at load factor ``n`` (see the module's text). Returns the spar's peak bending stress, the
    deflections, the bars' forces and buckling margins (negative force: compression). ``distribution``: the lift's
    spanwise shape (``elliptic`` or ``uniform``); ``inertia=False`` leaves out the wings' and the motors' inertia (the
    check against beam theory)."""
    p = kestrel.resolve(p)
    nodes, beams, bars, idx, stations, L = _model(p, bracing)
    E = CARBON["E"]
    nn = len(nodes)
    K = np.zeros((3 * nn, 3 * nn))
    F = np.zeros(3 * nn)

    def add(i, j, A, I, beam=True):
        (y1, z1), (y2, z2) = nodes[i], nodes[j]
        dy, dz = y2 - y1, z2 - z1
        Le = math.hypot(dy, dz)
        c, s = dy / Le, dz / Le
        k = np.zeros((6, 6))
        k[[0, 3], [0, 3]] = E * A / Le
        k[0, 3] = k[3, 0] = -E * A / Le
        if beam:
            b = E * I / Le ** 3
            sub = b * np.array([[12, 6 * Le, -12, 6 * Le], [6 * Le, 4 * Le ** 2, -6 * Le, 2 * Le ** 2],
                                [-12, -6 * Le, 12, -6 * Le], [6 * Le, 2 * Le ** 2, -6 * Le, 4 * Le ** 2]])
            ii = [1, 2, 4, 5]
            for a_ in range(4):
                for b_ in range(4):
                    k[ii[a_], ii[b_]] += sub[a_, b_]
        T = np.zeros((6, 6))
        R = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
        T[:3, :3] = R
        T[3:, 3:] = R
        kg = T.T @ k @ T
        dofs = [3 * i, 3 * i + 1, 3 * i + 2, 3 * j, 3 * j + 1, 3 * j + 2]
        K[np.ix_(dofs, dofs)] += kg
        return Le, c, s

    A_s, I_s = _section(*TUBES["spar"])
    for i, j, _ in beams:
        add(i, j, A_s, I_s, True)
    bar_geo = []
    for i, j, kind in bars:
        A, I = _section(*TUBES[kind])
        Le, c, s = add(i, j, A, I, False)
        bar_geo.append((i, j, kind, Le, c, s, A, I))
    # loads: elliptic lift on the half span of each wing, minus the wing's and the motors' inertia (n g)
    half_lift = n * mass_kg * G / 2
    b2 = L["span"] / 2
    wing_mass = ks.WING_AREAL_KG_M2 * L["S_wing"] + ks.SPAR_KG_M * L["span"]
    motor_mass = ks.COMPONENTS["motor"]["mass_kg"] + ks.COMPONENTS["propeller"]["mass_kg"] + ks.COMPONENTS["esc"]["mass_kg"] + ks.NACELLE_KG
    for wing, share in (("upper", upper_share), ("lower", 1 - upper_share)):
        for a, b in zip(stations[:-1], stations[1:]):
            ya, yb = a / b2, b / b2
            if distribution == "uniform":
                frac = yb - ya
            else:                                            # ∫ √(1 − η²) dη over the segment, over π/4
                frac = (2 / math.pi) * ((yb * math.sqrt(1 - yb ** 2) + math.asin(yb)) - (ya * math.sqrt(1 - ya ** 2) + math.asin(ya)))
            lift = half_lift * share * frac
            inert = n * G * wing_mass / 2 * (b - a) / b2 if inertia else 0.0
            for node in (idx[(wing, a)], idx[(wing, b)]):
                F[3 * node + 1] += 0.5 * (lift - inert)
        if inertia:
            F[3 * idx[(wing, L["motor_y"])] + 1] -= n * G * motor_mass
    # supports: the pod clamps both spars at y = 0 (the symmetry plane)
    fixed = []
    for wing in ("upper", "lower"):
        i0 = idx[(wing, stations[0])]
        fixed += [3 * i0, 3 * i0 + 1, 3 * i0 + 2]
    free = np.setdiff1d(np.arange(3 * nn), fixed)
    u = np.zeros(3 * nn)
    u[free] = np.linalg.solve(K[np.ix_(free, free)], F[free])
    # the spar's bending moment from the element end forces
    peak_M = 0.0
    for i, j, _ in beams:
        (y1, z1), (y2, z2) = nodes[i], nodes[j]
        Le = math.hypot(y2 - y1, z2 - z1)
        b = E * I_s / Le ** 3
        v1, t1, v2, t2 = u[3 * i + 1], u[3 * i + 2], u[3 * j + 1], u[3 * j + 2]
        M1 = b * Le * (6 * v1 + 4 * Le * t1 - 6 * v2 + 2 * Le * t2)
        M2 = b * Le * (-6 * v1 - 2 * Le * t1 + 6 * v2 - 4 * Le * t2)
        peak_M = max(peak_M, abs(M1), abs(M2))
    sigma = peak_M * TUBES["spar"][0] / 2 / I_s
    bar_rows = []
    for i, j, kind, Le, c, s, A, I in bar_geo:
        du = (u[3 * j] - u[3 * i]) * c + (u[3 * j + 1] - u[3 * i + 1]) * s
        N = E * A / Le * du
        Pcr = math.pi ** 2 * E * I / Le ** 2
        bar_rows.append({"member": kind, "length [m]": Le, "force [N]": N, "Euler load [N]": Pcr,
                         "buckling margin": (Pcr / -N) if N < 0 else math.inf, "stress [MPa]": N / A / 1e6})
    ym = L["motor_y"]
    w_up, w_lo = u[3 * idx[("upper", ym)] + 1], u[3 * idx[("lower", ym)] + 1]
    tip_up, tip_lo = u[3 * idx[("upper", stations[-1])] + 1], u[3 * idx[("lower", stations[-1])] + 1]
    return {"bracing": bracing, "n": n, "spar peak moment [N·m]": peak_M, "spar stress [MPa]": sigma / 1e6,
            "spar reserve": CARBON["sigma_allow"] / sigma if sigma > 0 else math.inf,
            "tip deflection, upper [mm]": tip_up * 1e3, "tip deflection, lower [mm]": tip_lo * 1e3,
            "motor station, upper [mm]": w_up * 1e3, "motor station, lower [mm]": w_lo * 1e3,
            "propeller gap change [mm]": (w_up - w_lo) * 1e3, "propeller gap under load [mm]": L["prop_gap"] * 1e3 + (w_up - w_lo) * 1e3,
            "bars": bar_rows}


def compare_bracing(p=None, *, mass_kg: float, n: float = 4.0, upper_share: float = 0.5) -> "pd.DataFrame":
    import pandas as pd
    rows = {}
    for br, label in (("braced", "posts + X diagonals"), ("posts", "posts only"), ("none", "no truss (two cantilevers)")):
        r = frame(p, n=n, mass_kg=mass_kg, bracing=br, upper_share=upper_share)
        comp = [b for b in r["bars"] if b["force [N]"] < 0]
        rows[label] = {k: v for k, v in r.items() if k not in ("bars", "bracing", "n")}
        rows[label]["least buckling margin"] = min((b["buckling margin"] for b in comp), default=math.inf)
        rows[label]["truss mass [g]"] = 1000 * (
            (2 * ks.POST_KG_M * kestrel.layout(p)["post_length"] if br != "none" else 0.0)
            + (4 * ks.DIAGONAL_KG_M * kestrel.layout(p)["diagonal_length"] if br == "braced" else 0.0)
            + (12 * ks.JOINT_KG if br == "braced" else 4 * ks.JOINT_KG if br == "posts" else 0.0))
    return pd.DataFrame(rows).T


# ================================================================================================= standing on its feet
def _hull(points):
    pts = sorted(set(points))
    if len(pts) < 3:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for pnt in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], pnt) <= 0:
            lower.pop()
        lower.append(pnt)
    for pnt in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], pnt) <= 0:
            upper.pop()
        upper.append(pnt)
    return lower[:-1] + upper[:-1]


def ground_stance(p=None, *, mass_kg: float, x_cg: float, z_cg: float = 0.0, cd_broadside: float = 1.2, cd_edge: float = 1.1,
                  sink_ms: float = 1.0, stroke_m: float = 0.008, rho: float = ks.RHO0) -> dict:
    """The footprint (the feet's (y, z) on the ground plane x = x_feet), the CG's height above it, the tip-over angle
    over each edge, the steady wind that tips it over broadside (along z, onto the wings' planform) and edge-on (along
    y, onto the fin and the pod's side), and the touchdown on the feet at ``sink_ms`` stopped over ``stroke_m``."""
    L = kestrel.layout(p)
    H = L["x_feet"] - x_cg
    hull = _hull([(round(y, 6), round(z, 6)) for y, z in L["feet"]])
    c = np.array([0.0, z_cg])
    edges = []
    for a, b in zip(hull, hull[1:] + hull[:1]):
        a, b = np.array(a), np.array(b)
        t = b - a
        nrm = np.array([t[1], -t[0]]) / np.linalg.norm(t)
        d = abs(float(np.dot(c - a, nrm)))
        edges.append({"edge": f"({a[0]:+.2f}, {a[1]:+.2f}) – ({b[0]:+.2f}, {b[1]:+.2f})", "distance [m]": d,
                      "tip-over angle [deg]": math.degrees(math.atan2(d, H)), "normal": nrm})
    W = mass_kg * G
    # the wind's point of action: the wings' centroid along x (broadside); the fin and pod's (edge-on)
    h_wing = L["x_feet"] - (L["x_le"] + 0.5 * L["chord"])
    A_broad = L["S_ref"]
    fin_area = (L["fin_height"] + L["ventral_height"]) * 0.5 * (L["x_feet"] - L["fin_root_x"] + L["fin_tip_chord"])
    A_edge = fin_area + L["pod_length"] * L["gap"] * 0.8
    h_edge = L["x_feet"] - 0.5 * (L["fin_root_x"] + L["x_feet"])
    d_z = min(e["distance [m]"] for e in edges if abs(e["normal"][1]) > 0.7)      # edges facing ±z
    d_y = min(e["distance [m]"] for e in edges if abs(e["normal"][0]) > 0.7)      # edges facing ±y
    V_broad = math.sqrt(2 * W * d_z / (rho * cd_broadside * A_broad * h_wing))
    V_edge = math.sqrt(2 * W * d_y / (rho * cd_edge * A_edge * h_edge))
    F_td = mass_kg * (G + sink_ms ** 2 / (2 * stroke_m))
    A_r, I_r = _section(*TUBES["foot rod"])
    P_rod = math.pi ** 2 * CARBON["E"] * I_r / (L["foot_rod_length"] ** 2)          # pinned
    return {"CG height above the ground [m]": H, "edges": [{k: v for k, v in e.items() if k != "normal"} for e in edges],
            "least tip-over angle [deg]": min(e["tip-over angle [deg]"] for e in edges),
            "wind that tips it, broadside [m/s]": V_broad, "wind that tips it, edge-on [m/s]": V_edge,
            "touchdown load [N]": F_td, "per foot, two feet first [N]": F_td / 2, "foot rod Euler load [N]": P_rod,
            "foot rod margin, two feet first": P_rod / (F_td / 2)}
