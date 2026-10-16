"""Kestrel-4Q — the hold point: can four fixed, parallel rotors control the aircraft? (notebook 38)

The allocation matrix ``B`` maps the four thrusts to the collective thrust and the three body moments about the CG:

    [T, L, M, N]ᵀ = B · [T1, T2, T3, T4]ᵀ

with every thrust along −x (toward the nose). A rotor at (dx, y, dz) from the CG pushing T toward the nose gives
r × F = (0, −dz·T, y·T): the **vertical** offset (the wing gap) moves M, the **spanwise** offset moves N, and a
fore–aft offset moves nothing — in a tailsitter the thrust axis is the body axis, so fore–aft rotor offsets carry no
lever at all. L (about the thrust axis: the heading in the hover, the roll in cruise) only comes from the rotors'
reaction torques, −s·κ·T for a rotor turning with sense s about +x (κ = Q/T, Boreas). Standing nose-up, the y–z
plane is horizontal: the illustrated 2 × 2 layout (spanwise pairs, a vertical gap) is exactly a quadcopter's X seen
from above. The risk the drawing hides is not the layout but the senses of rotation, and the weak heading lever.

- ``SPINS``: the senses of rotation looked at (mirror-symmetric ones and others).
- ``allocation``, ``rank_table``: B, its rank, singular values and condition for each.
- ``attainable``: the largest moment about each axis with the collective held (hover: the weight) and the other moments
  zero, every rotor inside [T_min, T_max] — a linear programme per direction.
- ``motor_out``: one rotor lost — what can still be held.
- ``hover_wind``: the lean into a steady wind standing in the hover (the wings broadside), and the margin left.
- ``roll_authority``: the cruise roll lever (reaction torque only) against ailerons on the blown outer panels.

Units SI. Moments: L about +x (aft), M about +y (right; positive = nose up), N about +z (up).
"""
from __future__ import annotations

import math

import numpy as np

import kestrel
import kestrel_systems as ks

G = ks.G
SPINS = {
    "diagonal pairs (M1, M4 | M2, M3)": (1, -1, -1, 1),        # mirror-symmetric: upper pair and lower pair opposite
    "side columns alike (M1, M3 | M2, M4)": (1, -1, 1, -1),    # mirror-symmetric: each side's two rotors turn alike
    "rows alike (M1, M2 | M3, M4)": (1, 1, -1, -1),            # upper pair one way, lower the other (not mirror-symmetric)
    "all the same way": (1, 1, 1, 1),
}
BASELINE_SPIN = "diagonal pairs (M1, M4 | M2, M3)"
AXES = ("T", "L", "M", "N")


def rotor_offsets(p=None, cg=None) -> np.ndarray:
    """(dx, y, dz) of M1..M4 from the CG [m] (rows)."""
    L = kestrel.layout(p)
    cg = cg or {"x_cg": L["x_le"] + 0.1 * L["chord"], "y_cg": 0.0, "z_cg": 0.0}
    return np.array([[s["x"] - cg["x_cg"], s["y"] - cg["y_cg"], s["z"] - cg["z_cg"]] for s in L["motors"].values()])


def allocation(p=None, *, spins=BASELINE_SPIN, kappa: float = 0.012, cg=None) -> np.ndarray:
    """B (4 × 4): rows T, L, M, N; columns M1..M4."""
    s = np.asarray(SPINS[spins] if isinstance(spins, str) else spins, float)
    r = rotor_offsets(p, cg)
    return np.vstack([np.ones(4), -s * kappa, -r[:, 2], r[:, 1]])


def rank_table(p=None, *, kappa: float = 0.012, cg=None):
    """Rank, singular values and condition of B for each sense-of-rotation set; the axis lost when it is singular.
    The rows are scaled to the hover's natural units first (T in N; L, M, N in N·m divided by the matching lever: κ,
    the half gap, the motor station), so a singular value of 2 means a full lever and the condition compares levers."""
    import pandas as pd
    Lh = kestrel.layout(p)
    rows = {}
    for name in SPINS:
        B = allocation(p, spins=name, kappa=kappa, cg=cg)
        sv = np.linalg.svd(B, compute_uv=False)
        rk = int(np.linalg.matrix_rank(B, tol=1e-9 * sv[0]))
        lost = ""
        if rk < 4:
            _, _, vt = np.linalg.svd(B.T)
            nul = vt[-1]                                  # left null vector: a combination of T, L, M, N never produced
            lost = " + ".join(f"{c:+.2f}·{a}" for c, a in zip(nul / np.abs(nul).max(), AXES) if abs(c) > 1e-6)
        mirror = SPINS[name][0] == -SPINS[name][1] and SPINS[name][2] == -SPINS[name][3]
        rows[name] = {"senses (M1..M4)": str(SPINS[name]), "mirror-symmetric": mirror, "rank": rk,
                      "σ (N, N·m)": ", ".join(f"{v:.3g}" for v in sv), "condition": sv[0] / sv[-1] if sv[-1] > 1e-12 else math.inf,
                      "never produced": lost or "—",
                      "verdict": "full control (thrust + 3 moments)" if rk == 4 else "NOT controllable in the hover"}
    df = pd.DataFrame(rows).T
    df.attrs["levers"] = {"kappa_m": kappa, "half_gap_m": Lh["h"], "motor_y_m": Lh["motor_y"]}
    return df


def attainable(B: np.ndarray, collective: float, t_min, t_max) -> dict:
    """The largest positive and negative moment about each of L, M, N that B can produce with Σ T held at
    ``collective`` and the other two moments zero, every rotor in [t_min, t_max] (scalars or per-rotor arrays):
    {axis: (most negative, most positive)}; (nan, nan) if even the trim is infeasible."""
    from scipy.optimize import linprog
    t_min = np.broadcast_to(np.asarray(t_min, float), (4,))
    t_max = np.broadcast_to(np.asarray(t_max, float), (4,))
    out = {}
    for k, ax in ((1, "L"), (2, "M"), (3, "N")):
        others = [i for i in (1, 2, 3) if i != k]
        A_eq = np.vstack([B[0], B[others]])
        b_eq = np.array([collective, 0.0, 0.0])
        ext = []
        for sgn in (-1.0, 1.0):
            res = linprog(-sgn * B[k], A_eq=A_eq, b_eq=b_eq, bounds=list(zip(t_min, t_max)), method="highs")
            ext.append(float(B[k] @ res.x) if res.status == 0 else math.nan)
        out[ax] = tuple(ext)
    return out


def hover_authority(p=None, *, mass_kg: float, cg=None, spins=BASELINE_SPIN, drive=None, inertia=None) -> "pd.DataFrame":
    """The attainable moments in the hover (collective = the weight, rotors in [T_min, T_max(0)]) and the angular
    accelerations they give with the inertia; the yaw (heading) lever is the reaction torque's."""
    import pandas as pd
    drive = drive or ks.drive_table()
    kap = float(drive.kap(0.0))
    B = allocation(p, spins=spins, kappa=kap, cg=cg)
    att = attainable(B, mass_kg * G, drive.T_min, float(drive.t_max(0.0)))
    I = inertia if inertia is not None else np.diag([0.03, 0.03, 0.03])
    names = {"L": "about the thrust axis (hover heading / cruise roll)", "M": "about y (pitch-over / cruise pitch)",
             "N": "about z (hover tilt sideways / cruise yaw)"}
    rows = {}
    for k, ax in enumerate(("L", "M", "N")):
        lo, hi = att[ax]
        rows[names[ax]] = {"most negative [N·m]": lo, "most positive [N·m]": hi, "inertia [kg·m²]": I[k, k],
                           "angular acceleration [rad/s²]": min(abs(lo), abs(hi)) / I[k, k]}
    return pd.DataFrame(rows).T


def motor_out(p=None, *, mass_kg: float, cg=None, spins=BASELINE_SPIN, drive=None) -> "pd.DataFrame":
    """One rotor stopped: in the hover, can the other three hold the weight with M = N = 0 — and what reaction torque
    (about the thrust axis) is then left unbalanced (the aircraft spins up about its axis: a reduced-attitude
    controller that lets it spin, or a parachute)? In cruise the thrust needed is small: the diagonal partner must
    go to zero too, two rotors fly on, and the roll torque left over is what the ailerons hold."""
    import pandas as pd
    drive = drive or ks.drive_table()
    kap = float(drive.kap(0.0))
    B = allocation(p, spins=spins, kappa=kap, cg=cg)
    tmax = float(drive.t_max(0.0))
    rows = {}
    for j, mname in enumerate(kestrel.MOTORS):
        keep = [i for i in range(4) if i != j]
        A = B[[0, 2, 3]][:, keep]
        try:
            t = np.linalg.solve(A, [mass_kg * G, 0.0, 0.0])
        except np.linalg.LinAlgError:
            t = np.full(3, math.nan)
        Lres = float(B[1, keep] @ t)
        idle = bool(np.all(t >= -0.05 * mass_kg * G / 4) and np.all(t <= tmax))       # the diagonal partner at ~0 N
        verdict = ("cannot hold the weight level" if not idle else
                   "flies on one diagonal pair (the partner idles): holds thrust and tilt, spins about its axis")
        rows[f"{mname} out"] = {"thrusts of the others [N]": ", ".join(f"{x:.2f}" for x in t), "within limits": idle,
                                "unbalanced torque about the axis [N·m]": Lres, "hover": verdict}
    return pd.DataFrame(rows).T


def hover_wind(p=None, *, mass_kg: float, cg=None, winds=(2.0, 4.0, 6.0, 8.0, 10.0), cd_broadside: float = 1.2, drive=None,
               rho: float = ks.RHO0) -> "pd.DataFrame":
    """Standing in a steady wind in the hover with the wings broadside to it (the worst heading; edge-on the wings
    see nothing): the drag of both wings, pod and fin as flat plates, the lean into the wind that balances it, the
    thrust each rotor then needs and the pitch moment left for control. The aircraft can also turn edge-on (yaw)
    — that is the operating rule this table argues for."""
    import pandas as pd
    L = kestrel.layout(p)
    drive = drive or ks.drive_table()
    W = mass_kg * G
    area = L["S_ref"]                               # both wings' planform (the pod and the fin plate are edge-on to this wind)
    rows = {}
    for v in winds:
        F = 0.5 * rho * v ** 2 * area * cd_broadside
        lean = math.degrees(math.atan2(F, W))
        T_tot = math.hypot(F, W)
        B = allocation(p, kappa=float(drive.kap(0.0)), cg=cg)
        att = attainable(B, T_tot, drive.T_min, float(drive.t_max(0.0)))
        rows[f"{v:.0f} m/s"] = {"wing drag [N]": F, "lean into the wind [deg]": lean, "thrust per rotor [N]": T_tot / 4,
                                "T_max per rotor [N]": float(drive.t_max(0.0)), "pitch moment left [N·m]": min(abs(att["M"][0]), abs(att["M"][1]))}
    return pd.DataFrame(rows).T


def roll_authority(p=None, *, V: float, thrust_each: float, cl_alpha_3d: float, aileron_chord: float = 0.25,
                   aileron_deg: float = 15.0, tau: float = 0.45, drive=None, rho: float = ks.RHO0) -> dict:
    """Roll (about x) in cruise at airspeed ``V``: the reaction-torque lever (one diagonal pair — the same sense,
    pitch- and yaw-neutral — up to T_max, the other down to T_min) against ailerons on both wings' outer panels behind the propellers (from motor_y − R to the tip,
    ``aileron_chord`` of the chord, effectiveness ``tau``, both wings, ±``aileron_deg``); the slipstream raises the
    ailerons' dynamic pressure by the momentum-theory factor."""
    L = kestrel.layout(p)
    drive = drive or ks.drive_table()
    kap = float(drive.kap(V))
    torque = 2 * kap * (max(float(drive.t_max(V)) - thrust_each, 0.0) + max(thrust_each - drive.T_min, 0.0))
    y0, y1 = L["motor_y"] - L["R_prop"], L["span"] / 2
    s_panel = (y1 - y0) * L["chord"]
    v_slip2 = V ** 2 + 2 * thrust_each / (rho * L["A_disc"])            # far-wake velocity squared
    q_blown = 0.5 * rho * v_slip2
    dCL = cl_alpha_3d * tau * math.radians(aileron_deg)
    aileron = 2 * 2 * q_blown * s_panel * dCL * 0.5 * (y0 + y1)         # two wings, two sides
    return {"V": V, "torque lever [N·m]": torque, "ailerons [N·m]": aileron, "ratio": aileron / torque if torque > 0 else math.inf,
            "aileron span per side [m]": y1 - y0}
