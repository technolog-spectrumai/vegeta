"""Locomotion metrics on an episode log — protocol ``docs/myropod_stability.md`` §6 and §9.2.

Pure numpy/scipy: no MuJoCo and no other Vegeta import, so the same functions work on a log written by
``ChironLab`` or on one recorded on a real robot, as long as it follows Chiron's episode log format (the
dict ``ChironLab.run`` writes into ``Episode.log``; keys used here: ``t``, ``bodies``, ``body_pos``, ``body_quat``, ``body_angvel``, ``com``,
``com_vel``, ``ang_mom``, ``total_mass``, ``gravity``, ``feet``, ``foot_body``, ``foot_group``, ``foot_pos``,
``foot_force``, ``foot_normal``, ``foot_contact_pos``, ``foot_mu``, ``foot_jac``, ``foot_joints``, ``joints``,
``joint_kind``, ``joint_active``, ``q``, ``qd``, ``tau``, ``tau_stall``, ``tau_rated``, ``qd_noload``,
``i_stall``, ``voltage``, ``q_range``, ``belly_contact``, ``disturbances``, ``nominal_hip_height``).

Conventions (fixed here, used by every function)

* World frame z up; the route runs along +x from x = 0. Units SI in, explicit unit suffixes out
  (``_m``, ``_s``, ``_deg``, ``_rad_s``, ``_J``, ``_W``, ``_Nm``, ``_hz``, ``_cycles``). Angles are *reported*
  in degrees because the protocol states its thresholds in degrees; the log itself is in radians.
* Forces: ``foot_force`` is the total contact force ON the foot BY the environment (the ground reaction
  acting on the robot), world frame, N. ``foot_normal`` is the unit normal pointing from the ground INTO the
  foot. The normal force is ``N = f · n``; when the logged normal is missing (nan) the world vertical
  component ``f_z`` is used (a real-robot log without normals).
* Loaded: a foot is loaded when ``N > load_fraction · m |g|`` (protocol: 2 % of the robot's weight). The
  contact record of the duty factor, the phases, the slip episodes, the support polygon and the
  feasibility LP all use this one definition.
* Orientation: quaternions w, x, y, z; body frame x forward, y left, z up. Yaw ψ, pitch θ and roll φ are the
  intrinsic z-y-x Euler angles, ``R = Rz(ψ) Ry(θ) Rx(φ)``: ``ψ = atan2(R10, R00)``, ``θ = asin(−R20)``,
  ``φ = atan2(R21, R22)``. Positive pitch = nose down (rotation about the left axis), positive roll = right
  side down. Tilt = angle between the body z axis and the world vertical, ``arccos(R22)`` (independent of
  yaw; for small angles tilt² ≈ roll² + pitch²).
* Aggregation over bodies: ``worst_*`` is the maximum over the logged bodies, never an average
  (protocol §6).

Entity-specific values in the flat :func:`trial_metrics` dict are keyed ``<metric>@<entity>``
(e.g. ``tilt_p95_deg@segment 2``).
"""
from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

__all__ = [
    "FAILURE_REASONS", "achieved_speed", "actuator_demand", "belly_contacts", "body_angular_motion",
    "body_undulation", "circular_stats", "contact_force_feasibility", "cost_of_transport",
    "default_phase_pairs", "distance_before_failure", "duty_factor", "euler_zyx", "foot_loaded",
    "foot_normal_force", "foot_slip", "friction_pyramid", "interlimb_phases", "intersegment_angles",
    "lateral_deviation_and_heading", "phase_recovery", "quat_to_matrix", "recovery",
    "signed_support_distance", "support_margin", "tilt_angle", "trial_metrics", "trial_result", "weight",
]

FAILURE_REASONS = ("fall", "stall", "off_course")
"""Outcome reasons that are failures (protocol §5); ``timeout`` and ``success`` are not."""

_NAN = math.nan


# ----------------------------------------------------------------------------------------------- helpers
def _item(x: Any) -> Any:
    """Unwrap 0-d numpy arrays (what ``np.load`` returns for scalars, strings and pickled dicts)."""
    if isinstance(x, np.ndarray) and x.shape == ():
        return x.item()
    return x


def _names(seq) -> list[str]:
    return [str(s) for s in _item(seq)]


def _t(log) -> np.ndarray:
    return np.asarray(log["t"], dtype=float).reshape(-1)


def _g(log) -> float:
    return float(np.linalg.norm(np.asarray(log["gravity"], dtype=float)))


def weight(log) -> float:
    """Robot weight ``m |g|`` in N."""
    return float(_item(log["total_mass"])) * _g(log)


def _finite(x) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    return x[np.isfinite(x)]


def _rms(x) -> float:
    x = _finite(x)
    return float(np.sqrt(np.mean(x * x))) if x.size else _NAN


def _pct(x, q) -> float:
    x = _finite(x)
    return float(np.percentile(x, q)) if x.size else _NAN


def _max(x) -> float:
    x = _finite(x)
    return float(np.max(x)) if x.size else _NAN


def _min(x) -> float:
    x = _finite(x)
    return float(np.min(x)) if x.size else _NAN


def _mean(x) -> float:
    x = _finite(x)
    return float(np.mean(x)) if x.size else _NAN


def _sd(x) -> float:
    x = _finite(x)
    return float(np.std(x, ddof=1)) if x.size >= 2 else _NAN


def _integrate(y, t) -> float:
    """Trapezoidal ∫ y dt over the samples."""
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    if y.size < 2:
        return 0.0
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(t)))


def _dt(t) -> float:
    d = np.diff(np.asarray(t, dtype=float))
    d = d[d > 0]
    return float(np.median(d)) if d.size else _NAN


def _runs(mask) -> list[tuple[int, int]]:
    """``[(start, end_exclusive), ...]`` of the True runs of a boolean vector."""
    m = np.asarray(mask, dtype=bool).reshape(-1).astype(np.int8)
    d = np.diff(np.concatenate([[0], m, [0]]))
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def _rising(mask) -> np.ndarray:
    """Indices where a boolean vector switches False -> True (sample 0 is never an entry)."""
    m = np.asarray(mask, dtype=bool).reshape(-1)
    if m.size < 2:
        return np.zeros(0, dtype=int)
    return np.flatnonzero(m[1:] & ~m[:-1]) + 1


def _debounce(mask, t, min_run_s: float) -> np.ndarray:
    """Fill interior gaps shorter than ``min_run_s``, then drop interior runs shorter than it."""
    m = np.asarray(mask, dtype=bool).copy()
    if min_run_s <= 0 or m.size < 3:
        return m
    dt = _dt(t)
    n = int(round(min_run_s / dt)) if np.isfinite(dt) else 0
    if n <= 1:
        return m
    for s, e in _runs(~m):
        if s > 0 and e < m.size and e - s < n:
            m[s:e] = True
    for s, e in _runs(m):
        if s > 0 and e < m.size and e - s < n:
            m[s:e] = False
    return m


def _end_index(log, outcome) -> int:
    """Index of the last log sample at or before the outcome time (the whole log without an outcome)."""
    t = _t(log)
    if t.size == 0:
        return -1
    t_end = None if outcome is None else outcome.get("t_end")
    if t_end is None or not np.isfinite(float(t_end)):
        return t.size - 1
    k = int(np.searchsorted(t, float(t_end) + 1e-9, side="right")) - 1
    return int(np.clip(k, 0, t.size - 1))


def _disturbances(log) -> list[dict]:
    d = _item(log.get("disturbances", []))
    if d is None:
        return []
    return [dict(_item(x)) for x in list(d)]


# ------------------------------------------------------------------------------------------- orientation
def quat_to_matrix(q) -> np.ndarray:
    """Rotation matrices (..., 3, 3) of unit quaternions (..., 4) given as w, x, y, z (body -> world)."""
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = (q[..., i] for i in range(4))
    R = np.empty(q.shape[:-1] + (3, 3))
    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - w * z)
    R[..., 0, 2] = 2 * (x * z + w * y)
    R[..., 1, 0] = 2 * (x * y + w * z)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - w * x)
    R[..., 2, 0] = 2 * (x * z - w * y)
    R[..., 2, 1] = 2 * (y * z + w * x)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def euler_zyx(q) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(yaw, pitch, roll)`` in rad: intrinsic z-y-x Euler angles of quaternions (w, x, y, z).

    ``R = Rz(yaw) Ry(pitch) Rx(roll)``; yaw in (−π, π], pitch in [−π/2, π/2], roll in (−π, π].
    """
    R = quat_to_matrix(q)
    yaw = np.arctan2(R[..., 1, 0], R[..., 0, 0])
    pitch = np.arcsin(np.clip(-R[..., 2, 0], -1.0, 1.0))
    roll = np.arctan2(R[..., 2, 1], R[..., 2, 2])
    return yaw, pitch, roll


def tilt_angle(q) -> np.ndarray:
    """Angle (rad) between the body z axis and the world vertical."""
    R = quat_to_matrix(q)
    return np.arccos(np.clip(R[..., 2, 2], -1.0, 1.0))


# ---------------------------------------------------------------------------------------------- contacts
def foot_normal_force(log) -> np.ndarray:
    """(T, F) normal force N = f · n of the ground on each foot (N); ``f_z`` where the normal is nan."""
    f = np.asarray(log["foot_force"], dtype=float)
    n = np.asarray(log["foot_normal"], dtype=float)
    ok = np.all(np.isfinite(n), axis=-1)
    fn = np.einsum("...i,...i->...", f, np.where(ok[..., None], n, 0.0))
    return np.where(ok, fn, f[..., 2])


def foot_loaded(log, load_fraction: float = 0.02) -> np.ndarray:
    """(T, F) bool: normal force above ``load_fraction`` of the robot's weight (protocol §6: 2 %)."""
    return foot_normal_force(log) > load_fraction * weight(log)


def _contact_points(log) -> np.ndarray:
    """(T, F, 3) contact point, falling back to the foot centre where the log has no contact point."""
    fp = np.asarray(log["foot_pos"], dtype=float)
    cp = np.asarray(log.get("foot_contact_pos", fp), dtype=float)
    ok = np.all(np.isfinite(cp), axis=-1, keepdims=True)
    return np.where(ok, cp, fp)


def _normals(log) -> np.ndarray:
    """(T, F, 3) unit contact normals, world vertical where the log has none."""
    n = np.asarray(log["foot_normal"], dtype=float)
    ok = np.all(np.isfinite(n), axis=-1, keepdims=True)
    n = np.where(ok, n, np.array([0.0, 0.0, 1.0]))
    return n / np.linalg.norm(n, axis=-1, keepdims=True)


def _contact_record(log, load_fraction: float, min_run_s: float) -> tuple[np.ndarray, np.ndarray]:
    t = _t(log)
    loaded = foot_loaded(log, load_fraction)
    if min_run_s > 0 and loaded.ndim == 2 and loaded.shape[1]:
        loaded = np.stack([_debounce(loaded[:, i], t, min_run_s) for i in range(loaded.shape[1])], axis=1)
    return t, loaded


# ------------------------------------------------------------------------------------ progress and speed
def distance_before_failure(log, outcome: Mapping | None = None) -> dict:
    """COM progress along +x at the outcome (protocol §6).

    Returns ``progress_m`` = x_COM(t_end) − x_COM(0), ``x_end_m`` (absolute x of the COM at the outcome),
    ``t_end_s`` and ``completed`` (``outcome['success']``; None without an outcome) — completed trials are
    marked separately, their distance is the course.
    """
    com = np.asarray(log["com"], dtype=float)
    t = _t(log)
    k = _end_index(log, outcome)
    if k < 0:
        return {"progress_m": _NAN, "x_end_m": _NAN, "t_end_s": _NAN, "completed": None}
    completed = None
    if outcome is not None and outcome.get("success") is not None:
        completed = bool(outcome["success"])
    return {"progress_m": float(com[k, 0] - com[0, 0]), "x_end_m": float(com[k, 0]),
            "t_end_s": float(t[k]), "completed": completed}


def achieved_speed(log, outcome: Mapping | None = None) -> float:
    """COM progress / elapsed walking time to the outcome, m/s (protocol §6)."""
    t = _t(log)
    k = _end_index(log, outcome)
    if k < 1:
        return _NAN
    elapsed = t[k] - t[0]
    return float(distance_before_failure(log, outcome)["progress_m"] / elapsed) if elapsed > 0 else _NAN


# ---------------------------------------------------------------------------------- body angular motion
_BODY_METRICS = ("roll_rate_rms_rad_s", "pitch_rate_rms_rad_s", "tilt_p95_deg", "roll_abs_p95_deg",
                 "pitch_abs_p95_deg", "tilt_max_deg")


def body_angular_motion(log, payload: str | None = None) -> dict:
    """Per logged body: RMS roll and pitch rate, 95th percentiles of tilt, |roll| and |pitch| (protocol §6).

    * ``roll_rate_rms_rad_s``, ``pitch_rate_rms_rad_s``: RMS of the BODY-frame angular velocity ω_x, ω_y.
    * ``tilt_p95_deg``, ``tilt_max_deg``: angle of the body z axis from the vertical.
    * ``roll_abs_p95_deg``, ``pitch_abs_p95_deg``: |roll|, |pitch| of the intrinsic z-y-x Euler angles
      (module docstring).

    Returns ``{'per_body': {body: {metric: value}}, 'worst_<metric>': max over bodies,
    'worst_<metric>_body': which body, 'payload_body', 'payload_<metric>'}``. ``payload`` names the payload
    body (Cleopatra: ``'head'``); without it (and without a ``'payload'`` key in the log) no payload values
    are reported — nothing is guessed. The worst body is the maximum per metric, never an average.
    """
    bodies = _names(log["bodies"])
    q = np.asarray(log["body_quat"], dtype=float)
    w = np.asarray(log["body_angvel"], dtype=float)
    per: dict[str, dict] = {}
    if q.size:
        _, pitch, roll = euler_zyx(q)
        tilt = tilt_angle(q)
    for b, name in enumerate(bodies):
        if not q.size:
            per[name] = {m: _NAN for m in _BODY_METRICS}
            continue
        per[name] = {
            "roll_rate_rms_rad_s": _rms(w[:, b, 0]),
            "pitch_rate_rms_rad_s": _rms(w[:, b, 1]),
            "tilt_p95_deg": math.degrees(_pct(tilt[:, b], 95)),
            "roll_abs_p95_deg": math.degrees(_pct(np.abs(roll[:, b]), 95)),
            "pitch_abs_p95_deg": math.degrees(_pct(np.abs(pitch[:, b]), 95)),
            "tilt_max_deg": math.degrees(_max(tilt[:, b])),
        }
    out: dict[str, Any] = {"per_body": per}
    for m in _BODY_METRICS:
        vals = np.array([per[n][m] for n in bodies], dtype=float)
        if vals.size and np.any(np.isfinite(vals)):
            i = int(np.nanargmax(vals))
            out[f"worst_{m}"], out[f"worst_{m}_body"] = float(vals[i]), bodies[i]
        else:
            out[f"worst_{m}"], out[f"worst_{m}_body"] = _NAN, None
    payload = payload if payload is not None else _item(log.get("payload"))
    if payload is not None:
        if str(payload) not in per:
            raise ValueError(f"payload body {payload!r} is not a logged body {bodies}")
        out["payload_body"] = str(payload)
        for m in _BODY_METRICS:
            out[f"payload_{m}"] = per[str(payload)][m]
    return out


# ----------------------------------------------------------------------------------- intersegment angles
def _body_joint_indices(log, kinds: Iterable[str] | None = None) -> list[int]:
    kind = _names(log["joint_kind"])
    if kinds is None:
        return [j for j, k in enumerate(kind) if k.startswith("body_")]
    kinds = set(kinds)
    return [j for j, k in enumerate(kind) if k in kinds]


def intersegment_angles(log, near_limit_deg: float = 1.0) -> dict:
    """Per body joint (tag ``body_*``; one hinge = one axis): angle statistics (protocol §6).

    * ``max_abs_deg``, ``p95_abs_deg``: max and 95th percentile of |q|.
    * ``near_limit_fraction``: fraction of samples within ``near_limit_deg`` of either limit of ``q_range``
      (nan when the joint has no finite limit).
    * ``limit_hits``: entries into that band (False -> True transitions; a joint already in the band at the
      first sample counts as one hit).

    Returns ``{'per_joint': {joint: {...,'kind'}}, 'per_kind': {kind: {...}}}`` where per kind the angles and
    fractions are the maximum over the joints of that kind and ``limit_hits`` is their sum. A robot without
    body joints (``locked``) returns empty dicts.
    """
    joints = _names(log["joints"])
    kinds = _names(log["joint_kind"])
    q = np.asarray(log["q"], dtype=float)
    rng = np.asarray(log["q_range"], dtype=float)
    band = math.radians(near_limit_deg)
    per: dict[str, dict] = {}
    for j in _body_joint_indices(log):
        a = q[:, j] if q.size else np.zeros(0)
        lo, hi = rng[j]
        if a.size and np.isfinite(lo) and np.isfinite(hi):
            near = (a >= hi - band) | (a <= lo + band)
            frac = float(np.mean(near))
            hits = int(_rising(near).size + (1 if near[0] else 0))
        else:
            frac, hits = _NAN, 0
        per[joints[j]] = {"kind": kinds[j], "max_abs_deg": math.degrees(_max(np.abs(a))),
                          "p95_abs_deg": math.degrees(_pct(np.abs(a), 95)), "near_limit_fraction": frac,
                          "limit_hits": hits}
    per_kind: dict[str, dict] = {}
    for name, v in per.items():
        k = per_kind.setdefault(v["kind"], {"max_abs_deg": _NAN, "p95_abs_deg": _NAN,
                                            "near_limit_fraction": _NAN, "limit_hits": 0, "joints": []})
        for m in ("max_abs_deg", "p95_abs_deg", "near_limit_fraction"):
            k[m] = float(np.nanmax([k[m], v[m]])) if np.isfinite([k[m], v[m]]).any() else _NAN
        k["limit_hits"] += v["limit_hits"]
        k["joints"].append(name)
    return {"per_joint": per, "per_kind": per_kind}


# ---------------------------------------------------------------------------------------- belly contact
def belly_contacts(log) -> dict:
    """Fraction of time any body-role geom (shell, head) touches the ground, overall and per body (§6).

    ``events`` counts the touchdowns of "any belly contact" (False -> True transitions).
    """
    bodies = _names(log["bodies"])
    bc = log.get("belly_contact")
    if bc is None:
        return {"fraction": _NAN, "events": 0, "per_body": {b: _NAN for b in bodies}}
    bc = np.asarray(bc, dtype=bool)
    if bc.size == 0:
        return {"fraction": _NAN, "events": 0, "per_body": {b: _NAN for b in bodies}}
    anyb = bc.any(axis=1)
    return {"fraction": float(anyb.mean()), "events": int(_rising(anyb).size + (1 if anyb[0] else 0)),
            "per_body": {b: float(bc[:, i].mean()) for i, b in enumerate(bodies)}}


# -------------------------------------------------------------------------------------------- foot slip
def foot_slip(log, load_fraction: float = 0.02, slip_tol: float = 0.005,
              outcome: Mapping | None = None) -> dict:
    """Foot slip per loaded contact episode (protocol §6).

    A loaded contact episode is a maximal run of samples with ``N > load_fraction · m|g|`` for one foot.
    Its slip is the tangential path length of the foot centre in the contact plane:
    ``Σ_k |Δp_k − (Δp_k · n_k) n_k|`` over consecutive samples of the episode, with ``n_k`` the normalised
    mean of the two samples' contact normals. Motion while unloaded is not counted. (For a spherical pad the
    centre also moves when the pad rolls; the protocol measures the centre.)

    Returns ``slip_per_m`` (total slip / COM progress; nan when the progress is not positive),
    ``slip_p95_m`` (95th percentile over episodes), ``fraction_over_tol`` (episodes slipping more than
    ``slip_tol``, default 5 mm), ``slip_total_m``, ``n_episodes``, ``per_foot_m`` (total per foot) and
    ``episodes`` (list of dicts foot, t_start, t_end, slip_m).
    """
    t = _t(log)
    feet = _names(log["feet"])
    loaded = foot_loaded(log, load_fraction)
    p = np.asarray(log["foot_pos"], dtype=float)
    n = _normals(log)
    episodes = []
    per_foot = {}
    for i, name in enumerate(feet):
        total = 0.0
        for s, e in _runs(loaded[:, i]) if loaded.size else []:
            slip = 0.0
            if e - s >= 2:
                dp = np.diff(p[s:e, i], axis=0)
                nn = n[s:e - 1, i] + n[s + 1:e, i]
                nn /= np.linalg.norm(nn, axis=-1, keepdims=True)
                tan = dp - np.sum(dp * nn, axis=-1, keepdims=True) * nn
                slip = float(np.sum(np.linalg.norm(tan, axis=-1)))
            episodes.append({"foot": name, "t_start": float(t[s]), "t_end": float(t[e - 1]), "slip_m": slip})
            total += slip
        per_foot[name] = total
    s_ep = np.array([e["slip_m"] for e in episodes], dtype=float)
    total = float(s_ep.sum()) if s_ep.size else 0.0
    progress = distance_before_failure(log, outcome)["progress_m"]
    return {
        "slip_total_m": total,
        "slip_per_m": total / progress if np.isfinite(progress) and progress > 0 else _NAN,
        "slip_p95_m": _pct(s_ep, 95),
        "fraction_over_tol": float(np.mean(s_ep > slip_tol)) if s_ep.size else _NAN,
        "n_episodes": int(s_ep.size),
        "slip_tol_m": slip_tol,
        "per_foot_m": per_foot,
        "episodes": episodes,
    }


# ------------------------------------------------------------------------------------- cost of transport
def cost_of_transport(log, outcome: Mapping | None = None) -> dict:
    """Mechanical and (estimated) electrical cost of transport (protocol §6).

    * ``energy_pos_J`` = ``E₊ = ∫ Σ_j max(τ_j q̇_j, 0) dt`` over the ACTUATED joints only (``joint_active``:
      every leg joint and any active body joint; passive spring/damper joints are excluded — they do no
      actuator work). ``cot_mech = E₊ / (m g d)`` with d the COM progress.
    * Electrical estimate (labelled an estimate): DC-motor model from the stall data,
      ``k_t = τ_stall / I_stall``, ``R = V / I_stall``, ``I = |τ| / k_t``, ``P_j = max(τ_j q̇_j + I_j² R_j, 0)``
      per actuator (no regeneration); ``energy_el_J = ∫ Σ_j P_j dt``, ``cot_el_est = E_el / (m g d)``.

    Integration is trapezoidal over the log samples up to the outcome. Costs are nan when d ≤ 0.
    """
    t = _t(log)
    k = _end_index(log, outcome)
    if k < 1:
        return {"energy_pos_J": _NAN, "cot_mech": _NAN, "energy_el_J": _NAN, "cot_el_est": _NAN,
                "power_mech_mean_W": _NAN, "power_el_mean_W": _NAN, "distance_m": _NAN, "duration_s": _NAN}
    act = np.asarray(log["joint_active"], dtype=bool)
    tau = np.asarray(log["tau"], dtype=float)[: k + 1][:, act]
    qd = np.asarray(log["qd"], dtype=float)[: k + 1][:, act]
    tt = t[: k + 1]
    p_mech = tau * qd
    e_pos = _integrate(np.maximum(p_mech, 0.0).sum(axis=1), tt)
    stall = np.asarray(log["tau_stall"], dtype=float)[act]
    i_stall = np.asarray(log["i_stall"], dtype=float)[act]
    volt = np.asarray(log["voltage"], dtype=float)[act]
    with np.errstate(divide="ignore", invalid="ignore"):
        kt = stall / i_stall
        res = volt / i_stall
        cur = np.abs(tau) / kt
        p_el = np.maximum(p_mech + cur * cur * res, 0.0)
    e_el = _integrate(p_el.sum(axis=1), tt) if np.all(np.isfinite(kt) & np.isfinite(res)) else _NAN
    d = float(np.asarray(log["com"], dtype=float)[k, 0] - np.asarray(log["com"], dtype=float)[0, 0])
    mgd = float(_item(log["total_mass"])) * _g(log) * d
    dur = float(tt[-1] - tt[0])
    return {
        "energy_pos_J": e_pos,
        "cot_mech": e_pos / mgd if d > 0 else _NAN,
        "energy_el_J": e_el,
        "cot_el_est": e_el / mgd if d > 0 and np.isfinite(e_el) else _NAN,
        "power_mech_mean_W": e_pos / dur if dur > 0 else _NAN,
        "power_el_mean_W": e_el / dur if dur > 0 else _NAN,
        "distance_m": d,
        "duration_s": dur,
    }


# ------------------------------------------------------------------------------------- actuator demand
def _group_of(kind: str) -> str:
    if kind.startswith("body_"):
        return "body"
    return kind or "other"


def actuator_demand(log, saturation: float = 0.98) -> dict:
    """Actuator demand per joint group of the actuated joints (protocol §6).

    Groups: the joint tag (``hip_yaw``, ``hip_pitch``, ``knee``, ``hip_roll``, ...), all active body joints
    together as ``body``. Per group:

    * ``peak_tau_Nm``: max |τ|; ``rms_tau_Nm``: RMS of τ pooled over the group's joints and samples;
      ``rms_tau_worst_Nm``: the largest per-joint RMS.
    * ``rms_over_rated`` / ``rms_over_rated_worst``: the same for τ / τ_rated (thermal torque).
    * ``torque_sat_fraction``: fraction of joint-samples with |τ| ≥ ``saturation`` × the speed-dependent
      limit ``τ_stall · max(1 − |q̇|/ω₀, 0)`` (at |q̇| ≥ ω₀ the limit is 0 and the sample counts as
      saturated); ``speed_sat_fraction``: fraction with |q̇| ≥ ``saturation`` · ω₀; ``*_worst``: the joint with
      the largest fraction.
    """
    joints = _names(log["joints"])
    kinds = _names(log["joint_kind"])
    act = np.asarray(log["joint_active"], dtype=bool)
    tau = np.asarray(log["tau"], dtype=float)
    qd = np.asarray(log["qd"], dtype=float)
    stall = np.asarray(log["tau_stall"], dtype=float)
    rated = np.asarray(log["tau_rated"], dtype=float)
    w0 = np.asarray(log["qd_noload"], dtype=float)
    groups: dict[str, list[int]] = {}
    for j in np.flatnonzero(act):
        groups.setdefault(_group_of(kinds[j]), []).append(int(j))
    out: dict[str, dict] = {}
    for g, idx in groups.items():
        tg, qg = tau[:, idx], qd[:, idx]
        with np.errstate(divide="ignore", invalid="ignore"):
            lim = stall[idx] * np.clip(1.0 - np.abs(qg) / w0[idx], 0.0, 1.0)
            tsat = np.abs(tg) >= saturation * lim
            ssat = np.abs(qg) >= saturation * w0[idx]
            ratio = tg / rated[idx]
        out[g] = {
            "joints": [joints[j] for j in idx],
            "peak_tau_Nm": _max(np.abs(tg)),
            "rms_tau_Nm": _rms(tg),
            "rms_tau_worst_Nm": _max([_rms(tg[:, i]) for i in range(len(idx))]),
            "rms_over_rated": _rms(ratio),
            "rms_over_rated_worst": _max([_rms(ratio[:, i]) for i in range(len(idx))]),
            "torque_sat_fraction": float(tsat.mean()) if tsat.size else _NAN,
            "torque_sat_fraction_worst": float(tsat.mean(axis=0).max()) if tsat.size else _NAN,
            "speed_sat_fraction": float(ssat.mean()) if ssat.size else _NAN,
            "speed_sat_fraction_worst": float(ssat.mean(axis=0).max()) if ssat.size else _NAN,
        }
    return out


# -------------------------------------------------------------------------------------- support margin
def _cross2(o, a, b) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _hull2d(points) -> np.ndarray:
    """Convex hull (counter-clockwise, collinear points removed) of 2-D points; 1 or 2 rows if degenerate."""
    pts = np.unique(np.asarray(points, dtype=float).reshape(-1, 2), axis=0)
    if len(pts) <= 2:
        return pts
    eps = 1e-14
    lower: list = []
    for p in pts:
        while len(lower) >= 2 and _cross2(lower[-2], lower[-1], p) <= eps:
            lower.pop()
        lower.append(p)
    upper: list = []
    for p in pts[::-1]:
        while len(upper) >= 2 and _cross2(upper[-2], upper[-1], p) <= eps:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    return np.array(hull)


def _seg_dist(c, a, b) -> float:
    ab = b - a
    L2 = float(ab @ ab)
    s = 0.0 if L2 == 0 else float(np.clip((c - a) @ ab / L2, 0.0, 1.0))
    return float(np.linalg.norm(c - (a + s * ab)))


_ON_EDGE = 1e-12   # m: distances below this are rounding residue and count as on the boundary (s = 0)


def signed_support_distance(c, points) -> float:
    """Signed distance (m) from the point ``c`` (x, y) to the boundary of the convex hull of ``points``.

    Positive inside, negative outside, 0 on the boundary. Fewer than three non-collinear points: minus the
    distance to their hull (a segment or a point — never positive). No points: nan.
    """
    c = np.asarray(c, dtype=float)[:2]
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(pts) == 0:
        return _NAN
    h = _hull2d(pts)
    if len(h) == 1:
        s = -float(np.linalg.norm(c - h[0]))
    elif len(h) == 2:
        s = -_seg_dist(c, h[0], h[1])
    else:
        a, b = h, np.roll(h, -1, axis=0)
        e = b - a
        cr = e[:, 0] * (c[1] - a[:, 1]) - e[:, 1] * (c[0] - a[:, 0])
        if np.all(cr >= -_ON_EDGE * np.linalg.norm(e, axis=1)):
            s = float(np.min(cr / np.linalg.norm(e, axis=1)))
        else:
            s = -min(_seg_dist(c, a[i], b[i]) for i in range(len(h)))
    return 0.0 if abs(s) < _ON_EDGE else s


def support_margin(log, load_fraction: float = 0.02) -> dict:
    """Support margin s(t) (protocol §6): signed distance from the COM's ground projection to the boundary
    of the convex hull of the loaded contacts' (x, y) — positive inside; with fewer than three (or only
    collinear) loaded contacts minus the distance to their hull; nan without loaded contacts.

    A diagnostic on uneven ground, not a fall criterion (Bretl & Lall 2008). Returns the series ``s`` and
    ``n_contacts`` and the summary ``min_m``, ``p5_m``, ``outside_fraction`` (s < 0, over samples with at
    least one loaded contact) and ``no_contact_fraction``.
    """
    loaded = foot_loaded(log, load_fraction)
    cp = _contact_points(log)
    com = np.asarray(log["com"], dtype=float)
    T = com.shape[0]
    s = np.full(T, _NAN)
    for k in range(T):
        idx = np.flatnonzero(loaded[k]) if loaded.size else []
        if len(idx):
            s[k] = signed_support_distance(com[k, :2], cp[k, idx, :2])
    fin = np.isfinite(s)
    return {
        "t": _t(log), "s": s, "n_contacts": loaded.sum(axis=1) if loaded.size else np.zeros(T, int),
        "min_m": _min(s), "p5_m": _pct(s, 5),
        "outside_fraction": float(np.mean(s[fin] < 0)) if fin.any() else _NAN,
        "no_contact_fraction": float(np.mean(~fin)) if T else _NAN,
    }


# --------------------------------------------------------------------------- contact-force feasibility
def _tangent_basis(n) -> tuple[np.ndarray, np.ndarray]:
    """Tangent vectors (t1, t2) of a contact with unit normal n: t1 is world x projected into the contact
    plane (world y when n is nearly along x), t2 = n × t1 — a right-handed (t1, t2, n) frame."""
    ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    t1 = ref - (ref @ n) * n
    t1 /= np.linalg.norm(t1)
    return t1, np.cross(n, t1)


_N_FACETS = 8


def friction_pyramid(n, mu: float, kind: str = "inscribed") -> np.ndarray:
    """(3, 8) edge generators of the 8-facet linearised friction cone of a contact with normal ``n``.

    A force in the pyramid is ``f = Σ_j λ_j g_j``, λ ≥ 0. ``inscribed`` (default, the usual linearisation):
    edges ON the cone, ``g_j = n + μ (cos θ_j t1 + sin θ_j t2)``, θ_j = j·45°, so every pyramid force obeys
    Coulomb's law (|f_t| ≤ μ f_n), full μ along ±t1, ±t2 and μ·cos 22.5° between them. ``circumscribed``:
    facets tangent to the cone (edges at 22.5° + j·45° with radius μ / cos 22.5°), containing the cone.
    """
    n = np.asarray(n, dtype=float)
    n = n / np.linalg.norm(n)
    t1, t2 = _tangent_basis(n)
    if kind == "inscribed":
        th, r = np.arange(_N_FACETS) * 2 * np.pi / _N_FACETS, mu
    elif kind == "circumscribed":
        th, r = (np.arange(_N_FACETS) + 0.5) * 2 * np.pi / _N_FACETS, mu / math.cos(math.pi / _N_FACETS)
    else:
        raise ValueError("pyramid must be 'inscribed' or 'circumscribed'")
    return n[:, None] + r * (np.cos(th)[None, :] * t1[:, None] + np.sin(th)[None, :] * t2[:, None])


def _skew(r) -> np.ndarray:
    return np.array([[0.0, -r[2], r[1]], [r[2], 0.0, -r[0]], [-r[1], r[0], 0.0]])


def _wrench_feasible(gens, levers, F_req, M_req, act=None, tol=(0.0, 0.0)) -> bool | None:
    """LP: do λ ≥ 0 exist with Σ G_i λ_i = F_req and Σ r_i × (G_i λ_i) = M_req, each component within
    ``tol = (force N, moment N m)`` (slack variables), and |J_iᵀ G_i λ_i| ≤ τ for the legs in ``act``?

    Returns True / False, or None when the solver fails for another reason.
    """
    from scipy.optimize import linprog

    nc = len(gens)
    nv = _N_FACETS * nc
    A_eq = np.zeros((6, nv + 6))
    for i, (G, r) in enumerate(zip(gens, levers)):
        A_eq[:3, i * _N_FACETS:(i + 1) * _N_FACETS] = G
        A_eq[3:, i * _N_FACETS:(i + 1) * _N_FACETS] = _skew(r) @ G
    A_eq[:, nv:] = -np.eye(6)                                     # Σ ... − slack = required
    b_eq = np.concatenate([F_req, M_req])
    bounds = [(0, None)] * nv + [(-tol[0], tol[0])] * 3 + [(-tol[1], tol[1])] * 3
    A_ub = b_ub = None
    if act:
        rows, rhs = [], []
        for i, (Jac, lim) in act.items():
            JG = Jac.T @ gens[i]                                  # (n, 8): joint torques per generator
            for sign in (1.0, -1.0):
                blk = np.zeros((JG.shape[0], nv + 6))
                blk[:, i * _N_FACETS:(i + 1) * _N_FACETS] = sign * JG
                rows.append(blk)
                rhs.append(lim)
        if rows:
            A_ub, b_ub = np.vstack(rows), np.concatenate(rhs)
    c = np.r_[np.ones(nv), np.zeros(6)]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if res.status == 0:
        return True
    if res.status == 2:
        return False
    return None


def _push_wrench(log, t, com) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Known external push forces (the log's disturbances) as seen by the central finite differences.

    ``np.gradient`` gives a_k = Δv over [t_{k−1}, t_{k+1}] / its length, i.e. the mean of all forces over that
    stencil; the consistent push force at sample k is therefore F · overlap([t_start, t_start + duration],
    stencil) / stencil length, with F = impulse / duration along ``direction`` at the pushed body's COM
    (``body_pos``). Returns force (T, 3), moment about the COM (T, 3) and a (T,) mask of samples touched by a
    push whose body is not logged (those cannot be evaluated dynamically).
    """
    T = t.size
    F = np.zeros((T, 3))
    Mo = np.zeros((T, 3))
    unknown = np.zeros(T, bool)
    if T < 2:
        return F, Mo, unknown
    lo = t[np.clip(np.arange(T) - 1, 0, T - 1)]
    hi = t[np.clip(np.arange(T) + 1, 0, T - 1)]
    bodies = _names(log["bodies"]) if "bodies" in log else []
    bpos = np.asarray(log["body_pos"], dtype=float) if "body_pos" in log else None
    for d in _disturbances(log):
        t0, dur = float(d["t_start"]), float(d.get("duration", 0.0))
        imp = d.get("impulse_Ns")
        if dur <= 0 or imp is None:
            continue
        u = np.asarray(d.get("direction", (0.0, 1.0, 0.0)), dtype=float)
        u = u / np.linalg.norm(u)
        frac = np.clip(np.minimum(hi, t0 + dur) - np.maximum(lo, t0), 0.0, None) / (hi - lo)
        hit = frac > 0
        if str(d.get("body")) not in bodies or bpos is None:
            unknown |= hit
            continue
        f = (float(imp) / dur) * frac[:, None] * u[None, :]
        F += f
        Mo += np.cross(bpos[:, bodies.index(str(d.get("body")))] - com, f)
    return F, Mo, unknown


def contact_force_feasibility(log, every: int = 1, *, load_fraction: float = 0.02, mu=None,
                              actuator_limits: bool = True, exclude_belly: bool = True,
                              pyramid: str = "inscribed", wrench_tol=None) -> dict:
    """Second-stage stability test: contact-force feasibility by linear programming (protocol §6).

    At every ``every``-th log sample, with the loaded contacts (normal force > ``load_fraction`` of the
    weight) at their logged contact points p_i and logged normals n_i, ask whether contact forces f_i exist
    in 8-facet friction pyramids (μ of the contact, ``foot_mu``, or ``mu`` to override; see
    :func:`friction_pyramid`, tangent basis from each contact's own normal) that produce the wrench the
    motion needs. Signs: f_i is the force ON the foot (robot) FROM the ground; c is the whole-robot COM;
    g the gravity vector (0, 0, −9.81).

    * ``quasi_static``: ``Σ f_i = −m g`` and ``Σ (p_i − c) × f_i = 0``;
    * ``dynamic``: ``Σ f_i = m (a_COM − g) − F_push`` and ``Σ (p_i − c) × f_i = dL/dt − (p_b − c) × F_push``,
      with a_COM and dL/dt the central finite differences (``np.gradient``) of the logged COM velocity and
      angular momentum about the COM, and F_push the logged disturbances (force impulse/duration at the
      pushed body's COM p_b, averaged over the same difference stencil — :func:`_push_wrench`); the
      quasi-static variant ignores pushes (it asks whether the robot could stand still);
    * ``*_act`` (``actuator_limits=True``): additionally ``|J_iᵀ f_i| ≤ τ_stall`` per joint of each leg, with
      the logged per-foot Jacobians (d foot centre / d q of that leg's actuated joints; the pad radius
      moment and the leg's own inertia are neglected). Skipped when the log has no ``foot_jac``.

    Resolution (``wrench_tol``): every component of the produced wrench must match the required one within
    ``(force tolerance N, moment tolerance N m)``; default ``(load_fraction · m|g|, load_fraction · m|g| ·
    nominal_hip_height)`` — the contact record's own resolution (a contact below 2 % of the weight counts
    as unloaded and is left out). An exact equality is meaningless on logged data: with two (or collinear)
    contacts the moment about the contact line is fixed by the total force, so finite-difference noise alone
    would make every such sample infeasible (a trotting dog's whole stance). ``wrench_tol=(0, 0)`` gives the
    exact test. With the default the quasi-static COM may sit up to ``load_fraction · nominal_hip_height``
    (3 mm for Cleopatra) off a two-contact line.

    Only feet carry contact forces in the log, so samples with a belly contact cannot be represented; with
    ``exclude_belly`` (default) they are skipped and counted in ``n_belly_skipped``. Without loaded
    contacts a sample is feasible only if the required wrench is within the tolerance — never in the
    quasi-static case.

    The dynamic variant on simulated data: samples where a foot slides have their friction on the cone
    boundary, and finite-difference residuals can push them outside — those count as infeasible (the motion
    needs at least all the available friction).

    Returns ``{'t', 'index', 'status': {variant: (K,) float 1 feasible / 0 infeasible / nan not solved},
    'infeasible_fraction': {variant: float}, 'n_evaluated': {variant: int}, 'n_belly_skipped',
    'n_solver_failures', 'pyramid', 'wrench_tol'}``.
    """
    t = _t(log)
    T = t.size
    every = max(int(every), 1)
    idx = np.arange(0, T, every)
    m = float(_item(log["total_mass"]))
    gvec = np.asarray(log["gravity"], dtype=float)
    W = m * float(np.linalg.norm(gvec))
    loaded = foot_loaded(log, load_fraction)
    cp = _contact_points(log)
    nrm = _normals(log)
    com = np.asarray(log["com"], dtype=float)
    F = len(_names(log["feet"]))
    mus = np.broadcast_to(np.asarray(log["foot_mu"] if mu is None else mu, dtype=float), (F,))
    if T >= 2:
        acc = np.gradient(np.asarray(log["com_vel"], dtype=float), t, axis=0)
        dL = np.gradient(np.asarray(log["ang_mom"], dtype=float), t, axis=0)
    else:
        acc = np.full((T, 3), _NAN)
        dL = np.full((T, 3), _NAN)
    F_push, M_push, push_unknown = _push_wrench(log, t, com)
    belly = np.asarray(log.get("belly_contact", np.zeros((T, 1), bool)), dtype=bool)
    belly_any = belly.any(axis=1) if belly.ndim == 2 and belly.shape[1] else np.zeros(T, bool)
    actuator_limits = actuator_limits and all(k in log for k in ("foot_jac", "foot_joints", "tau_stall", "joints"))
    variants = ["quasi_static", "dynamic"] + (["quasi_static_act", "dynamic_act"] if actuator_limits else [])
    status = {v: np.full(idx.size, _NAN) for v in variants}
    if wrench_tol is None:
        h_ref = float(_item(log["nominal_hip_height"]))
        wrench_tol = (load_fraction * W, load_fraction * W * h_ref)
    tol = (float(wrench_tol[0]), float(wrench_tol[1]))
    act_data = None
    if actuator_limits:
        joints = _names(log["joints"])
        jix = {j: i for i, j in enumerate(joints)}
        stall = np.asarray(log["tau_stall"], dtype=float)
        jac = np.asarray(log["foot_jac"], dtype=float)
        fj = [[str(x) for x in list(_item(log["foot_joints"])[i])] for i in range(F)]
        act_data = (jac, [stall[[jix[j] for j in fj[i]]] for i in range(F)], [len(fj[i]) for i in range(F)])
    n_skip = n_fail = 0
    for kk, k in enumerate(idx):
        if exclude_belly and belly_any[k]:
            n_skip += 1
            continue
        c = com[k]
        feet = np.flatnonzero(loaded[k]) if loaded.size else np.zeros(0, int)
        gens = [friction_pyramid(nrm[k, i], mus[i], pyramid) for i in feet]
        levers = [cp[k, i] - c for i in feet]
        act = None
        if act_data is not None:
            jac, lims, nj = act_data
            act = {a: (jac[k, i, :, :nj[i]], lims[i]) for a, i in enumerate(feet)}
        reqs = {"quasi_static": (-m * gvec, np.zeros(3)),
                "dynamic": ((m * (acc[k] - gvec) - F_push[k], dL[k] - M_push[k]) if not push_unknown[k]
                            else (np.full(3, _NAN), np.full(3, _NAN)))}
        for v in variants:
            base = v.replace("_act", "")
            F_req, M_req = reqs[base]
            if not (np.all(np.isfinite(F_req)) and np.all(np.isfinite(M_req))):
                continue
            if v.endswith("_act") and status[base][kk] == 0.0:
                status[v][kk] = 0.0                                      # a subset of an infeasible set
                continue
            if feet.size == 0:
                ok = bool(np.all(np.abs(F_req) <= tol[0]) and np.all(np.abs(M_req) <= tol[1]))
            else:
                ok = _wrench_feasible(gens, levers, F_req, M_req, act if v.endswith("_act") else None, tol)
            if ok is None:
                n_fail += 1
                continue
            status[v][kk] = 1.0 if ok else 0.0
    frac, n_ev = {}, {}
    for v in variants:
        s = status[v][np.isfinite(status[v])]
        n_ev[v] = int(s.size)
        frac[v] = float(np.mean(s == 0.0)) if s.size else _NAN
    return {"t": t[idx], "index": idx, "status": status, "infeasible_fraction": frac, "n_evaluated": n_ev,
            "n_belly_skipped": n_skip, "n_solver_failures": n_fail, "pyramid": pyramid, "wrench_tol": tol}


# ---------------------------------------------------------------------------------------------- recovery
def _centered_mean(x, t, window_s: float) -> np.ndarray:
    """Centred moving average over ``window_s`` (samples within ±window/2; truncated at the ends)."""
    x = np.asarray(x, dtype=float)
    dt = _dt(t)
    half = int(round(0.5 * window_s / dt)) if np.isfinite(dt) else 0
    if half <= 0 or x.size == 0:
        return x.copy()
    c = np.concatenate([[0.0], np.cumsum(x)])
    i = np.arange(x.size)
    lo = np.clip(i - half, 0, x.size)
    hi = np.clip(i + half + 1, 0, x.size)
    return (c[hi] - c[lo]) / (hi - lo)


def recovery(log, outcome: Mapping | None = None, disturbance: Mapping | None = None, *,
             speed_window_s: float = 0.5, speed_tol: float = 0.30, tilt_limit_deg: float = 10.0,
             hold_s: float = 1.0, within_s: float = 5.0, pre_window_s: float = 2.0) -> dict:
    """Recovery after a push (protocol §6).

    Recovered if, within ``within_s`` (5 s) after the push ends, the ``speed_window_s`` (0.5 s) moving
    average of the forward COM speed (``com_vel`` x; centred window) is within ±``speed_tol`` (30 %) of its
    pre-push mean AND every logged body's tilt is within ``tilt_limit_deg`` (10°), both held for ``hold_s``
    (1.0 s), with no failure. Recovery time = start of that held interval − end of the push.

    Choices the protocol leaves open, fixed here: the pre-push mean is the mean forward COM speed over
    ``[t_start − pre_window_s, t_start)`` (2.0 s); "within 5 s" applies to the START of the held interval
    (recovery time ≤ 5 s) and the whole held interval must be in the log; any failure outcome (fall,
    stall, off_course) means not recovered. ``disturbance`` defaults to the log's first disturbance.
    """
    if disturbance is None:
        ds = _disturbances(log)
        if not ds:
            raise ValueError("the log has no disturbance; pass one explicitly")
        disturbance = ds[0]
    t = _t(log)
    v = np.asarray(log["com_vel"], dtype=float)[:, 0] if t.size else np.zeros(0)
    t0 = float(disturbance["t_start"])
    tpe = t0 + float(disturbance.get("duration", 0.0))
    pre = (t >= t0 - pre_window_s) & (t < t0)
    v_pre = _mean(v[pre])
    reason = None if outcome is None else outcome.get("reason")
    failed = reason in FAILURE_REASONS
    imp = disturbance.get("impulse_Ns")
    base = {"impulse_Ns": _NAN if imp is None else float(imp), "t_push_end_s": tpe, "v_pre_m_s": v_pre,
            "failed": bool(failed), "reason": reason}
    if t.size == 0 or not np.isfinite(v_pre):
        return {**base, "recovered": False, "recovery_time_s": _NAN}
    vma = _centered_mean(v, t, speed_window_s)
    tilt = np.degrees(tilt_angle(np.asarray(log["body_quat"], dtype=float)))       # (T, B)
    ok = (np.abs(vma - v_pre) <= speed_tol * abs(v_pre)) & np.all(tilt <= tilt_limit_deg, axis=1)
    post = t >= tpe - 1e-9
    ok = ok & post
    rec_time = _NAN
    for s, e in _runs(ok):
        if t[s] - tpe > within_s + 1e-9:
            break
        if t[e - 1] - t[s] >= hold_s - 1e-9:
            rec_time = float(t[s] - tpe)
            break
    recovered = bool(np.isfinite(rec_time) and not failed)
    return {**base, "recovered": recovered, "recovery_time_s": rec_time if recovered else _NAN}


# ------------------------------------------------------------------------------------- gait: duty factor
def duty_factor(log, load_fraction: float = 0.02, min_run_s: float = 0.0) -> dict:
    """Measured duty factor per leg from the contact record (protocol §9.2).

    A stride runs from one touchdown (unloaded -> loaded) of the foot to its next; duty = loaded time /
    stride time. ``min_run_s`` > 0 first removes contact chatter (interior gaps, then interior contacts,
    shorter than it); 0 (default) uses the raw record. Returns per foot ``mean``, ``sd`` (sample SD) and
    ``n_strides`` and ``values``; and over all strides of all legs ``mean`` and ``sd``.
    """
    t, loaded = _contact_record(log, load_fraction, min_run_s)
    feet = _names(log["feet"])
    dt_i = np.diff(t, append=t[-1] + (_dt(t) if t.size > 1 else 0.0)) if t.size else t
    per, allv = {}, []
    for i, name in enumerate(feet):
        td = _rising(loaded[:, i]) if loaded.size else np.zeros(0, int)
        vals = []
        for a, b in zip(td[:-1], td[1:]):
            stride = t[b] - t[a]
            if stride > 0:
                vals.append(float(np.sum(dt_i[a:b] * loaded[a:b, i]) / stride))
        per[name] = {"mean": _mean(vals), "sd": _sd(vals), "n_strides": len(vals), "values": vals}
        allv += vals
    return {"per_foot": per, "mean": _mean(allv), "sd": _sd(allv), "n_strides": len(allv)}


# ------------------------------------------------------------------------------- gait: interlimb phases
def circular_stats(phases) -> dict:
    """Circular mean and spread of phases given as cycle fractions.

    ``mean_cycles`` in [0, 1), resultant length ``R``, circular SD ``sd_rad = √(−2 ln R)`` and
    ``sd_cycles = sd_rad / 2π``; nan for no finite values.
    """
    p = _finite(phases)
    if p.size == 0:
        return {"mean_cycles": _NAN, "R": _NAN, "sd_rad": _NAN, "sd_cycles": _NAN, "n": 0}
    z = np.mean(np.exp(2j * np.pi * p))
    R = min(float(abs(z)), 1.0)
    sd = math.sqrt(-2.0 * math.log(R)) if R > 0 else math.inf
    return {"mean_cycles": float((np.angle(z) / (2 * np.pi)) % 1.0), "R": R, "sd_rad": sd,
            "sd_cycles": sd / (2 * np.pi), "n": int(p.size)}


def _feet_local_xy(log) -> np.ndarray:
    """(F, 2) median foot position in its body's frame (x forward, y left) — for left/right, front/rear."""
    fp = np.asarray(log["foot_pos"], dtype=float)
    F = fp.shape[1]
    if "foot_body" in log and "body_quat" in log and "body_pos" in log:
        fb = np.asarray(log["foot_body"], dtype=int).reshape(F)
        R = quat_to_matrix(np.asarray(log["body_quat"], dtype=float)[:, fb])     # (T, F, 3, 3)
        rel = fp - np.asarray(log["body_pos"], dtype=float)[:, fb]
        loc = np.einsum("tfji,tfj->tfi", R, rel)                                  # Rᵀ (p − p_body)
    else:
        loc = fp - np.nanmean(fp, axis=1, keepdims=True)
    return np.nanmedian(loc[..., :2], axis=0)


def default_phase_pairs(log) -> list[tuple[str, str, str]]:
    """Relations ``(reference foot, foot, kind)`` built from the log's feet and ``foot_group``.

    Per group (the feet mounted on one body, in order of first appearance): the reference is the rear-most
    left foot (body frame from the median pose); every other foot of the group is ``ipsilateral`` (same
    side), ``contralateral`` (other side, same fore-aft rank as the reference) or ``diagonal`` (other side,
    other rank). Between consecutive groups: reference -> next group's reference, ``intersegmental``.
    For Cleopatra this is, per segment, RL->FL, RL->RR, RL->FR and RL(seg k)->RL(seg k+1).
    """
    feet = _names(log["feet"])
    groups = _names(log["foot_group"]) if "foot_group" in log else ["all"] * len(feet)
    loc = _feet_local_xy(log)
    order: list[str] = []
    for g in groups:
        if g not in order:
            order.append(g)
    pairs: list[tuple[str, str, str]] = []
    refs = {}
    for g in order:
        idx = [i for i in range(len(feet)) if groups[i] == g]
        left = sorted([i for i in idx if loc[i, 1] >= 0], key=lambda i: loc[i, 0])
        right = sorted([i for i in idx if loc[i, 1] < 0], key=lambda i: loc[i, 0])
        ref = left[0] if left else right[0]
        refs[g] = ref
        same, other = (left, right) if ref in left else (right, left)
        rank = same.index(ref)
        for i in idx:
            if i == ref:
                continue
            if i in same:
                kind = "ipsilateral"
            else:
                kind = "contralateral" if other.index(i) == rank else "diagonal"
            pairs.append((feet[ref], feet[i], kind))
    for g0, g1 in zip(order, order[1:]):
        pairs.append((feet[refs[g0]], feet[refs[g1]], "intersegmental"))
    return pairs


def _relation_strides(t, loaded, i_ref, i_other):
    """Reference strides (start, end) and the other foot's first touchdown phase in each (nan if none)."""
    tr = t[_rising(loaded[:, i_ref])]
    to = t[_rising(loaded[:, i_other])]
    starts, ends = tr[:-1], tr[1:]
    ph = np.full(starts.size, _NAN)
    for k in range(starts.size):
        j = int(np.searchsorted(to, starts[k] - 1e-12, side="left"))
        if j < to.size and to[j] < ends[k] - 1e-12:
            ph[k] = (to[j] - starts[k]) / (ends[k] - starts[k])
    return starts, ends, ph


def _normalise_pairs(log, pairs) -> list[tuple[str, str, str]]:
    if pairs is None:
        return default_phase_pairs(log)
    out = []
    for p in pairs:
        p = tuple(p)
        out.append((str(p[0]), str(p[1]), str(p[2]) if len(p) > 2 else "relation"))
    return out


def interlimb_phases(log, pairs=None, load_fraction: float = 0.02, min_run_s: float = 0.0) -> dict:
    """Interlimb phase relations from the contact record (protocol §9.2).

    For each relation (reference foot, foot): per stride of the reference foot (touchdown to touchdown),
    the touchdown phase of the other foot = (its first touchdown in the stride − stride start) / stride
    time, in cycles [0, 1) (nan if it does not touch down in that stride). Then the circular mean and the
    circular SD √(−2 ln R) (:func:`circular_stats`). ``pairs``: list of ``(ref, foot[, kind])``; default
    :func:`default_phase_pairs`. Returns ``{'relations': {label: {ref, foot, kind, stride_start, phases,
    mean_cycles, R, sd_rad, sd_cycles, n}}}`` with label ``'<kind>:<ref>-><foot>'``.
    """
    t, loaded = _contact_record(log, load_fraction, min_run_s)
    feet = _names(log["feet"])
    fi = {f: i for i, f in enumerate(feet)}
    rel = {}
    for ref, foot, kind in _normalise_pairs(log, pairs):
        starts, ends, ph = _relation_strides(t, loaded, fi[ref], fi[foot])
        rel[f"{kind}:{ref}->{foot}"] = {"ref": ref, "foot": foot, "kind": kind, "stride_start": starts,
                                         "stride_end": ends, "phases": ph, **circular_stats(ph)}
    return {"relations": rel}


def _wrap_cycles(d):
    return (np.asarray(d, dtype=float) + 0.5) % 1.0 - 0.5


def phase_recovery(log, disturbance: Mapping | None = None, pairs=None, *, tol_cycles: float = 0.1,
                   n_strides: int = 2, load_fraction: float = 0.02, min_run_s: float = 0.0) -> dict:
    """Phase recovery after a push (protocol §9.2).

    Per relation: the pre-push circular mean is taken over the reference strides that END before the push
    starts; after the push, the relation is back when ``n_strides`` (2) consecutive reference strides that
    START at or after the push end have their phase within ±``tol_cycles`` (0.1 cycle) of that mean; its
    time is the start of the first of them − the push end. Phase recovery time = the latest relation's;
    ``recovered`` only if every relation is back within the log.
    """
    if disturbance is None:
        ds = _disturbances(log)
        if not ds:
            raise ValueError("the log has no disturbance; pass one explicitly")
        disturbance = ds[0]
    t0 = float(disturbance["t_start"])
    tpe = t0 + float(disturbance.get("duration", 0.0))
    rel = interlimb_phases(log, pairs, load_fraction, min_run_s)["relations"]
    per = {}
    for label, r in rel.items():
        pre = r["stride_end"] <= t0 + 1e-9
        m = circular_stats(r["phases"][pre])["mean_cycles"]
        post = np.flatnonzero(r["stride_start"] >= tpe - 1e-9)
        tback = _NAN
        if np.isfinite(m) and post.size:
            ph = r["phases"]
            inb = np.abs(_wrap_cycles(ph - m)) <= tol_cycles + 1e-12
            inb &= np.isfinite(ph)
            for a in range(post.size - n_strides + 1):
                ks = post[a:a + n_strides]
                if np.all(np.diff(ks) == 1) and np.all(inb[ks]):
                    tback = float(r["stride_start"][ks[0]] - tpe)
                    break
        per[label] = {"pre_mean_cycles": m, "time_s": tback}
    times = np.array([v["time_s"] for v in per.values()], dtype=float)
    ok = bool(times.size and np.all(np.isfinite(times)))
    return {"recovered": ok, "time_s": float(times.max()) if ok else _NAN, "per_relation": per,
            "t_push_end_s": tpe}


# ------------------------------------------------------------------------------------------- undulation
def _dominant_frequency(x, dt: float) -> float:
    """Frequency (Hz) of the largest non-DC FFT peak of a demeaned signal (Hann window, parabolic
    interpolation of the log magnitude around the peak bin)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 8 or not np.isfinite(dt) or np.allclose(x, x.mean()):
        return _NAN
    X = np.abs(np.fft.rfft((x - x.mean()) * np.hanning(n)))
    k = int(np.argmax(X[1:])) + 1
    delta = 0.0
    if 1 <= k < X.size - 1 and np.all(X[k - 1:k + 2] > 0):
        a, b, c = np.log(X[k - 1:k + 2])
        den = a - 2 * b + c
        delta = 0.5 * (a - c) / den if den != 0 else 0.0
    return float((k + delta) / (n * dt))


def body_undulation(log, t_from: float = 1.0, t_to: float | None = None) -> dict:
    """Body undulation during steady walking (protocol §9.2).

    Window: ``t_from`` (default 1.0 s of walking, after the start from the standing pose) to ``t_to``
    (default: the first push if the log has one, else the end). Reported:

    * per body-yaw joint: ``rms_deg`` (RMS of the joint angle about its zero), ``p2p_deg`` (max − min) and
      ``freq_hz`` (dominant frequency by FFT of the demeaned angle);
    * ``com_lateral``: ``amplitude_m`` (half the peak-to-peak of the linearly detrended COM y), ``rms_m``,
      ``freq_hz``;
    * per logged body: yaw about the mean heading — the mean heading is the circular mean of every logged
      body's yaw over the window; ``rms_deg`` and ``p2p_deg`` of the wrapped deviation (defined for
      ``locked`` too); ``worst_yaw_rms_deg`` = max over bodies.
    """
    t = _t(log)
    if t_to is None:
        ds = _disturbances(log)
        t_to = min(float(d["t_start"]) for d in ds) if ds else (t[-1] if t.size else 0.0)
    w = (t >= t_from - 1e-9) & (t <= t_to + 1e-9)
    tw = t[w]
    dt = _dt(tw) if tw.size > 1 else _NAN
    joints = _names(log["joints"])
    q = np.asarray(log["q"], dtype=float)
    yaw_j = {}
    for j in _body_joint_indices(log, ("body_yaw",)):
        a = np.degrees(q[w, j])
        yaw_j[joints[j]] = {"rms_deg": _rms(a), "p2p_deg": (_max(a) - _min(a)) if a.size else _NAN,
                            "freq_hz": _dominant_frequency(a, dt)}
    y = np.asarray(log["com"], dtype=float)[w, 1] if tw.size else np.zeros(0)
    if y.size >= 2:
        yd = y - np.polyval(np.polyfit(tw, y, 1), tw)
        lat = {"amplitude_m": 0.5 * float(yd.max() - yd.min()), "rms_m": _rms(yd), "freq_hz": _dominant_frequency(yd, dt)}
    else:
        lat = {"amplitude_m": _NAN, "rms_m": _NAN, "freq_hz": _NAN}
    bodies = _names(log["bodies"])
    per_body = {}
    heading = _NAN
    if tw.size:
        yaw, _, _ = euler_zyx(np.asarray(log["body_quat"], dtype=float)[w])          # (Tw, B)
        heading = float(np.arctan2(np.mean(np.sin(yaw)), np.mean(np.cos(yaw))))
        dev = np.degrees((yaw - heading + np.pi) % (2 * np.pi) - np.pi)
        for b, name in enumerate(bodies):
            per_body[name] = {"rms_deg": _rms(dev[:, b]), "p2p_deg": float(dev[:, b].max() - dev[:, b].min())}
    else:
        per_body = {name: {"rms_deg": _NAN, "p2p_deg": _NAN} for name in bodies}
    yaw_rms = [v["rms_deg"] for v in per_body.values()]
    return {
        "window_s": (float(t_from), float(t_to)),
        "body_yaw_joints": yaw_j,
        "body_yaw_rms_max_deg": _max([v["rms_deg"] for v in yaw_j.values()]) if yaw_j else _NAN,
        "com_lateral": lat,
        "heading_mean_deg": math.degrees(heading) if np.isfinite(heading) else _NAN,
        "yaw_about_heading": per_body,
        "worst_yaw_rms_deg": _max(yaw_rms) if yaw_rms else _NAN,
    }


# ------------------------------------------------------------------------- lateral deviation and heading
def lateral_deviation_and_heading(log, heading_body: str | None = None, outcome: Mapping | None = None) -> dict:
    """Max |y_COM| (m) and the RMS yaw of the heading body relative to the route (+x), deg (protocol §6).

    ``heading_body`` defaults to the first logged body (Cleopatra: the head, welded to segment 1, so its
    yaw is segment 1's). Yaw is the z-y-x Euler yaw, unwrapped over time.
    """
    bodies = _names(log["bodies"])
    hb = str(heading_body) if heading_body is not None else bodies[0]
    if hb not in bodies:
        raise ValueError(f"heading body {hb!r} is not a logged body {bodies}")
    k = _end_index(log, outcome)
    if k < 0:
        return {"lateral_max_m": _NAN, "heading_rms_deg": _NAN, "heading_max_abs_deg": _NAN, "heading_body": hb}
    y = np.asarray(log["com"], dtype=float)[: k + 1, 1]
    yaw, _, _ = euler_zyx(np.asarray(log["body_quat"], dtype=float)[: k + 1, bodies.index(hb)])
    yaw = np.degrees(np.unwrap(yaw))
    return {"lateral_max_m": _max(np.abs(y)), "heading_rms_deg": _rms(yaw), "heading_max_abs_deg": _max(np.abs(yaw)),
            "heading_body": hb}


# ------------------------------------------------------------------------------------------ trial metrics
_META_KEYS = ("robot", "treatment", "controller", "seed", "v_target", "course_m", "total_mass")


def trial_metrics(log, outcome: Mapping | None = None, payload: str | None = None, feasibility_every: int | None = 10,
                  *, load_fraction: float = 0.02, slip_tol: float = 0.005, min_run_s: float = 0.0,
                  steady_from_s: float = 1.0, heading_body: str | None = None, phase_pairs=None) -> dict:
    """Every per-trial metric of protocol §6 and §9.2 as ONE flat dict of scalars (a table row).

    Keys: the log's identification (``robot``, ``treatment``, ``controller``, ``seed``, ``v_target``,
    ``course_m``, ``total_mass``, ``terrain_kind``, ``terrain_level``, ``terrain_seed``), the outcome's fields
    verbatim (``success``, ``reason``, ``t_end``, ``distance_m``, ...), then the metrics; per-entity values are
    ``<metric>@<entity>``. ``feasibility_every``: evaluate the contact-force LP at every n-th log sample
    (None or 0 skips it). Recovery and phase recovery are added when the log has a disturbance. A block whose
    log keys are missing (a sparse or real-robot log) is skipped and named in ``metrics_skipped``.
    """
    out: dict[str, Any] = {}
    skipped: list[str] = []

    def has(block: str, *keys: str) -> bool:
        ok = all(k in log for k in keys)
        if not ok:
            skipped.append(block)
        return ok

    for k in _META_KEYS:
        if k in log:
            v = _item(log[k])
            out[k] = v.item() if isinstance(v, np.generic) else v
    terr = _item(log.get("terrain"))
    if isinstance(terr, Mapping):
        for k in ("kind", "level", "seed"):
            if k in terr:
                out[f"terrain_{k}"] = terr[k]
    if outcome:
        for k, v in outcome.items():
            v = _item(v)
            if v is None or isinstance(v, (str, bool, int, float, np.generic)):
                out[k] = v.item() if isinstance(v, np.generic) else v
    t = _t(log)
    out["n_samples"] = int(t.size)
    if t.size < 2:
        return out
    contacts = ("foot_force", "foot_normal", "feet", "total_mass", "gravity")

    if has("progress", "com"):
        dist = distance_before_failure(log, outcome)
        out.update({"progress_m": dist["progress_m"], "x_end_m": dist["x_end_m"],
                    "walk_time_s": dist["t_end_s"] - t[0], "completed": dist["completed"],
                    "achieved_speed_m_s": achieved_speed(log, outcome)})

    if has("body_angular_motion", "bodies", "body_quat", "body_angvel"):
        for k, v in body_angular_motion(log, payload).items():
            if k == "per_body":
                for b, d in v.items():
                    for m, x in d.items():
                        out[f"{m}@{b}"] = x
            else:
                out[k] = v

    if has("intersegment_angles", "joints", "joint_kind", "q", "q_range"):
        isa = intersegment_angles(log)
        for j, d in isa["per_joint"].items():
            for m in ("max_abs_deg", "p95_abs_deg", "near_limit_fraction", "limit_hits"):
                out[f"isa_{m}@{j}"] = d[m]
        for kind, d in isa["per_kind"].items():
            for m in ("max_abs_deg", "p95_abs_deg", "near_limit_fraction", "limit_hits"):
                out[f"{kind}_{m}"] = d[m]

    if has("belly_contacts", "bodies", "belly_contact"):
        bc = belly_contacts(log)
        out["belly_contact_fraction"] = bc["fraction"]
        out["belly_contact_events"] = bc["events"]
        for b, x in bc["per_body"].items():
            out[f"belly_contact_fraction@{b}"] = x

    if has("foot_slip", "foot_pos", "com", *contacts):
        sl = foot_slip(log, load_fraction, slip_tol, outcome)
        out.update({"slip_per_m": sl["slip_per_m"], "slip_p95_m": sl["slip_p95_m"],
                    "slip_fraction_over_tol": sl["fraction_over_tol"], "slip_total_m": sl["slip_total_m"],
                    "slip_episodes": sl["n_episodes"]})

    if has("cost_of_transport", "tau", "qd", "joint_active", "tau_stall", "i_stall", "voltage", "com"):
        cot = cost_of_transport(log, outcome)
        out.update({k: cot[k] for k in ("energy_pos_J", "cot_mech", "energy_el_J", "cot_el_est",
                                         "power_mech_mean_W", "power_el_mean_W")})

    if has("actuator_demand", "tau", "qd", "joint_active", "joint_kind", "tau_stall", "tau_rated", "qd_noload"):
        for g, d in actuator_demand(log).items():
            for m, x in d.items():
                if m != "joints":
                    out[f"{g}_{m}"] = x

    if has("support_margin", "com", "foot_pos", *contacts):
        sm = support_margin(log, load_fraction)
        out.update({"support_margin_min_m": sm["min_m"], "support_margin_p5_m": sm["p5_m"],
                    "support_outside_fraction": sm["outside_fraction"],
                    "support_no_contact_fraction": sm["no_contact_fraction"]})

    if feasibility_every and has("contact_force_feasibility", "com", "com_vel", "ang_mom", "foot_pos", "foot_mu",
                                 "nominal_hip_height", *contacts):
        fe = contact_force_feasibility(log, feasibility_every, load_fraction=load_fraction)
        for v, x in fe["infeasible_fraction"].items():
            out[f"infeasible_fraction_{v}"] = x
        out["feasibility_samples"] = max(fe["n_evaluated"].values()) if fe["n_evaluated"] else 0
        out["feasibility_belly_skipped"] = fe["n_belly_skipped"]

    gait_ok = has("gait", *contacts)
    if gait_ok:
        du = duty_factor(log, load_fraction, min_run_s)
        out["duty_mean"], out["duty_sd"] = du["mean"], du["sd"]
        for f, d in du["per_foot"].items():
            out[f"duty_mean@{f}"] = d["mean"]
            out[f"duty_sd@{f}"] = d["sd"]
        if phase_pairs is not None or all(k in log for k in ("foot_pos", "foot_group")):
            sds = []
            for label, d in interlimb_phases(log, phase_pairs, load_fraction, min_run_s)["relations"].items():
                out[f"phase_mean_cycles@{label}"] = d["mean_cycles"]
                out[f"phase_sd_cycles@{label}"] = d["sd_cycles"]
                sds.append(d["sd_cycles"])
            out["phase_sd_max_cycles"] = _max(sds) if sds else _NAN
        else:
            skipped.append("interlimb_phases")

    if has("body_undulation", "joints", "joint_kind", "q", "com", "bodies", "body_quat"):
        un = body_undulation(log, t_from=steady_from_s)
        for j, d in un["body_yaw_joints"].items():
            for m, x in d.items():
                out[f"body_yaw_{m}@{j}"] = x
        out["body_yaw_rms_max_deg"] = un["body_yaw_rms_max_deg"]
        out["com_lateral_amplitude_m"] = un["com_lateral"]["amplitude_m"]
        out["com_lateral_freq_hz"] = un["com_lateral"]["freq_hz"]
        for b, d in un["yaw_about_heading"].items():
            out[f"yaw_about_heading_rms_deg@{b}"] = d["rms_deg"]
        out["worst_yaw_about_heading_rms_deg"] = un["worst_yaw_rms_deg"]

    if has("lateral_deviation_and_heading", "com", "bodies", "body_quat"):
        out.update(lateral_deviation_and_heading(log, heading_body, outcome))

    if _disturbances(log):
        if has("recovery", "com_vel", "body_quat"):
            rc = recovery(log, outcome)
            out.update({"impulse_Ns": rc["impulse_Ns"], "recovered": rc["recovered"],
                        "recovery_time_s": rc["recovery_time_s"], "v_pre_push_m_s": rc["v_pre_m_s"]})
        if gait_ok and (phase_pairs is not None or "foot_group" in log):
            pr = phase_recovery(log, pairs=phase_pairs, load_fraction=load_fraction, min_run_s=min_run_s)
            out["phase_recovered"] = pr["recovered"]
            out["phase_recovery_time_s"] = pr["time_s"]
    out["metrics_skipped"] = ",".join(skipped)
    return out


def trial_result(log, outcome: Mapping | None = None, payload: str | None = None, feasibility_every: int | None = 10,
                 **options):
    """:func:`trial_metrics` wrapped in the common Vegeta ``Result`` shape (kind ``chiron.metrics``).

    ``metrics`` is the flat row; skipped blocks are listed in ``messages``; ``metadata`` records the options,
    the payload body and the protocol reference.
    """
    import time

    from .result import Result

    t0 = time.perf_counter()
    row = trial_metrics(log, outcome, payload, feasibility_every, **options)
    res = Result(kind="chiron.metrics", metrics=row,
                 metadata={"protocol": "docs/myropod_stability.md §6, §9.2", "payload": payload,
                           "feasibility_every": feasibility_every, "options": dict(options),
                           "units": "SI; angles in deg as suffixed"})
    if row.get("metrics_skipped"):
        res.messages.append(f"skipped (log keys missing): {row['metrics_skipped']}")
    res.duration_s = time.perf_counter() - t0
    return res
