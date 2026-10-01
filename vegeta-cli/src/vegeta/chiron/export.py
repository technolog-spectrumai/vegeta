"""CSV export of episode logs and trial tables: raw time series (gzip CSV), per-run summaries, configurations.

Pure numpy/pandas (pandas is imported when a function needs it, so ``import vegeta.chiron.export`` stays light).
Robot-agnostic: everything is read from Chiron's episode log format (``docs/chiron.md``, "The episode log"),
whether ``ChironLab`` wrote it or a real robot did. Units are SI; angles are radians.

**Raw time series** — :func:`raw_frame` turns a log into one row per kept log sample (every ``every``-th sample,
walking time from 0 at the end of the settle) with the curated measurements below; :func:`write_raw_csv` writes it
as gzip CSV (``float_format='%.6g'``; flags as ``True``/``False``). Columns are ``<quantity>@<entity>``, the entity
being a logged body, a foot, a joint or ``com`` (the whole robot):

=================================  ===========  ======================================================================
column                             unit         meaning
=================================  ===========  ======================================================================
``t``                              s            walking time of the sample (0 at the end of the settle)
``x@B``, ``y@B``, ``z@B``          m            body B's centre of mass, world frame (z up, course along +x)
``roll@B``, ``pitch@B``, ``yaw@B`` rad          intrinsic z-y-x Euler angles of body B, ``R = Rz(yaw) Ry(pitch)
                                                Rx(roll)`` (``chiron.metrics.euler_zyx``: + pitch = nose down, + roll =
                                                right side down; yaw in (−π, π], pitch in [−π/2, π/2], roll in (−π, π])
``wx@B``, ``wy@B``, ``wz@B``       rad/s        body B's angular velocity in its own frame (x forward, y left, z up)
``belly_contact@B``                bool         a ``body``-role geom of B (shell, head) touches the ground
``x@com``, ``y@com``, ``z@com``    m            whole-robot centre of mass, world frame
``vx@com``, ``vy@com``, ``vz@com`` m/s          whole-robot centre-of-mass velocity, world frame
``normal_force@F``                 N            ground → foot F normal force ``N = f · n`` (``chiron.metrics.
                                                foot_normal_force``; ``f_z`` where the log has no normal; 0 in the air)
``tangential_force@F``             N            magnitude of the contact force's component in the contact plane,
                                                ``|f − (f · n) n|`` (n = world vertical where the log has no normal)
``loaded@F``                       bool         ``N > load_fraction · m |g|`` (protocol §6: 2 % of the robot's weight;
                                                ``chiron.metrics.foot_loaded``)
``foot_x@F``, ``foot_y@F``,        m            centre of foot F's pad, world frame
``foot_z@F``
``q@J``                            rad          joint J's angle (m for a slide joint)
``qd@J``                           rad/s        joint J's speed (m/s for a slide joint)
``tau@J``                          N·m          servo torque applied at J (N for a slide joint); 0 for a passive joint,
                                                whose spring–damper torque ``−k (q − q0) − c q̇`` is not in the log
``torque_sat@J``                   bool         actuated joints only: ``|τ| ≥ saturation · τ_stall ·
                                                max(0, 1 − |q̇|/ω₀)``
``speed_sat@J``                    bool         actuated joints only: ``|q̇| ≥ saturation · ω₀`` (both: protocol
                                                §6 actuator demand, ``saturation`` = 0.98; ``chiron.servo.saturation``)
``deflection@J``                   rad          body joints only (tag ``body_*``): ``q − q0``, q0 the spring's rest
                                                angle
=================================  ===========  ======================================================================

``q0`` comes from the ``q0`` argument, else from the log's ``joint_springref`` (one value per joint, rad), else it is
taken as 0 rad with a ``UserWarning`` (``df.attrs['q0_source']`` records which). A block whose log keys are missing
(a sparse or real-robot log) is skipped and named in ``df.attrs['skipped']``; ``df.attrs['units']`` maps every
column to its unit (also :func:`column_units`).

**Tables** — :func:`write_runs_csv` writes the per-run table of ``chiron.experiments.run_trials`` (one row per run:
configuration, seeds, terrain parameters, outcome, every metric; non-scalar cells as JSON); :func:`write_configs_csv`
writes a list of configurations (dicts, ``Trial`` objects or dataclasses) flattened to dotted columns
(``robot_kwargs.body_k_pitch``, ``terrain.seed``, ...), seeds included. Both infer gzip from a ``.gz`` suffix.
"""
from __future__ import annotations

import dataclasses
import json
import math
import warnings
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .metrics import euler_zyx, foot_normal_force
from .servo import saturation as _saturation

__all__ = ["QUANTITY_UNITS", "column_units", "flatten_config", "raw_frame", "write_configs_csv", "write_raw_csv",
           "write_runs_csv"]

#: Unit of every raw-frame quantity (the part of a column name before ``@``; ``t`` has no entity).
QUANTITY_UNITS = {
    "t": "s",
    "x": "m", "y": "m", "z": "m",
    "roll": "rad", "pitch": "rad", "yaw": "rad",
    "wx": "rad/s", "wy": "rad/s", "wz": "rad/s",
    "belly_contact": "bool",
    "vx": "m/s", "vy": "m/s", "vz": "m/s",
    "normal_force": "N", "tangential_force": "N", "loaded": "bool",
    "foot_x": "m", "foot_y": "m", "foot_z": "m",
    "q": "rad", "qd": "rad/s", "tau": "N·m",
    "torque_sat": "bool", "speed_sat": "bool",
    "deflection": "rad",
}

COM = "com"
"""Entity name of the whole robot's centre of mass in raw-frame columns."""


# ----------------------------------------------------------------------------------------------- helpers
def _item(x: Any) -> Any:
    """Unwrap 0-d numpy arrays (what ``np.load`` returns for scalars, strings and pickled objects)."""
    if isinstance(x, np.ndarray) and x.shape == ():
        return x.item()
    return x


def _names(seq) -> list[str]:
    return [str(s) for s in _item(seq)]


def _as_log(log) -> Mapping:
    """The log dict of an episode log or of anything carrying one in ``.log`` (an ``Episode``)."""
    inner = getattr(log, "log", None)
    if isinstance(inner, Mapping):
        return inner
    if not isinstance(log, Mapping):
        raise TypeError(f"expected an episode log dict or an Episode, got {type(log).__name__}")
    return log


def _weight(log) -> float:
    """Robot weight ``m |g|`` [N]."""
    return float(_item(log["total_mass"])) * float(np.linalg.norm(np.asarray(log["gravity"], dtype=float)))


def _unit_normals(log) -> np.ndarray:
    """(T, F, 3) unit contact normals (ground → foot); the world vertical where the log has none (in the air)."""
    n = np.asarray(log["foot_normal"], dtype=float)
    ok = np.all(np.isfinite(n), axis=-1, keepdims=True)
    n = np.where(ok, n, np.array([0.0, 0.0, 1.0]))
    return n / np.linalg.norm(n, axis=-1, keepdims=True)


def column_units(columns: Iterable[str]) -> dict[str, str]:
    """``{column: unit}`` for raw-frame columns (``'<quantity>@<entity>'`` or ``'t'``); unknown quantities → ''."""
    out = {}
    for c in columns:
        out[c] = QUANTITY_UNITS.get(str(c).split("@", 1)[0], "")
    return out


def _resolve_q0(log, joints: list[str], body: list[int], q0) -> tuple[np.ndarray, str]:
    """Spring rest angles [rad] per joint and where they came from (see the module docstring)."""
    J = len(joints)
    if q0 is not None:
        if isinstance(q0, Mapping):
            unknown = sorted(set(map(str, q0)) - set(joints))
            if unknown:
                raise ValueError(f"q0 names joints that are not in the log: {unknown}")
            missing = [joints[j] for j in body if joints[j] not in q0]
            if missing:
                raise ValueError(f"q0 must give every body joint's rest angle [rad]; missing {missing}")
            arr = np.zeros(J)
            for name, v in q0.items():
                arr[joints.index(str(name))] = float(v)
            return arr, "argument q0 (per joint)"
        if np.ndim(q0) == 0:
            return np.full(J, float(q0)), "argument q0 (one value for every body joint)"
        arr = np.asarray(q0, dtype=float).reshape(-1)
        if arr.size != J:
            raise ValueError(f"q0 has {arr.size} values for {J} joints")
        return arr, "argument q0 (per joint)"
    if "joint_springref" in log:
        arr = np.asarray(_item(log["joint_springref"]), dtype=float).reshape(-1)
        if arr.size != J:
            raise ValueError(f"log 'joint_springref' has {arr.size} values for {J} joints")
        return arr, "log joint_springref"
    if body:
        warnings.warn("the log has no 'joint_springref' and no q0 was given: body-joint deflection is q − 0 rad "
                      "(pass q0= to state the springs' rest angles)", UserWarning, stacklevel=3)
    return np.zeros(J), "assumed 0 rad (log has no joint_springref; pass q0=)"


# ----------------------------------------------------------------------------------------------- raw frame
def raw_frame(log, every: int = 2, *, q0=None, load_fraction: float = 0.02, saturation: float = 0.98):
    """The curated raw measurements of an episode log as a ``pandas.DataFrame``, one row per kept sample.

    ``log``: a Chiron episode log dict (or an ``Episode``); ``every``: keep every n-th log sample (default 2: 50 Hz
    from the 100 Hz log), starting with the first; ``q0``: the body joints' spring rest angles [rad] — a mapping
    joint → rad, one value for every body joint, or a sequence over the log's joints (default: the log's
    ``joint_springref``, else 0 with a warning); ``load_fraction``: a foot is loaded when its normal force exceeds
    this fraction of the robot's weight (protocol §6: 0.02); ``saturation``: the fraction of the servo limits at
    which a joint counts as torque- or speed-saturated (protocol §6: 0.98).

    Columns and units: see the module docstring (``df.attrs['units']`` too). ``df.attrs`` also holds ``every``,
    ``log_dt_s`` (the kept samples' spacing [s]), ``q0_source``, ``skipped`` (blocks whose log keys are missing) and
    ``meta`` (robot, treatment, controller, seed, v_target [m/s], course_m [m], total_mass [kg], terrain spec).
    """
    import pandas as pd

    log = _as_log(log)
    every = int(every)
    if every < 1:
        raise ValueError("every must be a positive integer")
    t = np.asarray(log["t"], dtype=float).reshape(-1)
    idx = np.arange(0, t.size, every)
    cols: dict[str, np.ndarray] = {"t": t[idx]}
    skipped: list[str] = []

    def has(block: str, *keys: str) -> bool:
        ok = all(k in log for k in keys)
        if not ok:
            skipped.append(block)
        return ok

    def put(name: str, values) -> None:
        if name in cols:
            raise ValueError(f"duplicate column {name!r}: two entities share a name")
        cols[name] = np.asarray(values)[idx]

    # ---- bodies
    bodies = _names(log["bodies"]) if "bodies" in log else []
    if COM in bodies:
        raise ValueError(f"a logged body is named {COM!r}, the whole-robot entity of the export")
    pos = np.asarray(log["body_pos"], dtype=float) if has("body_position", "bodies", "body_pos") else None
    if bodies and has("body_orientation", "bodies", "body_quat"):
        yaw, pitch, roll = euler_zyx(np.asarray(log["body_quat"], dtype=float))
    else:
        yaw = pitch = roll = None
    w = np.asarray(log["body_angvel"], dtype=float) if has("body_angvel", "bodies", "body_angvel") else None
    belly = np.asarray(log["belly_contact"], dtype=bool) if has("belly_contact", "bodies", "belly_contact") else None
    for b, name in enumerate(bodies):
        if pos is not None:
            for k, ax in enumerate("xyz"):
                put(f"{ax}@{name}", pos[:, b, k])
        if roll is not None:
            put(f"roll@{name}", roll[:, b])
            put(f"pitch@{name}", pitch[:, b])
            put(f"yaw@{name}", yaw[:, b])
        if w is not None:
            for k, ax in enumerate(("wx", "wy", "wz")):
                put(f"{ax}@{name}", w[:, b, k])
        if belly is not None:
            put(f"belly_contact@{name}", belly[:, b])

    # ---- whole robot
    if has("com", "com"):
        com = np.asarray(log["com"], dtype=float)
        for k, ax in enumerate("xyz"):
            put(f"{ax}@{COM}", com[:, k])
    if has("com_vel", "com_vel"):
        cv = np.asarray(log["com_vel"], dtype=float)
        for k, ax in enumerate(("vx", "vy", "vz")):
            put(f"{ax}@{COM}", cv[:, k])

    # ---- feet
    feet = _names(log["feet"]) if "feet" in log else []
    if feet and has("foot_forces", "feet", "foot_force", "foot_normal", "total_mass", "gravity"):
        f = np.asarray(log["foot_force"], dtype=float)
        nf = foot_normal_force(log)
        n = _unit_normals(log)
        tang = np.linalg.norm(f - np.sum(f * n, axis=-1, keepdims=True) * n, axis=-1)
        loaded = nf > load_fraction * _weight(log)
    else:
        nf = tang = loaded = None
    fpos = np.asarray(log["foot_pos"], dtype=float) if feet and has("foot_position", "feet", "foot_pos") else None
    for i, name in enumerate(feet):
        if nf is not None:
            put(f"normal_force@{name}", nf[:, i])
            put(f"tangential_force@{name}", tang[:, i])
            put(f"loaded@{name}", loaded[:, i])
        if fpos is not None:
            for k, ax in enumerate("xyz"):
                put(f"foot_{ax}@{name}", fpos[:, i, k])

    # ---- joints
    joints = _names(log["joints"]) if "joints" in log else []
    state = {k: np.asarray(log[k], dtype=float) for k in ("q", "qd", "tau") if k in log}
    if joints and len(state) < 3:
        skipped.append("joint_state")
    if joints and has("saturation", "joint_active", "tau", "qd", "tau_stall", "qd_noload"):
        active = np.asarray(log["joint_active"], dtype=bool).reshape(-1)
        tsat, ssat = _saturation(np.asarray(log["tau"], dtype=float), np.asarray(log["qd"], dtype=float),
                                 np.asarray(log["tau_stall"], dtype=float),
                                 np.asarray(log["qd_noload"], dtype=float), saturation)
    else:
        active = tsat = ssat = None
    kinds = _names(log["joint_kind"]) if "joint_kind" in log else None
    body_j = [j for j, k in enumerate(kinds) if k.startswith("body_")] if kinds is not None else []
    if joints and kinds is None:
        skipped.append("deflection")
    q0_arr, q0_source = (_resolve_q0(log, joints, body_j, q0) if body_j and "q" in state
                         else (np.zeros(len(joints)), "no body joints" if not body_j else "no joint angles"))
    for j, name in enumerate(joints):
        for k in ("q", "qd", "tau"):
            if k in state:
                put(f"{k}@{name}", state[k][:, j])
        if active is not None and active[j]:
            put(f"torque_sat@{name}", tsat[:, j])
            put(f"speed_sat@{name}", ssat[:, j])
        if j in body_j and "q" in state:
            put(f"deflection@{name}", state["q"][:, j] - q0_arr[j])

    df = pd.DataFrame(cols)
    dt = np.diff(cols["t"])
    meta = {}
    for k in ("robot", "treatment", "controller", "seed", "v_target", "course_m", "total_mass", "terrain"):
        if k in log:
            v = _item(log[k])
            meta[k] = v.item() if isinstance(v, np.generic) else v
    df.attrs.update({"units": column_units(df.columns), "every": every,
                     "log_dt_s": float(np.median(dt)) if dt.size else math.nan, "q0_source": q0_source,
                     "skipped": skipped, "meta": meta})
    return df


def write_raw_csv(log, path, every: int = 2, **options) -> Path:
    """Write :func:`raw_frame` ``(log, every, **options)`` as gzip CSV (``float_format='%.6g'``, no index).

    ``path``: the file; ``.gz`` is appended when the name does not end with it. Returns the path written. Read it
    back with ``pandas.read_csv(path)`` (flags come back as bool); units are in the module docstring and
    :func:`column_units`.
    """
    df = raw_frame(log, every, **options)
    path = Path(path)
    if not path.name.endswith(".gz"):
        path = path.with_name(path.name + ".gz")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.6g", compression="gzip")
    return path


# ----------------------------------------------------------------------------------------------- tables
def _cell(v):
    """A CSV-safe cell: scalars as they are, containers as JSON, numpy scalars unwrapped."""
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return json.dumps(v.tolist())
    if isinstance(v, (Mapping, list, tuple, set)):
        try:
            return json.dumps(_plain(v), sort_keys=isinstance(v, Mapping))
        except (TypeError, ValueError):
            return repr(v)
    return v


def _plain(v):
    if isinstance(v, Mapping):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [_plain(x) for x in v]
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    if dataclasses.is_dataclass(v) and not isinstance(v, type):
        return _plain(dataclasses.asdict(v))
    return v


def _compression(path: Path) -> str | None:
    return "gzip" if path.name.endswith(".gz") else None


def write_runs_csv(df, path) -> Path:
    """Write the per-run table (``chiron.experiments.run_trials`` rows: configuration, seeds, terrain parameters,
    outcome and every metric, one row per run) as CSV; gzip when ``path`` ends with ``.gz``.

    Floats are written at full precision (they read back exactly); container cells (dicts, lists, arrays) as JSON
    text. Units are those of the column names (``chiron.metrics``: ``_m``, ``_s``, ``_deg``, ``_rad_s``, ``_J``,
    ``_W``, ``_Nm``, ...). Returns the path.
    """
    import pandas as pd

    path = Path(path)
    out = pd.DataFrame(df).copy()
    for c in out.columns:
        if out[c].dtype == object:
            out[c] = out[c].map(_cell)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, compression=_compression(path))
    return path


def flatten_config(config, prefix: str = "") -> dict:
    """One configuration as a flat dict: nested mappings become dotted keys (``robot_kwargs.body_k_pitch``), lists
    and tuples JSON text, scalars stay. ``config``: a mapping, an object with ``to_dict()`` (a
    ``chiron.experiments.Trial``) or a dataclass instance."""
    if not isinstance(config, Mapping):
        if hasattr(config, "to_dict"):
            config = config.to_dict()
        elif dataclasses.is_dataclass(config) and not isinstance(config, type):
            config = dataclasses.asdict(config)
        else:
            raise TypeError(f"cannot flatten a {type(config).__name__}: pass a dict, a Trial or a dataclass")
    out: dict[str, Any] = {}
    for k, v in config.items():
        key = f"{prefix}{k}"
        if isinstance(v, Mapping):
            out.update(flatten_config(v, prefix=f"{key}."))
        elif dataclasses.is_dataclass(v) and not isinstance(v, type):
            out.update(flatten_config(dataclasses.asdict(v), prefix=f"{key}."))
        else:
            out[key] = _cell(v)
    return out


def write_configs_csv(configs: Sequence, path) -> Path:
    """Write configurations — the treatment parameters, trial specs with their seeds — one row each, flattened by
    :func:`flatten_config` (columns in order of first appearance; a key a configuration lacks is empty). Gzip when
    ``path`` ends with ``.gz``. Returns the path."""
    import pandas as pd

    rows = [flatten_config(c) for c in configs]
    columns = list(dict.fromkeys(k for r in rows for k in r))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False, compression=_compression(path))
    return path
