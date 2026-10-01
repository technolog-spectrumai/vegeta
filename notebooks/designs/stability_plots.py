"""Comparative plots of the Cleopatra stability study — protocol ``docs/myropod_stability.md`` §11.6 and §12.6.

Matplotlib only. Every function builds its own ``matplotlib.figure.Figure`` (no pyplot state: save it with
``fig.savefig(path)``, show it in a notebook with ``display(fig)``) and returns ``(fig, table)``: the figure and the
tidy ``pandas.DataFrame`` it plotted, one row per plotted point (with ``n``, the runs behind it, also written next
to every point on the figure).

**Inputs.** The per-run table of ``vegeta.chiron.experiments.run_trials`` (one row per run): the trial spec flattened
(``robot_kwargs.body_connection``, ``robot_kwargs.body_k_yaw`` [N·m/rad], ``controller_factory``,
``controller_kwargs.sigma`` [rad/(N·s)], ``terrain.kind``, ``terrain.level`` (the study's normalised difficulty:
h/L, or degrees for ``cross_slope``), ``terrain.height`` / ``terrain.rms`` [m], ``terrain.angle_deg`` [deg],
``terrain.spacing`` [m], ``seed``, ``v_target`` [m/s], ``lab_kwargs.*``, ...), the run's ``status`` (``'ok'``, or
``'error'`` for a simulation that raised), the outcome (``success``, ``reason``, ``distance_m`` [m]), the log's
``treatment`` and ``controller`` names, and the metrics of ``vegeta.chiron.metrics.trial_metrics``
(``achieved_speed_m_s`` [m/s], ``payload_roll_rate_rms_rad_s`` [rad/s], ``roll_rate_rms_rad_s@<body>``,
``payload_tilt_p95_deg`` [deg], ``slip_per_m`` [m of foot slip per m of progress], ``cot_mech`` [–],
``body_yaw_rms_max_deg`` [deg], ...). Episode logs (dicts or ``Episode`` s) for :func:`synchronized_traces`.

**Rules applied everywhere.**

* *Runs, not frames, are the samples*; every interval and test is ``vegeta.chiron.stats``: success rates with Wilson
  95 % intervals (``rate_ci``); means over runs with a percentile bootstrap of the runs (``paired_difference_ci``
  of the values against 0 — 10 000 resamples, seed 0; no interval from fewer than two runs); paired differences
  with the paired bootstrap (``paired_difference_ci``; success: ``success_difference_ci(method='bootstrap')`` and
  its exact McNemar p); the interaction with ``difference_of_differences_ci``; the undulation onset with
  ``onset_speed``.
* *Failures are always shown.* Metrics of failed runs cover the run up to its failure; functions with ``which=``
  can restrict to successful runs, and then say so in the title and in the table's ``which`` column.
* *Treatments* are Amendment D's ``spring`` (c = 0) and ``spring_damper`` (c > 0), read from the ``treatment``
  column (else ``robot_kwargs.body_connection``). Legacy runs (pre-Amendment D: ``legacy:*``, ``rigid``,
  ``flexible``, ``flexible+yaw``, ``locked``, ``flexible+roll``) and rows with ``status != 'ok'`` are left out and
  counted in a footnote — never pooled.
* *Controllers* are named by the log (``fixed`` = baseline fixed-phase, σ = 0; ``adaptive`` = load feedback) with
  σ from ``controller_kwargs.sigma``; different σ are different controllers.
* *No pooling of different configurations.* Rows that differ in a setting that is not a plotted factor (robot
  keywords other than the treatment's c, lab and rule settings, controller settings other than σ) raise
  ``ValueError`` — filter first. Speeds and terrains are faceted, not pooled.
* Paired comparisons match runs on the seed and every configuration column (terrain spec with its seed, speed,
  robot keywords except the treatment and its c, lab and rule settings, and the controller's within a controller);
  a duplicate raises ``ValueError``, an unmatched run is counted.
* The verdict per terrain, difficulty and controller is the protocol's §11.6 rule (:func:`classify`): *helps* /
  *hurts* when the paired-difference interval excludes 0 — and, for success, the exact McNemar p < 0.05 — in the
  better / worse direction; otherwise *no clear difference*. Damping is not assumed to win.

Colours: treatment by hue (spring-only blue, spring–damper orange), controller by marker and line style (fixed:
circle, solid; adaptive: square, dashed); successful runs filled, failed runs hollow.
"""
from __future__ import annotations

import math
import textwrap
from typing import Sequence

import numpy as np
import pandas as pd

from vegeta.chiron import export as cx
from vegeta.chiron import stats

__all__ = ["BETTER", "TREATMENTS", "achieved_vs_commanded_speed", "angular_motion_and_slip_vs_roughness",
           "classify", "cost_vs_speed", "factorial_interaction", "paired_difference_forest", "prepare_runs",
           "success_vs_roughness", "synchronized_traces", "undulation_vs_speed"]

TREATMENTS = ("spring", "spring_damper")
TREATMENT_LABEL = {"spring": "spring-only (c = 0)", "spring_damper": "spring–damper (c > 0)"}
LEGACY_TREATMENTS = ("rigid", "flexible", "flexible+yaw", "locked", "flexible+roll")
CONTROLLER_ORDER = ("fixed", "adaptive")
N_BOOT = 10_000            # bootstrap resamples (protocol §7)
BOOT_SEED = 0              # fixed bootstrap seed (protocol §7)
ALPHA = 0.05               # McNemar level of the §11.6 rule

#: Direction in which a metric is better (``paired_difference_forest`` needs it to say helps / hurts).
BETTER = {
    "success": "higher", "achieved_speed_m_s": "higher", "progress_m": "higher", "distance_m": "higher",
    "cot_mech": "lower", "cot_el_est": "lower", "energy_pos_J": "lower", "slip_per_m": "lower",
    "slip_p95_m": "lower", "slip_fraction_over_tol": "lower", "belly_contact_fraction": "lower",
    "payload_roll_rate_rms_rad_s": "lower", "payload_pitch_rate_rms_rad_s": "lower",
    "worst_roll_rate_rms_rad_s": "lower", "worst_pitch_rate_rms_rad_s": "lower",
    "payload_tilt_p95_deg": "lower", "worst_tilt_p95_deg": "lower", "payload_tilt_max_deg": "lower",
    "worst_tilt_max_deg": "lower", "lateral_max_m": "lower", "heading_rms_deg": "lower",
    "support_outside_fraction": "lower", "support_margin_min_m": "higher", "support_margin_p5_m": "higher",
    "infeasible_fraction_quasi_static": "lower", "infeasible_fraction_dynamic": "lower",
}

#: Units of the metrics plotted here (axis labels).
UNITS = {"achieved_speed_m_s": "m/s", "v_target": "m/s", "cot_mech": "–", "slip_per_m": "m/m",
         "body_yaw_rms_max_deg": "deg", "distance_m": "m", "progress_m": "m"}

_ALIASES = {"longitudinal_bumps": "long_bumps", "alternating_bumps": "alt_bumps"}
_DIFFICULTY = {"rough": "RMS height / leg length  h/L", "long_bumps": "bump height / leg length  h/L",
               "alt_bumps": "bump height / leg length  h/L", "steps": "step height / leg length  h/L",
               "cross_slope": "cross-slope angle [deg]", "flat": "level"}
_SI_PARAM = {"rough": ("terrain.rms", "m"), "long_bumps": ("terrain.height", "m"),
             "alt_bumps": ("terrain.height", "m"), "steps": ("terrain.height", "m"),
             "cross_slope": ("terrain.angle_deg", "deg")}

# palette (validated categorical slots; text never wears a series colour)
COLOR = {"spring": "#2a78d6", "spring_damper": "#eb6834"}
VERDICT_COLOR = {"helps": "#008300", "hurts": "#e34948", "no clear difference": "#8a8984"}
STIFFNESS_RAMP = ("#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b")
ENTITY_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
_MARKERS = {"fixed": "o", "adaptive": "s"}
_LINES = {"fixed": "-", "adaptive": "--"}
_OTHER_MARKERS = ("^", "D", "v", "P")


# ============================================================================================ preparation
def _first(d: pd.DataFrame, names: Sequence[str]) -> str | None:
    for n in names:
        if n in d.columns:
            return n
    return None


def _truthy(v) -> float:
    if v is None:
        return math.nan
    if isinstance(v, str):
        s = v.strip().lower()
        return 1.0 if s in ("true", "1", "yes") else 0.0 if s in ("false", "0", "no") else math.nan
    try:
        f = float(v)
    except (TypeError, ValueError):
        return math.nan
    return math.nan if math.isnan(f) else float(f > 0.5)


def _num(d: pd.DataFrame, col: str | None) -> pd.Series:
    if col is None or col not in d.columns:
        return pd.Series(np.nan, index=d.index, dtype=float)
    return pd.to_numeric(d[col], errors="coerce").astype(float)


def _kind(v) -> str:
    s = "" if v is None or (isinstance(v, float) and math.isnan(v)) else str(v)
    return _ALIASES.get(s, s)


def _ctrl_label(name: str, sigma: float) -> str:
    return f"{name} (σ = {sigma:g})" if np.isfinite(sigma) else name


def prepare_runs(df) -> tuple[pd.DataFrame, list[str]]:
    """The run table with the columns every plot uses, and the footnotes on what was left out.

    Added columns: ``_treatment`` ('spring' | 'spring_damper'), ``_controller`` (name), ``_sigma`` [rad/(N·s)]
    (``controller_kwargs.sigma``; 0 for ``fixed``), ``_ctrl`` (label), ``_terrain`` (kind, aliases resolved),
    ``_level`` (``terrain.level``), ``_v`` (``v_target`` [m/s]), ``_success`` (1/0, nan if missing).
    Rows with ``status != 'ok'`` and legacy or unknown treatments are dropped and counted in the notes.
    """
    d = pd.DataFrame(df).copy().reset_index(drop=True)
    notes: list[str] = []
    if "status" in d.columns:
        bad = d["status"].notna() & (d["status"].map(str) != "ok")
        if bad.any():
            notes.append(f"{int(bad.sum())} run(s) with a simulation error (status ≠ 'ok') left out")
            d = d.loc[~bad].reset_index(drop=True)
    # the log's names win: run_trials stores them as 'metric.<name>' when a factor column of that name differs
    tcols = [c for c in ("metric.treatment", "treatment", "robot_kwargs.body_connection") if c in d.columns]
    if not tcols:
        raise ValueError("the run table has no 'treatment' or 'robot_kwargs.body_connection' column")
    treat = pd.Series([""] * len(d), index=d.index, dtype=object)
    for c in reversed(tcols):                      # the first of tcols that is set wins
        v = d[c].map(lambda x: "" if x is None or (isinstance(x, float) and math.isnan(x)) else str(x))
        treat = v.where(v != "", treat)
    legacy = treat.map(lambda s: s.startswith("legacy") or s in LEGACY_TREATMENTS)
    other = ~legacy & ~treat.isin(TREATMENTS)
    if legacy.any():
        notes.append(f"{int(legacy.sum())} legacy (pre-Amendment D) run(s) left out — never pooled")
    if other.any():
        notes.append(f"{int(other.sum())} run(s) of other treatments left out: {sorted(set(treat[other]))}")
    keep = ~legacy & ~other
    d, treat = d.loc[keep].reset_index(drop=True), treat[keep].reset_index(drop=True)
    d["_treatment"] = treat
    ccol = _first(d, ("metric.controller", "controller"))
    if ccol is not None:
        name = d[ccol].astype(object).map(lambda x: "" if x is None or (isinstance(x, float) and math.isnan(x))
                                          else str(x))
    elif "controller_factory" in d.columns:
        name = d["controller_factory"].astype(object).map(lambda x: str(x).rsplit(":", 1)[-1])
    else:
        raise ValueError("the run table has no 'controller' or 'controller_factory' column")
    sigma = _num(d, "controller_kwargs.sigma")
    sigma = sigma.where(~(sigma.isna() & (name == "fixed")), 0.0)
    d["_controller"], d["_sigma"] = name, sigma
    d["_ctrl"] = [_ctrl_label(n, s) for n, s in zip(name, sigma)]
    kcol = _first(d, ("terrain.kind", "terrain_kind"))
    d["_terrain"] = d[kcol].map(_kind) if kcol else ""
    d["_level"] = _num(d, _first(d, ("terrain.level", "terrain_level", "level")))
    d.loc[d["_level"].isna() & (d["_terrain"] == "flat"), "_level"] = 0.0      # flat ground is difficulty 0
    d["_v"] = _num(d, "v_target")
    d["_success"] = d["success"].map(_truthy).astype(float) if "success" in d.columns else math.nan
    return d, notes


def _filter_terrain(d: pd.DataFrame, terrain) -> pd.DataFrame:
    if terrain is None:
        return d
    kinds = [_kind(terrain)] if isinstance(terrain, str) else [_kind(t) for t in terrain]
    out = d.loc[d["_terrain"].isin(kinds)].reset_index(drop=True)
    if out.empty:
        raise ValueError(f"no runs on terrain {terrain!r} (have {sorted(set(d['_terrain']))})")
    return out


def _filter_value(d: pd.DataFrame, col: str, value, what: str) -> pd.DataFrame:
    if value is None:
        return d
    out = d.loc[np.isclose(d[col].to_numpy(float), float(value), rtol=1e-9, atol=1e-12)].reset_index(drop=True)
    if out.empty:
        raise ValueError(f"no runs with {what} = {value!r} (have {sorted(set(d[col].dropna()))})")
    return out


def _config_columns(d: pd.DataFrame, extra_ok: Sequence[str] = ()) -> list[str]:
    """Settings that must be equal within one plotted series (not factors of these plots): robot keywords other
    than the treatment and its c, controller keywords other than σ, terrain parameters other than the seed and the
    level labels (so callers group by level), lab, rule and metric settings, course, settle, duration, pushes."""
    out = []
    for c in d.columns:
        if c in extra_ok:
            continue
        if c.startswith("robot_kwargs."):
            if c == "robot_kwargs.body_connection" or c.startswith("robot_kwargs.body_c_"):
                continue
        elif c.startswith("controller_kwargs."):
            if c == "controller_kwargs.sigma":
                continue
        elif c.startswith(("terrain.", "terrain_kwargs.")):
            if c.split(".", 1)[1] in ("seed", "kind", "level", "label", "difficulty"):
                continue
        elif not (c.startswith(("lab_kwargs.", "rules.", "metrics_kwargs.")) or
                  c in ("robot_factory", "terrain_factory", "course_m", "settle", "duration", "disturbances",
                        "version")):
            continue
        out.append(c)
    return out


def _check_single(d: pd.DataFrame, by: Sequence[str] = (), extra_ok: Sequence[str] = ()) -> None:
    """Raise if rows of one group (``by``) differ in a configuration setting (no pooling)."""
    cols = _config_columns(d, extra_ok)
    if not cols or d.empty:
        return
    groups = d.groupby(list(by), dropna=False, sort=False) if by else [((), d)]
    for _, g in groups:
        mixed = {c: sorted(set(g[c].map(str))) for c in cols if g[c].map(str).nunique() > 1}
        if mixed:
            raise ValueError(f"the runs mix configurations {mixed}: filter the table to one configuration first "
                             "(results of different settings are never pooled)")


def _pair_columns(d: pd.DataFrame, free: Sequence[str] = ()) -> list[str]:
    """Columns a paired comparison matches on: the seed and every configuration column except those in ``free``."""
    if "seed" not in d.columns:
        raise ValueError("paired comparisons need a 'seed' column")
    cols = ["seed"]
    for c in d.columns:
        if c == "seed" or c in free:
            continue
        if c.startswith(("robot_kwargs.", "controller_kwargs.", "terrain.", "terrain_kwargs.", "lab_kwargs.", "rules.",
                         "metrics_kwargs.")) or c in ("robot_factory", "controller_factory", "terrain_factory",
                                                       "v_target", "course_m", "duration", "settle", "disturbances",
                                                       "version"):
            if c == "robot_kwargs.body_connection" or c.startswith("robot_kwargs.body_c_"):
                continue
            cols.append(c)
    return cols


def _pairs(d: pd.DataFrame, side: str, sides: Sequence, values: Sequence[str], free: Sequence[str] = ()):
    """Wide table of paired runs: one row per matched key, columns ``(value, side)``; plus the unmatched count."""
    keys = _pair_columns(d, free)
    key = d[keys].map(str).agg("|".join, axis=1)          # str(nan) == 'nan': missing settings match
    sub = d.loc[d[side].isin(sides)]
    k = key[sub.index]
    dup = pd.DataFrame({"k": k, "s": sub[side]}).duplicated()
    if dup.any():
        raise ValueError(f"runs are not unique per {side} on the pairing key {keys} (e.g. key {k[dup].iloc[0]!r}); "
                         "filter the table to one configuration first")
    present = pd.crosstab(k, sub[side]).reindex(columns=list(sides), fill_value=0)
    full = present.index[(present > 0).all(axis=1)]
    unmatched = int(present.loc[~present.index.isin(full)].to_numpy().sum())
    wide = {}
    for v in values:
        w = pd.DataFrame({"k": k, "s": sub[side], "v": sub[v]}).pivot(index="k", columns="s", values="v")
        w = w.reindex(index=full, columns=list(sides))
        for s in sides:
            wide[(v, s)] = w[s].to_numpy()
    return wide, len(full), unmatched


# ============================================================================================ statistics
def _rate(x) -> dict:
    return stats.rate_ci(np.asarray(x, dtype=float))


def _mean_ci(x) -> dict:
    """Mean over runs with the percentile bootstrap of the runs (chiron.stats; no interval below two runs)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0, "mean": math.nan, "lo": math.nan, "hi": math.nan}
    r = stats.paired_difference_ci(x, np.zeros(x.size), n_boot=N_BOOT, seed=BOOT_SEED)
    lo, hi = (r["lo"], r["hi"]) if x.size >= 2 else (math.nan, math.nan)
    return {"n": int(x.size), "mean": float(r["estimate"]), "lo": lo, "hi": hi}


def classify(lo: float, hi: float, better: str, p_value: float | None = None, alpha: float = ALPHA) -> str:
    """The protocol's §11.6 verdict for a paired difference ``spring_damper − spring`` with interval ``[lo, hi]``.

    'helps' / 'hurts' when the interval excludes 0 on the ``better`` side ('higher' or 'lower') / the other side
    and — when a McNemar ``p_value`` is given (success) — ``p_value < alpha``; otherwise 'no clear difference'.
    """
    if better not in ("higher", "lower"):
        raise ValueError("better must be 'higher' or 'lower'")
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "no clear difference"
    if p_value is not None and not (np.isfinite(p_value) and p_value < alpha):
        return "no clear difference"
    if lo > 0:
        return "helps" if better == "higher" else "hurts"
    if hi < 0:
        return "hurts" if better == "higher" else "helps"
    return "no clear difference"


# ============================================================================================ drawing helpers
def _figure(width: float, height: float):
    from matplotlib.figure import Figure

    return Figure(figsize=(width, height), layout="constrained")


def _style(ax) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)


def _finish(fig, title: str, notes: Sequence[str] = ()) -> None:
    fig.suptitle(title, fontsize=11, color=INK)
    width = max(40, int(fig.get_figwidth() * 15))
    lines = [ln for n in notes if n for ln in textwrap.wrap(n, width)]
    if lines:
        frac = min(0.35, (0.15 * len(lines) + 0.08) / fig.get_figheight())
        fig.get_layout_engine().set(rect=(0.0, frac, 1.0, 1.0 - frac))     # (left, bottom, width, height)
        fig.text(0.01, 0.01, "\n".join(lines), fontsize=7.5, color=INK2, ha="left", va="bottom")


def _marker(ctrl_name: str, i: int = 0) -> str:
    return _MARKERS.get(ctrl_name, _OTHER_MARKERS[i % len(_OTHER_MARKERS)])


def _line(ctrl_name: str) -> str:
    return _LINES.get(ctrl_name, ":")


def _series(d: pd.DataFrame) -> list[tuple[str, str, str, float]]:
    """(treatment, controller label, controller name, σ) in plotting order."""
    ctrls = d[["_ctrl", "_controller", "_sigma"]].drop_duplicates()
    order = {n: i for i, n in enumerate(CONTROLLER_ORDER)}
    ctrls = sorted(ctrls.itertuples(index=False), key=lambda r: (order.get(r[1], 99), r[1], np.nan_to_num(r[2])))
    return [(t, c[0], c[1], c[2]) for t in TREATMENTS if (d["_treatment"] == t).any() for c in ctrls
            if ((d["_treatment"] == t) & (d["_ctrl"] == c[0])).any()]


def _step(xs) -> float:
    """Smallest spacing of the x levels (a quarter of the value for one level, 0.1 for a single 0)."""
    xs = np.unique(np.asarray(xs, dtype=float)[np.isfinite(np.asarray(xs, dtype=float))])
    if xs.size > 1:
        return float(np.min(np.diff(xs)))
    return 0.25 * abs(float(xs[0])) if xs.size and xs[0] != 0 else 0.1


def _dodge(i: int, m: int, xs) -> float:
    """x offset of series i of m at a level, so the series' points and intervals do not overlap."""
    return (i - (m - 1) / 2) * 0.1 * _step(xs)


def _annotate_n(ax, x, y, n) -> None:
    if np.isfinite(x) and np.isfinite(y):
        ax.annotate(f"n={n}", (x, y), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                    fontsize=6.5, color=INK2, rotation=90)


def _err(ax, x, mean, lo, hi, **kw):
    yerr = None
    if np.isfinite(lo) and np.isfinite(hi):
        yerr = [[max(mean - lo, 0.0)], [max(hi - mean, 0.0)]]
    kw.setdefault("markersize", 6)
    ax.errorbar([x], [mean], yerr=yerr, capsize=2.5, elinewidth=1.0, **kw)


def _legend_handles(series, filled_hollow: bool = False):
    from matplotlib.lines import Line2D

    hs = [Line2D([], [], color=COLOR[t], marker="o", ls="-", label=TREATMENT_LABEL[t])
          for t in TREATMENTS if any(s[0] == t for s in series)]
    seen = []
    for _, label, name, _ in series:
        if label not in seen:
            seen.append(label)
            hs.append(Line2D([], [], color=INK2, marker=_marker(name, len(seen)), ls=_line(name), label=label))
    if filled_hollow:
        hs.append(Line2D([], [], color=INK2, marker="o", ls="", label="successful run (filled)"))
        hs.append(Line2D([], [], color=INK2, marker="o", ls="", markerfacecolor="none", label="failed run (hollow)"))
    return hs


def _unit_of(metric: str) -> str:
    """Unit of a metric column from ``chiron.metrics``' suffix convention."""
    if metric in UNITS:
        return UNITS[metric]
    for suffix, unit in (("_rad_s", "rad/s"), ("_m_s", "m/s"), ("_deg", "deg"), ("_Nm", "N·m"), ("_hz", "Hz"),
                         ("_J", "J"), ("_W", "W"), ("_s", "s"), ("_m", "m"), ("_fraction", "–"), ("_cycles", "cycles")):
        if metric.endswith(suffix):
            return unit
    return "–"


def _difficulty_label(kind: str) -> str:
    return _DIFFICULTY.get(kind, "difficulty level (terrain.level)")


def _si_column(d: pd.DataFrame, kind: str) -> tuple[str | None, str]:
    col, unit = _SI_PARAM.get(kind, (None, ""))
    return (col if col in d.columns else None), unit


def _which(d: pd.DataFrame, which: str) -> tuple[pd.DataFrame, str]:
    if which == "all":
        return d, "all runs, failed runs included (their metrics cover the run up to the failure)"
    if which == "success":
        return d.loc[d["_success"] == 1].reset_index(drop=True), "successful runs only (failed runs excluded)"
    raise ValueError("which must be 'all' or 'success'")


# ============================================================================================ 1. success
def success_vs_roughness(df, terrain: str):
    """Success rate vs difficulty on one terrain, per treatment × controller, with Wilson 95 % intervals.

    x: ``terrain.level`` (h/L, or degrees for ``cross_slope``); y: success rate [0–1] with its Wilson interval
    (``chiron.stats.rate_ci``); one panel per commanded speed ``v_target`` [m/s] (never pooled). Table: terrain,
    v_target [m/s], level, the physical parameter (``difficulty_si`` [m or deg]) and ``spacing_m`` when recorded,
    treatment, controller, sigma, n, k (successes), rate, lo, hi.
    """
    d, notes = prepare_runs(df)
    d = _filter_terrain(d, terrain)
    d = d.loc[d["_success"].notna()].reset_index(drop=True)
    _check_single(d, by=["_treatment", "_ctrl", "_level", "_v"])
    kind = _kind(terrain)
    si_col, si_unit = _si_column(d, kind)
    speeds = sorted(d["_v"].dropna().unique()) or [math.nan]
    series = _series(d)
    fig = _figure(max(5.5, 4.2 * len(speeds)) + 1.8, 4.2)
    axes = fig.subplots(1, len(speeds), sharey=True, squeeze=False)[0]
    rows = []
    for ax, v in zip(axes, speeds):
        dv = d if not np.isfinite(v) else d.loc[np.isclose(d["_v"], v)]
        levels = np.sort(dv["_level"].dropna().unique())
        for i, (t, label, name, sigma) in enumerate(series):
            g = dv.loc[(dv["_treatment"] == t) & (dv["_ctrl"] == label)]
            xs, ys = [], []
            for L in levels:
                gl = g.loc[np.isclose(g["_level"], L)]
                if gl.empty:
                    continue
                r = _rate(gl["_success"])
                x = L + _dodge(i, len(series), levels)
                _err(ax, x, r["rate"], r["lo"], r["hi"], color=COLOR[t], marker=_marker(name, i), ls="none")
                _annotate_n(ax, x, r["hi"], r["n"])
                xs.append(x)
                ys.append(r["rate"])
                rows.append({"terrain": kind, "v_target": v, "level": float(L),
                             "difficulty_si": float(_num(gl, si_col).mean()) if si_col else math.nan,
                             "difficulty_si_unit": si_unit,
                             "spacing_m": float(_num(gl, "terrain.spacing").mean()),
                             "treatment": t, "controller": name, "sigma": sigma, **r})
            ax.plot(xs, ys, color=COLOR[t], ls=_line(name), lw=1.5)
        _style(ax)
        ax.set_ylim(-0.03, 1.2)                      # headroom for the n labels
        ax.set_xlabel(_difficulty_label(kind))
        ax.set_title(f"v_target = {v:g} m/s" if np.isfinite(v) else "", fontsize=9)
    axes[0].set_ylabel("success rate (Wilson 95 % CI)")
    axes[-1].legend(handles=_legend_handles(series), fontsize=7.5, loc="center left", bbox_to_anchor=(1.02, 0.5),
                    frameon=False)
    _finish(fig, f"Traversal success vs difficulty — {kind}",
            ["All runs; n = runs per point (trials, not frames, are the samples)."] + notes)
    return fig, pd.DataFrame(rows)


# ============================================================================================ 2. speed
def _facets(d: pd.DataFrame) -> list[tuple[str, float]]:
    f = d[["_terrain", "_level"]].drop_duplicates()
    return sorted(((str(a), float(b)) for a, b in f.itertuples(index=False)),
                  key=lambda r: (r[0], np.nan_to_num(r[1], nan=-1.0)))


def _facet_axes(fig, n: int, ncols: int = 4, sharex=True, sharey=True):
    ncols = min(ncols, n)
    nrows = int(math.ceil(n / ncols))
    axes = fig.subplots(nrows, ncols, sharex=sharex, sharey=sharey, squeeze=False).ravel()
    for ax in axes[n:]:
        ax.set_visible(False)
    return axes[:n]


def _level_title(kind: str, level: float) -> str:
    if not np.isfinite(level):
        return kind
    return f"{kind}, {'angle' if kind == 'cross_slope' else 'h/L'} = {level:g}"


def achieved_vs_commanded_speed(df, *, terrain=None, level=None):
    """Achieved vs commanded speed [m/s]: every run (failed runs hollow — their speed is the progress to the
    failure over the time to it), per-cell means with bootstrap 95 % intervals over runs, and the identity line.

    Cells are treatment × controller × ``v_target``; one panel per terrain and level (filter with ``terrain`` /
    ``level``). Table: one row per cell — terrain, level, treatment, controller, sigma, v_target, n (runs), n_success,
    mean, lo, hi (achieved speed, m/s).
    """
    d, notes = prepare_runs(df)
    d = _filter_terrain(d, terrain)
    d = _filter_value(d, "_level", level, "level")
    d["_va"] = _num(d, "achieved_speed_m_s")
    d = d.loc[d["_va"].notna()].reset_index(drop=True)
    if d.empty:
        raise ValueError("no runs with an achieved_speed_m_s value")
    _check_single(d, by=["_treatment", "_ctrl", "_terrain", "_level"])
    facets = _facets(d)
    series = _series(d)
    ncols = min(4, len(facets))
    fig = _figure(3.6 * ncols + 2.0, 3.4 * int(math.ceil(len(facets) / ncols)) + 0.6)
    axes = _facet_axes(fig, len(facets))
    rng = np.random.default_rng(0)
    rows = []
    vmax = float(np.nanmax(d["_v"]))
    for ax, (kind, L) in zip(axes, facets):
        dv = d.loc[(d["_terrain"] == kind) & (np.isclose(d["_level"], L) | (np.isnan(L) & d["_level"].isna()))]
        speeds = np.sort(dv["_v"].dropna().unique())
        ax.plot([0, vmax * 1.08], [0, vmax * 1.08], color=INK2, lw=0.8, ls=":", label="achieved = commanded")
        for i, (t, label, name, sigma) in enumerate(series):
            g = dv.loc[(dv["_treatment"] == t) & (dv["_ctrl"] == label)]
            for v in speeds:
                gv = g.loc[np.isclose(g["_v"], v)]
                if gv.empty:
                    continue
                x0 = v + _dodge(i, len(series), speeds)
                xs = x0 + rng.uniform(-0.03, 0.03, len(gv)) * _step(speeds)
                ok = (gv["_success"] == 1).to_numpy()
                y = gv["_va"].to_numpy(float)
                ax.scatter(xs[ok], y[ok], s=12, color=COLOR[t], marker=_marker(name, i), alpha=0.55, linewidths=0)
                ax.scatter(xs[~ok], y[~ok], s=12, facecolors="none", edgecolors=COLOR[t], marker=_marker(name, i),
                           alpha=0.8, linewidths=0.8)
                r = _mean_ci(y)
                _err(ax, x0, r["mean"], r["lo"], r["hi"], color=COLOR[t], marker=_marker(name, i), ls="none",
                     markeredgecolor=INK, zorder=4)
                _annotate_n(ax, x0, max(r["hi"], np.nanmax(y)) if np.isfinite(r["hi"]) else r["mean"], r["n"])
                rows.append({"terrain": kind, "level": L, "treatment": t, "controller": name, "sigma": sigma,
                             "v_target": float(v), "n_success": int(ok.sum()), **r})
        _style(ax)
        ax.margins(y=0.12)
        ax.set_title(_level_title(kind, L), fontsize=9)
        ax.set_xlabel("commanded speed v_target [m/s]")
        ax.set_ylabel("achieved speed [m/s]")
    axes[-1].legend(handles=_legend_handles(series, True), fontsize=7.5, loc="center left",
                    bbox_to_anchor=(1.02, 0.5), frameon=False)
    _finish(fig, "Achieved vs commanded speed",
            ["Every run shown, failed runs hollow (speed = progress to the failure / time to it); large markers: "
             "mean over the runs of the cell with a bootstrap 95 % CI; n = runs per cell; dotted: identity."] + notes)
    return fig, pd.DataFrame(rows)


# ============================================================================================ 3. angular motion
def _worst_segment(d: pd.DataFrame, metric: str) -> tuple[pd.Series, str]:
    """Row-wise maximum of ``<metric>@segment *`` (else of every non-payload body; else ``worst_<metric>``)."""
    per = [c for c in d.columns if c.startswith(f"{metric}@")]
    payload = set(d["payload_body"].dropna().map(str)) if "payload_body" in d.columns else set()
    segs = [c for c in per if c.split("@", 1)[1].startswith("segment")]
    cols = segs or [c for c in per if c.split("@", 1)[1] not in payload]
    if cols:
        return d[cols].apply(pd.to_numeric, errors="coerce").max(axis=1, skipna=True), \
            f"max over {', '.join(c.split('@', 1)[1] for c in cols)}"
    if f"worst_{metric}" in d.columns:
        return _num(d, f"worst_{metric}"), "worst logged body"
    return pd.Series(np.nan, index=d.index), "not in the table"


ANGULAR_PANELS = (
    ("payload_roll_rate_rms_rad_s", "payload RMS roll rate", "rad/s"),
    ("worst_segment:roll_rate_rms_rad_s", "worst-segment RMS roll rate", "rad/s"),
    ("payload_pitch_rate_rms_rad_s", "payload RMS pitch rate", "rad/s"),
    ("worst_segment:pitch_rate_rms_rad_s", "worst-segment RMS pitch rate", "rad/s"),
    ("payload_tilt_p95_deg", "payload p95 tilt", "deg"),
    ("worst_segment:tilt_p95_deg", "worst-segment p95 tilt", "deg"),
    ("slip_per_m", "foot slip per metre of progress", "m/m"),
)


def angular_motion_and_slip_vs_roughness(df, terrain: str, which: str = "all", *, v_target=None):
    """Payload and worst-segment body angular motion, and foot slip, vs difficulty on one terrain.

    Panels (``ANGULAR_PANELS``): RMS roll rate and RMS pitch rate [rad/s] (body frame), p95 tilt [deg] — each for
    the payload body (``payload_*``) and the worst segment (row-wise max of ``<metric>@segment i``) — and slip per
    metre of progress [m/m]. Per treatment × controller × level: mean over runs with a bootstrap 95 % interval, n
    per point. ``which``: 'all' (failed runs included; their metrics cover the run up to the failure) or
    'success' (successful runs only) — stated in the title and the table. One commanded speed only (``v_target``
    [m/s] selects it). Table (long): metric, label, unit, terrain, v_target, level, treatment, controller, sigma,
    which, n, mean, lo, hi.
    """
    d, notes = prepare_runs(df)
    d = _filter_terrain(d, terrain)
    d = _filter_value(d, "_v", v_target, "v_target")
    if d["_v"].nunique() > 1:
        raise ValueError(f"runs at several commanded speeds {sorted(d['_v'].unique())}: pass v_target=")
    d, which_text = _which(d, which)
    _check_single(d, by=["_treatment", "_ctrl", "_level"])
    kind = _kind(terrain)
    series = _series(d)
    fig = _figure(15.0, 7.4)
    axes = fig.subplots(2, 4, sharex=True, squeeze=False).ravel()
    rows = []
    levels = np.sort(d["_level"].dropna().unique())
    v = float(d["_v"].iloc[0]) if len(d) else math.nan
    for ax, (key, label, unit) in zip(axes, ANGULAR_PANELS):
        if key.startswith("worst_segment:"):
            values, how = _worst_segment(d, key.split(":", 1)[1])
            label_full = f"{label} ({how})"
        else:
            values, label_full = _num(d, key), label
        for i, (t, clabel, name, sigma) in enumerate(series):
            sel = (d["_treatment"] == t) & (d["_ctrl"] == clabel)
            xs, ys = [], []
            for L in levels:
                r = _mean_ci(values[sel & np.isclose(d["_level"], L)])
                if r["n"] == 0:
                    continue
                x = L + _dodge(i, len(series), levels)
                _err(ax, x, r["mean"], r["lo"], r["hi"], color=COLOR[t], marker=_marker(name, i), ls="none")
                _annotate_n(ax, x, r["hi"] if np.isfinite(r["hi"]) else r["mean"], r["n"])
                xs.append(x)
                ys.append(r["mean"])
                rows.append({"metric": key, "label": label_full, "unit": unit, "terrain": kind, "v_target": v,
                             "level": float(L), "treatment": t, "controller": name, "sigma": sigma, "which": which,
                             **r})
            ax.plot(xs, ys, color=COLOR[t], ls=_line(name), lw=1.3)
        _style(ax)
        ax.margins(y=0.15)
        ax.set_title(textwrap.fill(label_full, 42), fontsize=8.5)
        ax.set_ylabel(f"[{unit}]")
        ax.set_xlabel(_difficulty_label(kind))
    axes[-1].axis("off")
    axes[-1].legend(handles=_legend_handles(series), fontsize=8, loc="center", frameon=False,
                    title=textwrap.fill(which_text, 34), title_fontsize=8)
    _finish(fig, f"Body angular motion and foot slip vs difficulty — {kind}, v_target = {v:g} m/s — {which_text}",
            ["Mean over runs with a bootstrap 95 % CI; n = runs per point. Roll/pitch rates are body-frame; tilt is "
             "the body z axis from vertical; slip is the tangential path of loaded feet per metre of COM progress."]
            + notes)
    return fig, pd.DataFrame(rows)


# ============================================================================================ 4. energy
def cost_vs_speed(df, *, terrain=None):
    """Positive mechanical cost of transport ``CoT = E₊ / (m g d)`` [–] vs achieved speed [m/s], one point per run:
    successful runs filled, failed runs hollow — the CoT of a failed run covers its distance before the failure
    (E₊ and d up to the outcome). Runs without a CoT (no forward progress) are counted in the footnote. One panel
    per terrain kind. Table: one row per plotted run — terrain, level, treatment, controller, sigma, v_target, seed,
    success, reason, achieved_speed_m_s, cot_mech."""
    d, notes = prepare_runs(df)
    d = _filter_terrain(d, terrain)
    if "cot_mech" not in d.columns or "achieved_speed_m_s" not in d.columns:
        raise ValueError("the run table needs 'cot_mech' and 'achieved_speed_m_s'")
    d["_cot"], d["_va"] = _num(d, "cot_mech"), _num(d, "achieved_speed_m_s")
    missing = d["_cot"].isna() | d["_va"].isna()
    if missing.any():
        notes = [f"{int(missing.sum())} run(s) without a CoT (no forward progress) not drawn"] + notes
    d = d.loc[~missing].reset_index(drop=True)
    _check_single(d, by=["_treatment", "_ctrl", "_terrain", "_level"])
    kinds = sorted(d["_terrain"].unique())
    series = _series(d)
    fig = _figure(4.6 * min(3, len(kinds)) + 2.2, 4.0 * int(math.ceil(len(kinds) / 3)) + 0.4)
    axes = _facet_axes(fig, len(kinds), ncols=3)
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    handles = []
    for ax, kind in zip(axes, kinds):
        dk = d.loc[d["_terrain"] == kind]
        for i, (t, label, name, sigma) in enumerate(series):
            g = dk.loc[(dk["_treatment"] == t) & (dk["_ctrl"] == label)]
            ok = (g["_success"] == 1).to_numpy()
            ax.scatter(g["_va"][ok], g["_cot"][ok], s=16, color=COLOR[t], marker=_marker(name, i), alpha=0.65,
                       linewidths=0)
            ax.scatter(g["_va"][~ok], g["_cot"][~ok], s=16, facecolors="none", edgecolors=COLOR[t],
                       marker=_marker(name, i), linewidths=0.9)
            if kind == kinds[-1]:
                n_ok, n_fail = int(((d["_treatment"] == t) & (d["_ctrl"] == label) & (d["_success"] == 1)).sum()), \
                    int(((d["_treatment"] == t) & (d["_ctrl"] == label) & (d["_success"] != 1)).sum())
                handles.append(Line2D([], [], color=COLOR[t], marker=_marker(name, i), ls="",
                                      label=f"{TREATMENT_LABEL[t]} · {label}: n = {n_ok} ok + {n_fail} failed"))
        _style(ax)
        ax.xaxis.set_major_locator(MaxNLocator(6))
        ax.set_title(kind, fontsize=9)
        ax.set_xlabel("achieved speed [m/s]")
        ax.set_ylabel("mechanical CoT  E₊ / (m g d)  [–]")
    handles += [Line2D([], [], color=INK2, marker="o", ls="", label="successful run (filled)"),
                Line2D([], [], color=INK2, marker="o", ls="", markerfacecolor="none",
                       label="failed run (hollow): CoT over its distance before failure")]
    axes[-1].legend(handles=handles, fontsize=7.5, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    _finish(fig, "Positive mechanical cost of transport vs achieved speed",
            ["One point per run (all runs). The CoT of a failed run covers its distance before the failure. "
             "E₊ = ∫Σ max(τ q̇, 0) dt over the leg servos; excludes electrical losses and holding power."] + notes)
    cols = {"_terrain": "terrain", "_level": "level", "_treatment": "treatment", "_controller": "controller",
            "_sigma": "sigma", "_v": "v_target", "seed": "seed", "_success": "success", "reason": "reason",
            "_va": "achieved_speed_m_s", "_cot": "cot_mech"}
    table = d[[c for c in cols if c in d.columns]].rename(columns=cols)
    table["success"] = table["success"].astype(bool)
    return fig, table.reset_index(drop=True)


# ============================================================================================ 5. traces
def _log_and_outcome(x):
    log = getattr(x, "log", x)
    return log, getattr(x, "outcome", None)


def synchronized_traces(log_a, log_b, labels: Sequence[str] = ("spring", "spring_damper"), *, outcomes=None,
                        q0=None, bodies: Sequence[str] | None = None, t_max: float | None = None):
    """Two paired runs on a shared time axis [s]: each segment's roll and pitch [deg] (intrinsic z-y-x Euler,
    ``chiron.metrics`` convention), the feet's normal forces [N] stacked per segment, and the body-joint deflections
    ``q − q0`` [deg] per axis (yaw, pitch, roll), one column per run.

    ``log_a``, ``log_b``: episode logs (dicts) or ``Episode`` s (their outcome is marked); ``labels``: the two
    column titles; ``outcomes``: the two outcome dicts when logs are given (``t_end`` [s] is marked); ``q0``: the
    body joints' spring rest angles [rad] (as ``chiron.export.raw_frame``; default the log's, else 0); ``bodies``:
    which logged bodies to draw (default the ``segment *`` bodies, else all); ``t_max`` [s]: cut the time axis.
    Table (long): run, t [s], panel, entity, unit, value.
    """
    import warnings

    runs = []
    for i, x in enumerate((log_a, log_b)):
        log, out = _log_and_outcome(x)
        if outcomes is not None and outcomes[i] is not None:
            out = outcomes[i]
        with warnings.catch_warnings():
            if q0 is None:
                warnings.simplefilter("ignore", UserWarning)
            raw = cx.raw_frame(log, every=1, q0=q0)
        if t_max is not None:
            raw = raw.loc[raw["t"] <= t_max + 1e-12].reset_index(drop=True)
        runs.append((str(labels[i]), log, out, raw))
    log0 = runs[0][1]
    all_bodies = [str(b) for b in log0["bodies"]]
    if bodies is None:
        segs = [b for b in all_bodies if b.startswith("segment")]
        bodies = segs or all_bodies
    feet = [str(f) for f in log0["feet"]]
    foot_body = [all_bodies[int(i)] for i in np.asarray(log0["foot_body"]).reshape(-1)]
    support_bodies = [b for b in bodies if b in foot_body]
    joints = [str(j) for j in log0["joints"]]
    kinds = [str(k) for k in log0["joint_kind"]]
    axes_kinds = [k for k in ("body_yaw", "body_pitch", "body_roll") if k in kinds]
    panels = ([("roll", "roll [deg]"), ("pitch", "pitch [deg]")]
              + [(f"normal:{b}", f"{b}\nfoot normal force [N]") for b in support_bodies]
              + [(f"defl:{k}", f"{k.split('_', 1)[1]} deflection\nq − q0 [deg]") for k in axes_kinds])
    fig = _figure(12.0, 1.75 * len(panels) + 1.0)
    grid = fig.subplots(len(panels), 2, sharex=True, sharey="row", squeeze=False)
    rows = []

    def leg_key(f):
        return f.split(" ", 1)[1] if " " in f else f

    leg_colors = {}
    for f in feet:
        leg_colors.setdefault(leg_key(f), ENTITY_COLORS[len(leg_colors) % len(ENTITY_COLORS)])
    for col, (label, log, out, raw) in enumerate(runs):
        t = raw["t"].to_numpy()
        for r, (key, ylabel) in enumerate(panels):
            ax = grid[r, col]
            if key in ("roll", "pitch"):
                for i, b in enumerate(bodies):
                    c = f"{key}@{b}"
                    if c in raw:
                        y = np.degrees(raw[c].to_numpy())
                        ax.plot(t, y, color=ENTITY_COLORS[i % 8], lw=1.0, label=b)
                        rows.append(pd.DataFrame({"run": label, "t": t, "panel": key, "entity": b, "unit": "deg",
                                                  "value": y}))
            elif key.startswith("normal:"):
                b = key.split(":", 1)[1]
                fs = [f for f, fb in zip(feet, foot_body) if fb == b and f"normal_force@{f}" in raw]
                ys = [raw[f"normal_force@{f}"].to_numpy() for f in fs]
                if fs:
                    ax.stackplot(t, *ys, colors=[leg_colors[leg_key(f)] for f in fs], labels=[leg_key(f) for f in fs],
                                 alpha=0.85, edgecolor="white", linewidth=0.3)
                for f, y in zip(fs, ys):
                    rows.append(pd.DataFrame({"run": label, "t": t, "panel": "normal_force", "entity": f, "unit": "N",
                                              "value": y}))
            else:
                k = key.split(":", 1)[1]
                for i, j in enumerate(jn for jn, kk in zip(joints, kinds) if kk == k):
                    c = f"deflection@{j}"
                    if c in raw:
                        y = np.degrees(raw[c].to_numpy())
                        ax.plot(t, y, color=ENTITY_COLORS[i % 8], lw=1.0, label=j)
                        rows.append(pd.DataFrame({"run": label, "t": t, "panel": "deflection", "entity": j,
                                                  "unit": "deg", "value": y}))
            if out is not None and out.get("t_end") is not None and np.isfinite(float(out["t_end"])):
                ax.axvline(float(out["t_end"]), color=INK2, lw=0.8, ls=":")
            _style(ax)
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=8)
            if col == 1 and ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=6.5, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
        head = label
        if out is not None:
            head += f" — {out.get('reason', '?')}"
            if out.get("t_end") is not None:
                head += f" at {float(out['t_end']):.2f} s"
        grid[0, col].set_title(head, fontsize=9)
        grid[-1, col].set_xlabel("walking time t [s]")
    meta = [f"{k} {log0.get(k)}" for k in ("robot", "terrain", "seed", "v_target") if k in log0]
    notes = ["Shared time axis; dotted line: the outcome. Roll/pitch: intrinsic z-y-x Euler angles of each body; "
             "normal forces stacked per segment; deflection = q − q0 of each body hinge.", "; ".join(map(str, meta))]
    for label, _, _, raw in runs:
        nf = raw[[c for c in raw.columns if c.startswith("normal_force@")]].to_numpy(float)
        neg = int(np.sum(nf < 0))
        if neg:
            notes.append(f"{label}: the net contact force on a foot points into the ground (N < 0, down to "
                         f"{np.nanmin(nf):.0f} N) in {neg} of {nf.size} foot-samples — drawn as is (stacks dip below "
                         "0); a pad sunk into the height field and held by its side walls gives this.")
    _finish(fig, "Synchronised traces of a paired run", notes)
    table = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["run", "t", "panel", "entity", "unit", "value"])
    return fig, table


# ============================================================================================ 6. paired forest
def paired_difference_forest(df, metric: str = "success", *, better: str | None = None, terrain=None,
                             which: str = "all", alpha: float = ALPHA):
    """Paired effect of damping, ``spring_damper − spring``, per terrain × level × controller (× speed when the
    table has several), with 95 % paired-bootstrap intervals and the §11.6 verdict.

    ``metric``: ``'success'`` (rate difference; the interval is ``stats.success_difference_ci(method='bootstrap')``
    and the exact McNemar p is reported and required, p < ``alpha``) or a metric column (mean paired difference,
    ``stats.paired_difference_ci``; the McNemar part of the rule does not apply). ``better``: 'higher' or 'lower'
    (default from ``BETTER``; an unknown metric needs it). ``which``: 'all' runs (failures included) or 'success'
    (pairs where both runs succeeded). Table: terrain, level, controller, sigma, v_target, metric, which, n (pairs),
    n_unpaired, estimate, lo, hi, mean_spring_damper, mean_spring, p_mcnemar, n10 (only spring–damper succeeded),
    n01 (only spring-only succeeded), better, verdict.
    """
    better = better or ("higher" if metric == "success" else BETTER.get(metric))
    if better not in ("higher", "lower"):
        raise ValueError(f"say which direction of {metric!r} is better: better='higher' or 'lower'")
    d, notes = prepare_runs(df)
    d = _filter_terrain(d, terrain)
    value = "_success" if metric == "success" else metric
    if value not in d.columns:
        raise ValueError(f"no column {metric!r} in the run table")
    if metric != "success":
        d[value] = _num(d, value)
    if which not in ("all", "success"):
        raise ValueError("which must be 'all' or 'success'")
    _check_single(d, by=["_terrain", "_level", "_ctrl", "_v"])
    multi_v = d["_v"].nunique() > 1
    cells = d[["_terrain", "_level", "_ctrl", "_controller", "_sigma", "_v"]].drop_duplicates()
    order = {n: i for i, n in enumerate(CONTROLLER_ORDER)}
    cells = sorted(cells.itertuples(index=False),
                   key=lambda r: (r[0], np.nan_to_num(r[1], nan=-1), np.nan_to_num(r[5]), order.get(r[3], 99), r[2]))
    rows, skipped = [], []
    for kind, L, label, name, sigma, v in cells:
        same_level = np.isclose(d["_level"], L) | (np.isnan(L) & d["_level"].isna())
        sel = (d["_terrain"] == kind) & (d["_ctrl"] == label) & same_level
        if multi_v:
            sel &= np.isclose(d["_v"], v)
        g = d.loc[sel]
        if set(g["_treatment"]) != set(TREATMENTS):
            skipped.append(f"{kind} {L:g} {label}")
            continue
        wide, n_pairs, unmatched = _pairs(g, "_treatment", TREATMENTS, [value, "_success"])
        x, y = wide[(value, "spring_damper")], wide[(value, "spring")]
        if which == "success":
            both = (wide[("_success", "spring_damper")] == 1) & (wide[("_success", "spring")] == 1)
            x, y = x[both], y[both]
        row = {"terrain": kind, "level": L, "controller": name, "sigma": sigma, "v_target": v, "metric": metric,
               "which": which, "n_unpaired": unmatched, "better": better}
        if metric == "success":
            r = stats.success_difference_ci(x, y, method="bootstrap", n_boot=N_BOOT, seed=BOOT_SEED)
            row.update({"n": r["n"], "estimate": r["difference"], "lo": r["lo"], "hi": r["hi"],
                        "mean_spring_damper": r["rate_x"], "mean_spring": r["rate_y"], "p_mcnemar": r["p_mcnemar"],
                        "n10": r["n10"], "n01": r["n01"]})
            row["verdict"] = classify(r["lo"], r["hi"], better, r["p_mcnemar"], alpha)
        else:
            r = stats.paired_difference_ci(x, y, n_boot=N_BOOT, seed=BOOT_SEED)
            row.update({"n": r["n"], "estimate": r["estimate"], "lo": r["lo"], "hi": r["hi"],
                        "mean_spring_damper": r["mean_x"], "mean_spring": r["mean_y"], "p_mcnemar": math.nan,
                        "n10": math.nan, "n01": math.nan})
            row["verdict"] = classify(r["lo"], r["hi"], better) if r["n"] >= 2 else "no clear difference"
        rows.append(row)
    table = pd.DataFrame(rows)
    if table.empty:
        raise ValueError("no cell has runs of both treatments to pair")
    if skipped:
        notes = [f"cells with runs of one treatment only (not compared): {', '.join(skipped)}"] + notes
    fig = _figure(10.0, 0.34 * len(table) + 1.9)
    ax = fig.subplots()
    ys = np.arange(len(table))[::-1]
    for yv, r in zip(ys, table.itertuples(index=False)):
        c = VERDICT_COLOR[r.verdict]
        if np.isfinite(r.lo) and np.isfinite(r.hi):
            ax.plot([r.lo, r.hi], [yv, yv], color=c, lw=2.0, solid_capstyle="round")
        ax.plot([r.estimate], [yv], marker="o", color=c, markersize=6, markeredgecolor="white", markeredgewidth=1.0)
        extra = f" · McNemar p = {r.p_mcnemar:.3g}" if metric == "success" else ""
        ax.annotate(f"{r.verdict}{extra} · n = {r.n} pairs", (1.01, yv), xycoords=("axes fraction", "data"),
                    fontsize=7.5, color=INK2, va="center")
    labels = [f"{r.terrain} {'angle' if r.terrain == 'cross_slope' else 'h/L'} {r.level:g} · {r.controller}"
              + (f" σ={r.sigma:g}" if r.controller != "fixed" and np.isfinite(r.sigma) else "")
              + (f" · {r.v_target:g} m/s" if multi_v else "") for r in table.itertuples(index=False)]
    ax.set_yticks(ys, labels, fontsize=7.5)
    ax.axvline(0.0, color=INK, lw=0.8)
    _style(ax)
    ax.grid(axis="y", visible=False)
    unit = "" if metric == "success" else f" [{_unit_of(metric)}]"
    what = "success rate" if metric == "success" else metric
    ax.set_xlabel(f"Δ {what}{unit}: spring–damper − spring-only (paired, 95 % bootstrap CI) — {better} is better",
                  fontsize=8.5)
    which_text = ("all runs, failed runs included" if which == "all" else "pairs where both runs succeeded")
    rule = ("helps / hurts: McNemar p < 0.05 and the interval excludes 0 (§11.6)" if metric == "success" else
            "helps / hurts: the paired interval excludes 0 (§11.6; no McNemar test for a continuous metric)")
    _finish(fig, f"Effect of joint damping on {what} — {which_text}",
            [f"Verdict colour and text: helps (green), hurts (red), no clear difference (grey); {rule}. "
             f"Runs paired on seed and configuration; n = pairs."] + notes)
    return fig, table


# ============================================================================================ 7. factorial
def factorial_interaction(df, terrain: str, level: float, *, v_target=None,
                          controllers: Sequence[str] = CONTROLLER_ORDER):
    """Treatment × controller on one terrain and level: success per cell with Wilson 95 % intervals, the damping
    effect within each controller and the interaction (difference of paired differences)
    ``(spring_damper·adaptive − spring_damper·fixed) − (spring·adaptive − spring·fixed)`` with its paired
    bootstrap 95 % interval over seeds (``stats.difference_of_differences_ci``; seeds complete in all four cells).

    ``controllers``: (baseline, load feedback) names. Table (long, column ``kind``): 'cell' rows — treatment,
    controller, sigma, n, k, rate, lo, hi; 'effect' rows — the paired damping effect (spring_damper − spring) per
    controller; one 'interaction' row — estimate, lo, hi, n (seeds).
    """
    d, notes = prepare_runs(df)
    d = _filter_terrain(d, terrain)
    d = _filter_value(d, "_level", level, "level")
    d = _filter_value(d, "_v", v_target, "v_target")
    if d["_v"].nunique() > 1:
        raise ValueError(f"runs at several commanded speeds {sorted(d['_v'].unique())}: pass v_target=")
    base, fb = controllers
    d = d.loc[d["_controller"].isin(controllers) & d["_success"].notna()].reset_index(drop=True)
    _check_single(d, by=["_treatment", "_ctrl"])
    d["_cell"] = d["_treatment"] + "·" + d["_controller"]
    cells = [f"{t}·{c}" for t in TREATMENTS for c in controllers]
    sig = {c: float(d.loc[d["_controller"] == c, "_sigma"].iloc[0]) if (d["_controller"] == c).any() else math.nan
           for c in controllers}
    rows = []
    for t in TREATMENTS:
        for c in controllers:
            r = _rate(d.loc[d["_cell"] == f"{t}·{c}", "_success"])
            rows.append({"kind": "cell", "treatment": t, "controller": c, "sigma": sig[c], **r})
    wide, n_seeds, unmatched = _pairs(d, "_cell", cells, ["_success"],
                                      free=("controller_factory", "controller_kwargs.sigma"))
    v = {k[1]: wide[k] for k in wide}
    effects = []
    for c in controllers:
        e = stats.success_difference_ci(v[f"spring_damper·{c}"], v[f"spring·{c}"], method="bootstrap", n_boot=N_BOOT,
                                        seed=BOOT_SEED)
        effects.append({"kind": "effect", "treatment": "spring_damper − spring", "controller": c, "sigma": sig[c],
                        "n": e["n"], "estimate": e["difference"], "lo": e["lo"], "hi": e["hi"],
                        "p_mcnemar": e["p_mcnemar"],
                        "verdict": classify(e["lo"], e["hi"], "higher", e["p_mcnemar"])})
    dd = stats.difference_of_differences_ci(v[f"spring_damper·{fb}"], v[f"spring_damper·{base}"], v[f"spring·{fb}"],
                                            v[f"spring·{base}"], n_boot=N_BOOT, seed=BOOT_SEED)
    inter = {"kind": "interaction", "treatment": "(spring_damper − spring)", "controller": f"{fb} − {base}",
             "n": dd["n"], "estimate": dd["estimate"], "lo": dd["lo"], "hi": dd["hi"], "n_unpaired": unmatched}
    table = pd.DataFrame(rows + effects + [inter])
    kind = _kind(terrain)
    fig = _figure(11.0, 4.3)
    ax1, ax2 = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1.0, 1.15]})
    for i, t in enumerate(TREATMENTS):
        xs, ys = [], []
        for j, c in enumerate(controllers):
            r = next(x for x in rows if x["treatment"] == t and x["controller"] == c)
            x = j + (i - 0.5) * 0.16
            _err(ax1, x, r["rate"], r["lo"], r["hi"], color=COLOR[t], marker=_marker(c, j), ls="none")
            _annotate_n(ax1, x, r["hi"], r["n"])
            xs.append(x)
            ys.append(r["rate"])
        ax1.plot(xs, ys, color=COLOR[t], lw=1.5, label=TREATMENT_LABEL[t])
    ax1.set_xticks(range(len(controllers)), [_ctrl_label(c, sig[c]) for c in controllers])
    ax1.set_xlim(-0.5, len(controllers) - 0.5)
    ax1.set_ylim(-0.03, 1.2)
    ax1.set_ylabel("success rate (Wilson 95 % CI)")
    ax1.set_title("success per cell", fontsize=9)
    ax1.legend(fontsize=7.5, frameon=False, loc="lower right")
    _style(ax1)
    items = [(f"damping effect | {c}", e) for c, e in zip(controllers, effects)] + [("interaction (diff. of diffs)",
                                                                                      inter)]
    for k, (lab, e) in enumerate(items):
        yv = len(items) - 1 - k
        col = VERDICT_COLOR.get(e.get("verdict", ""), INK)
        if np.isfinite(e["lo"]) and np.isfinite(e["hi"]):
            ax2.plot([e["lo"], e["hi"]], [yv, yv], color=col, lw=2.0, solid_capstyle="round")
        ax2.plot([e["estimate"]], [yv], marker="o", color=col, markersize=6, markeredgecolor="white")
        extra = f" · {e['verdict']} (p = {e['p_mcnemar']:.3g})" if "verdict" in e else ""
        ax2.annotate(f"{e['estimate']:+.2f} [{e['lo']:+.2f}, {e['hi']:+.2f}] · n = {e['n']}{extra}", (1.01, yv),
                     xycoords=("axes fraction", "data"), fontsize=7.5, color=INK2, va="center")
    ax2.set_yticks(range(len(items))[::-1], [i[0] for i in items], fontsize=8)
    ax2.axvline(0.0, color=INK, lw=0.8)
    ax2.set_xlabel("Δ success rate (paired over seeds, 95 % bootstrap CI)")
    ax2.set_title("paired effects", fontsize=9)
    _style(ax2)
    ax2.grid(axis="y", visible=False)
    _finish(fig, f"Treatment × controller — {kind} {'angle' if kind == 'cross_slope' else 'h/L'} = {level:g}"
            + (f", v_target = {float(d['_v'].iloc[0]):g} m/s" if len(d) and np.isfinite(d['_v'].iloc[0]) else ""),
            [f"All runs. Interaction = (spring–damper·{fb} − spring–damper·{base}) − (spring·{fb} − spring·{base}) "
             f"per seed; {n_seeds} seed(s) complete in all four cells"
             + (f", {unmatched} run(s) without a complete seed set" if unmatched else "") + "."] + notes)
    return fig, table


# ============================================================================================ 8. undulation
def undulation_vs_speed(df, *, terrain="flat", which: str = "all", threshold_deg: float = 5.0,
                        metric: str = "body_yaw_rms_max_deg", default_k: float = 8.0):
    """Body undulation vs commanded speed per body-yaw stiffness and treatment, with the onset speed.

    y: ``metric`` (default ``body_yaw_rms_max_deg``: the largest RMS body-yaw joint angle [deg] in steady walking,
    protocol §9.2), mean over runs with a bootstrap 95 % interval and n per point; x: ``v_target`` [m/s]; colour:
    k = ``robot_kwargs.body_k_yaw`` [N·m/rad] (``default_k``, protocol §12.1's 8 N·m/rad, where the column is
    missing); line style: treatment; one panel per controller. Onset (``stats.onset_speed``, threshold 5° by
    default): the first speed whose mean exceeds the threshold (``onset_speed_m_s``, marked ▲), the interpolated
    crossing and its bootstrap interval. ``which``: 'all' runs (failed runs included, over their steady window up to
    the failure) or 'success'. Table: controller, sigma, k_yaw, treatment, v_target, which, n, mean, lo, hi,
    onset_speed_m_s, onset_interp_m_s, onset_lo, onset_hi, onset_flag.
    """
    d, notes = prepare_runs(df)
    d = _filter_terrain(d, terrain)
    if metric not in d.columns:
        raise ValueError(f"no column {metric!r} in the run table")
    d["_y"] = _num(d, metric)
    d, which_text = _which(d, which)
    k = _num(d, "robot_kwargs.body_k_yaw")
    d["_k"] = k.fillna(float(default_k))
    _check_single(d, by=["_treatment", "_ctrl", "_k", "_level"], extra_ok=("robot_kwargs.body_k_yaw",))
    series_c = list({lab: (lab, name, sig) for _, lab, name, sig in _series(d)}.values())
    ks = sorted(d["_k"].unique())
    ramp = ([STIFFNESS_RAMP[int(round(i))] for i in np.linspace(0, len(STIFFNESS_RAMP) - 1, len(ks))]
            if len(ks) <= len(STIFFNESS_RAMP) else None)                    # light = soft ... dark = stiff
    fig = _figure(5.2 * len(series_c) + 2.4, 4.4)
    axes = fig.subplots(1, len(series_c), sharey=True, squeeze=False)[0]
    rows = []
    for ax, (label, name, sigma) in zip(axes, series_c):
        dc = d.loc[d["_ctrl"] == label]
        speeds = np.sort(dc["_v"].dropna().unique())
        ax.axhline(threshold_deg, color=INK2, lw=0.8, ls=":")
        n_series = len(ks) * len(TREATMENTS)
        for a, kk in enumerate(ks):
            color = ramp[a] if ramp else ENTITY_COLORS[a % 8]
            for b, t in enumerate(TREATMENTS):
                g = dc.loc[(dc["_k"] == kk) & (dc["_treatment"] == t)]
                if g.empty:
                    continue
                on = stats.onset_speed(g["_v"], g["_y"], threshold_deg, n_boot=2000, seed=BOOT_SEED)
                xs, ys = [], []
                for v in speeds:
                    r = _mean_ci(g.loc[np.isclose(g["_v"], v), "_y"])
                    if r["n"] == 0:
                        continue
                    x = v + _dodge(a * len(TREATMENTS) + b, n_series, speeds)
                    _err(ax, x, r["mean"], r["lo"], r["hi"], color=color, marker="o" if t == "spring" else "D",
                         ls="none", markersize=5)
                    _annotate_n(ax, x, r["hi"] if np.isfinite(r["hi"]) else r["mean"], r["n"])
                    xs.append(x)
                    ys.append(r["mean"])
                    rows.append({"controller": name, "sigma": sigma, "k_yaw": kk, "treatment": t, "v_target": float(v),
                                 "which": which, **r, "onset_speed_m_s": on["onset_level"] if on["onset_level"]
                                 is not None else math.nan, "onset_interp_m_s": on["onset"], "onset_lo": on["lo"],
                                 "onset_hi": on["hi"], "onset_flag": on["flag"]})
                ax.plot(xs, ys, color=color, ls="-" if t == "spring" else "--", lw=1.4,
                        label=f"k = {kk:g} N·m/rad · {TREATMENT_LABEL[t]}")
                if on["onset_level"] is not None:
                    ax.plot([on["onset_level"]], [threshold_deg], marker="^", color=color, markersize=9,
                            markeredgecolor=INK, ls="none")
        _style(ax)
        ax.margins(y=0.12)
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("commanded speed v_target [m/s]")
    axes[0].set_ylabel(f"{metric} [deg]" if metric.endswith("_deg") else metric)
    axes[-1].legend(fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    _finish(fig, f"Body undulation vs speed — {_kind(terrain)} — {which_text}",
            [f"Mean over runs with a bootstrap 95 % CI; n = runs per point. Dotted: the {threshold_deg:g}° onset "
             f"threshold; ▲ onset speed = the first speed whose mean exceeds it. Solid/circles: spring-only, "
             f"dashed/diamonds: spring–damper; darker = stiffer."] + notes)
    return fig, pd.DataFrame(rows)
