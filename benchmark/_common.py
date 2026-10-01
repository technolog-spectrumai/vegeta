"""Shared machinery of the Myropod benchmarks (benchmark/cleopatra, benchmark/persephone): run metadata, the results
JSON, the plots and the Markdown report. Nothing here simulates; each benchmark's ``full_benchmark.py`` runs its
experiments through ``vegeta.chiron.experiments.run_trials`` (cached, one forked process per run, tqdm progress)
and hands the per-run tables to these functions.

Statistical rules (docs/myropod_stability.md §7, §11.6, §12.6): runs — not frames — are the samples; success
rates carry Wilson 95 % intervals, means and paired differences percentile-bootstrap intervals over runs; failed
runs are always counted and shown (hollow markers), and any table restricted to successful runs says so; the
verdict *helps* / *hurts* / *no clear difference* of spring–damper vs spring-only follows §11.6.
"""
from __future__ import annotations

import datetime as _dt
import json
import math
import platform
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DESIGNS = REPO / "notebooks" / "designs"
for _p in (str(DESIGNS),):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ------------------------------------------------------------------------------------------------- metadata
def now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def metadata(benchmark: str, scale: str, processes: int, experiments) -> dict:
    """Who ran what with which code: git commit and state, package versions, machine, scale."""
    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:  # noqa: BLE001
            return ""
    versions = {"python": platform.python_version()}
    for mod in ("mujoco", "numpy", "scipy", "pandas", "matplotlib"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            versions[mod] = "missing"
    return {"benchmark": benchmark, "scale": scale, "processes": processes, "experiments": list(experiments),
            "started_utc": now(), "git_commit": git("rev-parse", "HEAD"), "git_branch": git("rev-parse", "--abbrev-ref",
                                                                                            "HEAD"),
            "git_dirty": bool(git("status", "--porcelain")), "versions": versions,
            "machine": {"platform": platform.platform(), "cpus": __import__("os").cpu_count()},
            "protocol": "docs/myropod_stability.md (Amendments A–F)"}


# ---------------------------------------------------------------------------------------------------- JSON
def _jsonable(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if not math.isfinite(float(v)) else float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (Path,)):
        return str(v)
    if isinstance(v, np.ndarray):
        return [_jsonable(x) for x in v.tolist()]
    return v


def rows(df: pd.DataFrame) -> list[dict]:
    return [{k: _jsonable(v) for k, v in r.items()} for r in df.to_dict(orient="records")]


def save_results_json(path: Path, meta: dict, configs: list, tables: dict) -> Path:
    """``results.json``: metadata, configurations and every run's row, per experiment."""
    doc = {"metadata": {**meta, "finished_utc": now()}, "configs": [{k: _jsonable(v) for k, v in c.items()}
                                                                    for c in configs],
           "experiments": {name: {"n_runs": int(len(df)), "rows": rows(df)} for name, df in tables.items()}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1, default=str))
    return path


def load_results_json(path: Path) -> tuple[dict, dict]:
    """(document, {experiment: DataFrame}) from a ``results.json``."""
    doc = json.loads(Path(path).read_text())
    return doc, {k: pd.DataFrame(v["rows"]) for k, v in doc["experiments"].items()}


# --------------------------------------------------------------------------------------------------- plots
class PlotBook:
    """Saves figures as PNG and remembers what failed, so one broken plot never stops the others."""

    def __init__(self, out: Path):
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.saved: list[str] = []
        self.failed: list[tuple[str, str]] = []
        self.tables: dict[str, pd.DataFrame] = {}

    def __call__(self, name: str, fn, *args, **kwargs):
        try:
            res = fn(*args, **kwargs)
            fig, table = res if isinstance(res, tuple) else (res, None)
            fig.savefig(self.out / f"{name}.png", dpi=130, bbox_inches="tight")
            import matplotlib.pyplot as plt
            plt.close(fig)
            self.saved.append(f"{name}.png")
            if isinstance(table, pd.DataFrame):
                self.tables[name] = table
                table.to_csv(self.out / f"{name}.csv", index=False)
            return table
        except Exception as exc:  # noqa: BLE001
            self.failed.append((name, f"{type(exc).__name__}: {exc}"))
            (self.out / f"{name}.error.txt").write_text(traceback.format_exc())
            return None


def recovery_plot(df: pd.DataFrame):
    """Recovery probability vs push impulse per treatment × controller (Wilson intervals, logistic J50).
    Every run counts: a run that fell or stalled is 'not recovered'."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from vegeta.chiron import stats
    d = df[df.get("status", "ok") == "ok"].copy()
    jcol = "info.impulse_Ns" if "info.impulse_Ns" in d else "info.impulse_level_Ns"
    fig = Figure(figsize=(7.5, 4.5))
    ax = fig.add_subplot(111)
    out = []
    if "recovered" not in d:
        d["recovered"] = np.nan
    # J = 0 has no push: the baseline counts a run as 'recovered' when it walks the course without failing
    rec = d["recovered"].where(d["recovered"].notna(), d["success"])
    d["_rec"] = rec.map(lambda x: bool(x) if x is not None and x == x else False)
    for (tr, ctrl), g in d.groupby(["treatment", "controller"]):
        rc = stats.recovery_curve(g[jcol].astype(float), g["_rec"])
        lv = pd.DataFrame(rc["levels"])
        if len(lv):
            ax.errorbar(lv["impulse_Ns"], lv["rate"], yerr=[lv["rate"] - lv["lo"], lv["hi"] - lv["rate"]], capsize=3,
                        marker="o" if ctrl == "fixed" else "s", ls="-" if ctrl == "fixed" else "--",
                        label=f"{tr}, {ctrl} (J50 {rc.get('j50_Ns', float('nan')):.2g} N·s)")
            lv = lv.assign(treatment=tr, controller=ctrl, j50_Ns=rc.get("j50_Ns"))
            out.append(lv)
    ax.set(xlabel="sideways impulse on segment 2 [N·s]", ylabel="recovery probability",
           title="Recovery vs push (all runs; Wilson 95 % intervals)", ylim=(-0.05, 1.05))
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    return fig, (pd.concat(out) if out else pd.DataFrame())


def sensitivity_plot(df: pd.DataFrame, factor: str, label: str):
    """Success rate (Wilson) and mean distance (bootstrap) vs one robot factor, per terrain — all runs."""
    from matplotlib.figure import Figure
    from vegeta.chiron import stats
    d = df[df.get("status", "ok") == "ok"].copy()
    fig = Figure(figsize=(10, 4))
    a1, a2 = fig.add_subplot(121), fig.add_subplot(122)
    rows_ = []
    for (terrain, tr), g in d.groupby(["info.terrain", "treatment"]):
        pts = []
        for x, h in g.groupby(factor):
            k, n = int(h["success"].astype(bool).sum()), int(len(h))
            lo, hi = stats.wilson_ci(k, n)
            dist = h["distance_m"].astype(float).to_numpy()
            pts.append((x, k / n if n else math.nan, lo, hi, float(np.mean(dist)), n))
        pts.sort(key=lambda r: (str(type(r[0])), r[0]))
        xs = [str(p[0]) for p in pts]
        a1.errorbar(xs, [p[1] for p in pts], yerr=[[p[1] - p[2] for p in pts], [p[3] - p[1] for p in pts]],
                    marker="o", capsize=3, label=f"{terrain}, {tr}")
        a2.plot(xs, [p[4] for p in pts], marker="o", label=f"{terrain}, {tr}")
        rows_ += [dict(terrain=terrain, treatment=tr, value=p[0], success_rate=p[1], lo=p[2], hi=p[3],
                       mean_distance_m=p[4], n=p[5]) for p in pts]
    a1.set(xlabel=label, ylabel="success rate", title="success (all runs; Wilson 95 %)", ylim=(-0.05, 1.05))
    a2.set(xlabel=label, ylabel="distance before failure or finish [m]", title="mean distance (all runs)")
    for a in (a1, a2):
        a.grid(alpha=0.3)
        a.legend(fontsize=7)
    return fig, pd.DataFrame(rows_)


# -------------------------------------------------------------------------------------------------- report
def outcome_table(df: pd.DataFrame, keys) -> pd.DataFrame:
    """Runs and outcome counts per cell — failures listed by reason, simulation errors separately."""
    d = df.copy()
    keys = [k for k in keys if k in d.columns]
    d["outcome"] = np.where(d.get("status", "ok") != "ok", "sim_error", d["reason"].astype(str))
    t = d.groupby(keys, dropna=False)["outcome"].value_counts().unstack(fill_value=0)
    t.insert(0, "runs", t.sum(axis=1))
    if "success" in t.columns:
        t.insert(1, "success_rate", (t["success"] / t["runs"]).round(3))
    return t.reset_index()


def write_report(path: Path, title: str, meta: dict, sections: list[tuple[str, str]], book: PlotBook,
                 limitations: list[str]) -> Path:
    lines = [f"# {title}", "", f"Scale **{meta['scale']}**, {meta['processes']} processes, commit "
             f"`{meta['git_commit'][:10]}`{' (dirty tree)' if meta.get('git_dirty') else ''}, started "
             f"{meta['started_utc']}, finished {now()}.", "",
             "Runs are the samples. Failed runs are counted everywhere; a table that uses successful runs only says "
             "so. Verdicts follow docs/myropod_stability.md §11.6 (helps / hurts when the paired interval excludes 0 "
             "— and for success McNemar p < 0.05 — otherwise *no clear difference*).", ""]
    for head, body in sections:
        lines += [f"## {head}", "", body, ""]
    lines += ["## Plots", ""] + [f"- ![{p}](plots/{p})" for p in book.saved]
    if book.failed:
        lines += ["", "### Plots that could not be made", ""] + [f"- `{n}`: {e}" for n, e in book.failed]
    lines += ["", "## Limitations", ""] + [f"- {x}" for x in limitations]
    path.write_text("\n".join(lines) + "\n")
    return path


def md_table(df: pd.DataFrame, max_rows: int = 200) -> str:
    if df is None or not len(df):
        return "_(no rows)_"
    d = df.head(max_rows)
    try:
        return d.to_markdown(index=False)
    except Exception:  # noqa: BLE001 - tabulate missing
        return "```\n" + d.to_string(index=False) + "\n```"
