"""The autopilot tuner's training history as figures (``viz_autopilot_train_hist.sh``), from a course's checkpoint
``races/runs/tune/<course>.ckpt.json`` (every lap, and for CMA-ES the mean, σ and covariance of every generation)
or, without one, its ``<course>_converge.csv``.

**Candidates.** The laps are grouped into candidates (a setting flown in one generation / iteration) and scored as
the tuner scored them: the mean of the worst ⌈(K+1)/3⌉ of the K+1 laps (``races.validate.worst_mean``). A candidate
that racing dropped before its last scenarios has only a lower bound; it is drawn hollow.

**The search plane: principal component analysis.** The six settings, normalised to [0, 1] as the strategies search
them (``races.converge.to_x``), are centred and decomposed by SVD over every candidate of the run. PC1 and PC2 are the
two directions in which the search spread most; the axes give their share of the variance, the loadings panel which
settings make them. The candidates are projected onto that plane, and for CMA-ES so is each generation's sampling
distribution: the ellipse of 2σ√C projected (Wᵀ σ²C W, W the two loadings), drawn where the mean was — the
population's spread shrinking onto the optimum, as in the classic CMA-ES illustrations. The plane is a projection:
two candidates close on it may differ in the four directions it does not show.

Figures per course: ``<course>_train_history.png`` (the loss history, the plane by generation with the ellipses, the
plane by score, the loadings), ``<course>_params.png`` (every setting in its own units against the generation) and,
with ``animate``, ``<course>_search.gif`` (the plane generation by generation)."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from races import converge, validate

TUNE = Path(__file__).resolve().parent / "runs" / "tune"
UNITS = {"safety": "", "lookahead_s": "s", "omega_pos": "rad/s", "omega_att": "rad/s", "tilt_max_deg": "°", "v_cap": "m/s"}


@dataclass
class Candidate:
    step: int                       # the generation / iteration, in the order the run flew them
    stage: str
    settings: dict
    x: np.ndarray                   # normalised settings (to_x)
    score: float
    exact: bool                     # False: dropped by racing, ``score`` is a lower bound
    laps: int


@dataclass
class History:
    course: str
    strategy: str
    k1: int                         # laps per candidate (the nominal lap + K training scenarios)
    candidates: list
    start: dict | None = None
    steps: list = field(default_factory=list)       # step index → label ("cma 3", "pso 2", "gps 5")
    cma: list = field(default_factory=list)          # CMA-ES generations: mean, sigma, C (normalised coordinates)

    @property
    def X(self) -> np.ndarray:
        return np.array([c.x for c in self.candidates])

    @property
    def best_so_far(self) -> list:
        """(step, best exact score up to that step) for every step."""
        out, b = [], math.inf
        for k in range(len(self.steps)):
            sc = [c.score for c in self.candidates if c.step == k and c.exact]
            b = min([b, *sc])
            out.append((k, b))
        return out


def _num(v):
    if v is None or v == "" or (isinstance(v, float) and math.isnan(v)):
        return None
    return float(v)


def _settings(r: dict) -> dict:
    return {k: _num(r.get(k)) for k in converge.KEYS}


def load(course: str, path: Path | None = None) -> History:
    """A course's training history from its checkpoint (preferred) or CSV, or from ``path`` (either kind)."""
    path = Path(path) if path else (TUNE / f"{course}.ckpt.json" if (TUNE / f"{course}.ckpt.json").exists() else TUNE / f"{course}_converge.csv")
    strategy, k1, start, cma = "unknown", None, None, []
    if path.suffix == ".json":
        ck = json.loads(path.read_text())
        rows, fp = ck.get("rows") or [], ck.get("fingerprint") or {}
        strategy = fp.get("strategy") or fp.get("method") or "pso"
        k1 = fp.get("train_k", 0) + 1 if "train_k" in fp else None
        start = ck.get("start")
        cma = ((ck.get("strategy") or {}).get("history")) or []
    else:
        import csv
        rows = list(csv.DictReader(path.open()))
    if not rows:
        raise ValueError(f"no laps in {path}")
    k1 = k1 or (max(int(r.get("scenario") or 0) for r in rows) + 1)
    m = -(-k1 // 3)
    order, groups = {}, {}
    for r in rows:
        st = str(r.get("stage"))
        if st == "start":
            continue
        key = (st, int(float(r.get("iteration") or 0)))
        order.setdefault(key, len(order))
        s = _settings(r)
        g = groups.setdefault((order[key], json.dumps(s, sort_keys=True)), {"stage": st, "settings": s, "scores": {}})
        g["scores"][int(r.get("scenario") or 0)] = float(r["score"])
    cands = []
    for (step, _), g in groups.items():
        sc = list(g["scores"].values())
        exact = len(sc) >= k1
        cands.append(Candidate(step, g["stage"], g["settings"], np.array([converge.to_x(g["settings"])[k] for k in converge.KEYS]),
                               validate.worst_mean(sc, m), exact, len(sc)))
    if strategy == "unknown":
        strategy = "cma" if any(c.stage == "cma" for c in cands) else "pso"
    labels = [f"{st} {it}" for (st, it) in sorted(order, key=order.get)]
    return History(course, strategy, k1, sorted(cands, key=lambda c: c.step), start, labels, cma)


def pca(X: np.ndarray, k: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(centre, the k leading unit loadings as columns, the share of variance of each of all components)."""
    c = X.mean(axis=0)
    _, s, Vt = np.linalg.svd(X - c, full_matrices=False)
    var = s ** 2
    share = var / var.sum() if var.sum() > 0 else np.zeros_like(var)
    W = Vt[:k].T
    if W.shape[1] < k:                                               # fewer distinct candidates than components
        W = np.hstack([W, np.zeros((X.shape[1], k - W.shape[1]))])
    for j in range(W.shape[1]):                                      # a sign convention: the largest loading positive
        if W[np.argmax(np.abs(W[:, j])), j] < 0:
            W[:, j] *= -1
    return c, W, share


def ellipse(mean, sigma, C, c, W, nsig: float = 2.0, n: int = 80) -> np.ndarray:
    """The ``nsig``-σ ellipse of N(mean, σ²C) projected on the plane (centre c, loadings W): points (n, 2)."""
    P = W.T @ (sigma ** 2 * np.asarray(C)) @ W
    vals, vecs = np.linalg.eigh(P)
    a = np.linspace(0, 2 * np.pi, n)
    circle = np.stack([np.cos(a), np.sin(a)])
    return ((np.asarray(mean) - c) @ W)[None, :] + (vecs @ (nsig * np.sqrt(np.maximum(vals, 0))[:, None] * circle)).T


def _box(mean) -> np.ndarray:
    """CMA-ES's mean in the box: ``cma`` handles the bounds by a transformation of its internal coordinates, whose
    mean may lie outside [0, 1]; the candidates it produces are its image in the box (to first order, the clip)."""
    return np.clip(np.asarray(mean, dtype=float), 0.0, 1.0)


def _score_name(h: History) -> str:
    return "robust score" if h.k1 > 1 else "lap score"


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False,
                         "font.size": 9})
    return plt


def _plane(ax, h: History, c, W, share, *, color_by: str, upto: int | None = None, cmap="viridis", fig=None):
    import matplotlib.colors as mcolors
    cands = [d for d in h.candidates if upto is None or d.step <= upto]
    if color_by == "score":                                           # the best drawn last, on top
        cands = sorted(cands, key=lambda d: -d.score)
    P = (np.array([d.x for d in cands]) - c) @ W if cands else np.zeros((0, 2))
    n = max(len(h.steps) - 1, 1)
    if color_by == "step":
        vals, norm, label = [d.step for d in cands], mcolors.Normalize(0, n), "generation / iteration"
    else:
        vals = [max(d.score, 1e-3) for d in cands]
        lo, hi = min(d.score for d in h.candidates), max(d.score for d in h.candidates)
        norm, label, cmap = mcolors.LogNorm(max(lo, 1e-3), max(hi, lo * 1.01)), f"{_score_name(h)} (log)", "magma_r"
    ex = np.array([d.exact for d in cands], dtype=bool)
    sc = None
    if ex.any():
        sc = ax.scatter(P[ex, 0], P[ex, 1], c=np.array(vals)[ex], cmap=cmap, norm=norm, s=22, edgecolor="k", linewidth=0.3, zorder=3)
    if (~ex).any():
        cm = __import__("matplotlib").colormaps[cmap]
        ax.scatter(P[~ex, 0], P[~ex, 1], facecolor="none", edgecolor=[cm(norm(v)) for v in np.array(vals)[~ex]], s=22, linewidth=0.9,
                   zorder=2, label="dropped by racing (lower bound)")
    if color_by == "step" and h.cma:
        cm = __import__("matplotlib").colormaps[cmap]
        gens = [g for g in h.cma if upto is None or g["gen"] <= upto]
        for g in gens:
            E = ellipse(_box(g["mean"]), g["sigma"], g["C"], c, W)
            ax.plot(E[:, 0], E[:, 1], color=cm(norm(g["gen"])), lw=0.9, alpha=0.8, zorder=1)
        if gens:
            M = (np.array([_box(g["mean"]) for g in gens]) - c) @ W
            ax.plot(M[:, 0], M[:, 1], "-", color="0.25", lw=1.0, zorder=4, label="CMA-ES mean")
            ax.plot(M[-1, 0], M[-1, 1], "o", color="0.25", ms=4, zorder=4)
    ex_c = [d for d in cands if d.exact]
    if ex_c:                                                          # the best-so-far path: the incumbent after every step
        path, b, inc = [], math.inf, None
        for k in sorted({d.step for d in ex_c}):
            top = min((d for d in ex_c if d.step == k), key=lambda d: d.score)
            if top.score < b:
                b, inc = top.score, top
                path.append(inc.x)
        Q = (np.array(path) - c) @ W
        ax.plot(Q[:, 0], Q[:, 1], "--", color="#d62828", lw=1.0, zorder=5, label="best so far")
        ax.plot(Q[-1, 0], Q[-1, 1], "*", color="#d62828", ms=14, markeredgecolor="k", zorder=6, label=f"best {b:.2f}")
    if h.start:
        s0 = (np.array([converge.to_x(h.start)[k] for k in converge.KEYS]) - c) @ W
        ax.plot(s0[0], s0[1], "X", color="k", ms=8, zorder=6, label="start (stored settings)")
    ax.set_xlabel(f"PC1 ({share[0]:.0%} of the variance)")
    ax.set_ylabel(f"PC2 ({share[1]:.0%})" if len(share) > 1 else "PC2")
    if fig is not None and sc is not None:
        fig.colorbar(sc, ax=ax, label=label, shrink=0.85)
    return sc


def figures(h: History, out: Path, *, animate: bool = False, dpi: int = 110) -> list[Path]:
    """Write the course's figures into ``out``; returns their paths."""
    plt = _style()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    X = h.X
    c, W, share = pca(X)
    title = f"{h.course}: {h.strategy.upper()} training history — {len(h.candidates)} candidates, {len(h.steps)} steps, {h.k1} lap(s) per candidate"
    written = []

    # ---- figure 1: loss history, the plane by generation and by score, the loadings
    fig, axs = plt.subplots(2, 2, figsize=(13, 9.5))
    ax = axs[0, 0]
    S = np.array([d.step for d in h.candidates])
    F = np.array([d.score for d in h.candidates])
    ex = np.array([d.exact for d in h.candidates], dtype=bool)
    jit = (np.random.default_rng(0).random(len(S)) - 0.5) * 0.3
    ax.scatter(S[ex] + jit[ex], F[ex], s=12, color="#457b9d", alpha=0.6, label="candidate")
    if (~ex).any():
        ax.scatter(S[~ex] + jit[~ex], F[~ex], s=12, facecolor="none", edgecolor="#457b9d", alpha=0.6, label="dropped by racing (lower bound)")
    per = [min((d.score for d in h.candidates if d.step == k and d.exact), default=np.nan) for k in range(len(h.steps))]
    med = [np.median([d.score for d in h.candidates if d.step == k]) if any(d.step == k for d in h.candidates) else np.nan for k in range(len(h.steps))]
    bsf = [b for _, b in h.best_so_far]
    ax.plot(range(len(h.steps)), med, color="0.5", lw=1, label="median of the step")
    ax.plot(range(len(h.steps)), per, "o-", color="#1d3557", ms=3, lw=1, label="best of the step")
    ax.step(range(len(h.steps)), bsf, where="post", color="#d62828", lw=2, label=f"best so far ({bsf[-1]:.2f})")
    ax.set_yscale("log")
    ax.set_xlabel("generation / iteration")
    ax.set_ylabel(f"{_score_name(h)} (log)")
    ax.text(0.01, 0.01, "a lap: s + 2 per collision; a DNF: 1000 − 20 × zones passed" + (f"; robust: the mean of the worst {-(-h.k1 // 3)} of {h.k1} laps" if h.k1 > 1 else ""),
            transform=ax.transAxes, fontsize=7, color="0.35")
    ax.set_title("loss history")
    ax.legend(fontsize=7, loc="upper right")
    if len(h.steps) <= 30:
        ax.set_xticks(range(len(h.steps)), h.steps, rotation=60, fontsize=6)
    ax = axs[0, 1]
    _plane(ax, h, c, W, share, color_by="step", fig=fig)
    ax.set_title("search plane (PCA) by generation" + (" — CMA-ES 2σ ellipses" if h.cma else ""))
    ax.legend(fontsize=7, loc="best")
    ax = axs[1, 0]
    _plane(ax, h, c, W, share, color_by="score", fig=fig)
    ax.set_title("search plane (PCA) by score")
    ax = axs[1, 1]
    idx = np.arange(len(converge.KEYS))
    ax.barh(idx + 0.2, W[:, 0], height=0.4, color="#457b9d", label=f"PC1 ({share[0]:.0%})")
    ax.barh(idx - 0.2, W[:, 1], height=0.4, color="#e9c46a", label=f"PC2 ({share[1]:.0%})" if len(share) > 1 else "PC2")
    ax.set_yticks(idx, converge.KEYS)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("loading (normalised settings)")
    ax.set_title(f"what the plane's axes are — PC1 + PC2 hold {share[:2].sum():.0%} of the variance")
    ax.legend(fontsize=7)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    f1 = out / f"{h.course}_train_history.png"
    fig.savefig(f1, dpi=dpi)
    plt.close(fig)
    written.append(f1)

    # ---- figure 2: every setting in its own units
    fig, axs = plt.subplots(2, 3, figsize=(13, 6.5), sharex=True)
    bs_steps, bs_vals = [], {k: [] for k in converge.KEYS}
    for k in range(len(h.steps)):
        cand = min([d for d in h.candidates if d.step <= k and d.exact], key=lambda d: d.score, default=None)
        if cand is not None:
            bs_steps.append(k)
            for q in converge.KEYS:
                bs_vals[q].append(converge.to_settings(converge.to_x(cand.settings))[q] if cand.settings[q] is not None else converge.BOUNDS[q][1])
    for ax, q in zip(axs.flat, converge.KEYS):
        lo, hi = converge.BOUNDS[q]
        vals = np.array([d.settings[q] if d.settings[q] is not None else hi for d in h.candidates], dtype=float)
        ax.scatter(S + jit, vals, s=9, c=np.log10(np.maximum(F, 1e-3)), cmap="magma_r", alpha=0.7)
        ax.step(bs_steps, bs_vals[q], where="post", color="#d62828", lw=1.8, label="best so far")
        if h.cma:
            ax.plot([g["gen"] for g in h.cma], [lo + _box(g["mean"])[converge.KEYS.index(q)] * (hi - lo) for g in h.cma], color="0.25", lw=1, label="CMA-ES mean")
        ax.set_ylim(lo - 0.03 * (hi - lo), hi + 0.03 * (hi - lo))
        ax.set_title(f"{q}{' [' + UNITS[q] + ']' if UNITS[q] else ''}" + ("  (top = no cap)" if q == "v_cap" else ""), fontsize=9)
    axs[0, 0].legend(fontsize=7)
    for ax in axs[1]:
        ax.set_xlabel("generation / iteration")
    fig.suptitle(f"{h.course}: the settings per step (colour: score, lighter = better) — per setting, for reading the values; the plane shows the search", fontsize=10)
    fig.tight_layout()
    f2 = out / f"{h.course}_params.png"
    fig.savefig(f2, dpi=dpi)
    plt.close(fig)
    written.append(f2)

    # ---- the animation: the plane generation by generation
    if animate and len(h.steps) > 1:
        from matplotlib.animation import FuncAnimation, PillowWriter
        fig, ax = plt.subplots(figsize=(7, 6))
        P = (X - c) @ W
        if h.cma:                                                     # the frame holds the ellipses too
            P = np.vstack([P, *[ellipse(_box(g["mean"]), g["sigma"], g["C"], c, W) for g in h.cma]])
        pad = 0.08 * (np.ptp(P, axis=0) + 1e-9)
        lims = (P[:, 0].min() - pad[0], P[:, 0].max() + pad[0], P[:, 1].min() - pad[1], P[:, 1].max() + pad[1])

        def frame(k):
            ax.clear()
            _plane(ax, h, c, W, share, color_by="step", upto=k)
            ax.set_xlim(lims[0], lims[1]); ax.set_ylim(lims[2], lims[3])
            ax.set_title(f"{h.course} — {h.steps[k]}: best so far {h.best_so_far[k][1]:.2f}")
            ax.legend(fontsize=7, loc="upper right")
        anim = FuncAnimation(fig, frame, frames=len(h.steps))
        f3 = out / f"{h.course}_search.gif"
        anim.save(str(f3), writer=PillowWriter(fps=3))
        plt.close(fig)
        written.append(f3)
    return written


def summary(h: History) -> str:
    c, W, share = pca(h.X)
    top = lambda j: ", ".join(f"{converge.KEYS[i]} {W[i, j]:+.2f}" for i in np.argsort(-np.abs(W[:, j]))[:3])
    b = min((d for d in h.candidates if d.exact), key=lambda d: d.score, default=None)
    return (f"{h.course}: {h.strategy}, {len(h.candidates)} candidates in {len(h.steps)} steps ({sum(d.laps for d in h.candidates)} laps, "
            f"{sum(not d.exact for d in h.candidates)} dropped by racing); best {b.score if b else float('nan'):.2f}; "
            f"PC1 {share[0]:.0%} ({top(0)}), PC2 {share[1] if len(share) > 1 else 0:.0%} ({top(1)})")
