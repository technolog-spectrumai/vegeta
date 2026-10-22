"""Statistics for paired locomotion trials — protocol ``docs/myropod_stability.md`` §7 and §9.3.

Pure numpy/scipy. The trial (not a frame) is the independent observation; configurations are compared
on PAIRED seeds (both see the identical terrain and initial phase), so every comparison here is paired:

* :func:`wilson_ci` — success rate with a Wilson score interval;
* :func:`mcnemar_exact` — exact two-sided McNemar test on the discordant pairs;
* :func:`paired_difference_ci` — continuous metrics: mean paired difference, percentile bootstrap
  (10 000 resamples of the pairs, fixed seed);
* :func:`success_difference_ci` — paired difference of success rates (Agresti–Min adjusted Wald interval;
  Newcombe's hybrid score interval or the paired bootstrap on request);
* :func:`difference_of_differences_ci` — the body × controller synergy
  ``(a1 − a0) − (b1 − b0)`` per seed with a paired bootstrap over seeds;
* :func:`logistic_fit` — maximum-likelihood logistic regression of success on factors with interactions,
  Wald and bootstrap intervals;
* :func:`recovery_curve` — recovery rate per impulse (Wilson) and the logistic impulse at 50 % recovery
  (J50) with a bootstrap interval;
* :func:`cell_rates` — success rate and Wilson interval per cell of any factors (heatmaps);
* :func:`onset_speed` — the speed at which a metric's mean crosses a threshold (undulation onset, §9.3).

Missing values (nan) drop the whole pair (or seed) and the count used is always returned.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

__all__ = ["LogisticFit", "cell_rates", "difference_of_differences_ci", "logistic_fit", "mcnemar_exact",
           "onset_speed", "paired_difference_ci", "rate_ci", "recovery_curve", "success_difference_ci", "wilson_ci"]

_NAN = math.nan


def _z(conf: float) -> float:
    from scipy.stats import norm

    return float(norm.ppf(0.5 + conf / 2.0))


def _bool_pairs(x, y) -> tuple[np.ndarray, np.ndarray]:
    """Paired outcomes as bool arrays, dropping pairs where either is missing (None/nan)."""
    xa = np.array([_NAN if v is None else float(v) for v in np.asarray(x, dtype=object).reshape(-1)])
    ya = np.array([_NAN if v is None else float(v) for v in np.asarray(y, dtype=object).reshape(-1)])
    if xa.shape != ya.shape:
        raise ValueError("paired arrays must have the same length")
    keep = np.isfinite(xa) & np.isfinite(ya)
    return xa[keep] > 0.5, ya[keep] > 0.5


def _float_pairs(*arrays) -> tuple[list[np.ndarray], int]:
    arrs = [np.asarray(a, dtype=float).reshape(-1) for a in arrays]
    if len({a.size for a in arrs}) != 1:
        raise ValueError("paired arrays must have the same length")
    keep = np.all([np.isfinite(a) for a in arrs], axis=0)
    return [a[keep] for a in arrs], int((~keep).sum())


def _bootstrap(d: np.ndarray, stat: Callable, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """Statistic of ``n_boot`` resamples (with replacement) of the rows of ``d`` (vectorised, chunked)."""
    n = d.shape[0]
    out = np.empty(n_boot)
    chunk = max(1, int(4_000_000 // max(n, 1)))
    for s in range(0, n_boot, chunk):
        m = min(chunk, n_boot - s)
        idx = rng.integers(0, n, size=(m, n))
        out[s:s + m] = stat(d[idx], axis=1)
    return out


def _percentile_ci(samples, conf: float) -> tuple[float, float]:
    s = np.asarray(samples, dtype=float)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return _NAN, _NAN
    a = 100 * (1 - conf) / 2
    lo, hi = np.percentile(s, [a, 100 - a])
    return float(lo), float(hi)


# --------------------------------------------------------------------------------------------- rates
def wilson_ci(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for k successes out of n (protocol §7). ``(nan, nan)`` for n = 0."""
    if n <= 0:
        return _NAN, _NAN
    z = _z(conf)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def rate_ci(successes, conf: float = 0.95) -> dict:
    """``{k, n, rate, lo, hi}`` of a vector of outcomes (missing values dropped) with the Wilson interval."""
    a = np.array([_NAN if v is None else float(v) for v in np.asarray(successes, dtype=object).reshape(-1)])
    a = a[np.isfinite(a)] > 0.5
    k, n = int(a.sum()), int(a.size)
    lo, hi = wilson_ci(k, n, conf)
    return {"k": k, "n": n, "rate": k / n if n else _NAN, "lo": lo, "hi": hi}


# -------------------------------------------------------------------------------------------- McNemar
def mcnemar_exact(x, y=None) -> dict:
    """Exact two-sided McNemar test (protocol §7).

    Inputs: two paired outcome vectors ``x``, ``y`` (bool; missing pairs dropped); or the 2×2 table
    ``[[both, x only], [y only, neither]]`` (rows x success/failure, columns y success/failure) as ``x``
    alone; or the two discordant counts ``x = n10`` (x succeeds, y fails) and ``y = n01``.
    Under H0 the discordant pairs split 50/50: ``p = min(1, 2 · P(Bin(n10 + n01, ½) ≤ min(n10, n01)))``.
    """
    from scipy.stats import binom

    n = None
    if y is None:
        tab = np.asarray(x, dtype=float)
        if tab.shape != (2, 2):
            raise ValueError("a single argument must be the 2x2 table")
        n10, n01, n = int(tab[0, 1]), int(tab[1, 0]), int(tab.sum())
    elif np.ndim(x) == 0 and np.ndim(y) == 0:
        n10, n01 = int(x), int(y)
    else:
        xb, yb = _bool_pairs(x, y)
        n10, n01, n = int(np.sum(xb & ~yb)), int(np.sum(~xb & yb)), int(xb.size)
    m = n10 + n01
    p = 1.0 if m == 0 else min(1.0, 2.0 * float(binom.cdf(min(n10, n01), m, 0.5)))
    return {"p_value": p, "n10": n10, "n01": n01, "n_discordant": m, "n": n}


# --------------------------------------------------------------------------------- paired differences
def paired_difference_ci(x, y, *, n_boot: int = 10_000, conf: float = 0.95, seed: int = 0,
                         statistic: str = "mean") -> dict:
    """Paired difference ``x − y`` of a continuous metric with a percentile bootstrap interval (§7).

    The pairs (seeds) are resampled ``n_boot`` (10 000) times with a fixed ``seed``; ``statistic`` is
    ``'mean'`` (default) or ``'median'`` of the per-pair differences. Pairs with a missing value are
    dropped (``n_dropped``).
    """
    (xa, ya), dropped = _float_pairs(x, y)
    f = {"mean": np.mean, "median": np.median}[statistic]
    d = xa - ya
    if d.size == 0:
        return {"estimate": _NAN, "lo": _NAN, "hi": _NAN, "n": 0, "n_dropped": dropped,
                "mean_x": _NAN, "mean_y": _NAN}
    boots = _bootstrap(d, f, n_boot, np.random.default_rng(seed))
    lo, hi = _percentile_ci(boots, conf)
    return {"estimate": float(f(d)), "lo": lo, "hi": hi, "n": int(d.size), "n_dropped": dropped,
            "mean_x": float(xa.mean()), "mean_y": float(ya.mean()), "statistic": statistic,
            "n_boot": n_boot, "conf": conf}


def success_difference_ci(x, y, *, conf: float = 0.95, method: str = "agresti_min", n_boot: int = 10_000,
                          seed: int = 0) -> dict:
    """Paired difference of success rates ``rate_x − rate_y`` with a confidence interval.

    The estimate is always the plain ``(n10 − n01) / n``. Interval ``method``:

    * ``'agresti_min'`` (default): Agresti & Min (2005) adjusted Wald interval for paired proportions —
      add ½ to each of the four cells, ``d̃ = (n10 − n01)/(n + 2)``,
      ``se = √((n10' + n01') − (n10' − n01')²/(n + 2)) / (n + 2)``; never zero-width, good coverage at the
      study's 30 pairs;
    * ``'newcombe'``: Newcombe's (1998) hybrid score interval (his method 10: Wilson limits combined through
      the phi correlation of the pairs) — note it collapses to zero width when there is no discordant pair
      and both rates are ½;
    * ``'bootstrap'``: percentile bootstrap of the pairs (``n_boot``, fixed ``seed``).

    Also returns the exact McNemar p-value (protocol §7's test of the difference).
    """
    xb, yb = _bool_pairs(x, y)
    n = int(xb.size)
    a, b = int(np.sum(xb & yb)), int(np.sum(xb & ~yb))
    c, d = int(np.sum(~xb & yb)), int(np.sum(~xb & ~yb))
    out: dict[str, Any] = {"n": n, "both": a, "n10": b, "n01": c, "neither": d,
                           "p_mcnemar": mcnemar_exact(b, c)["p_value"], "method": method}
    if n == 0:
        return {**out, "rate_x": _NAN, "rate_y": _NAN, "difference": _NAN, "lo": _NAN, "hi": _NAN}
    p1, p2 = (a + b) / n, (a + c) / n
    diff = p1 - p2
    if method == "agresti_min":
        z = _z(conf)
        b2, c2, n2 = b + 0.5, c + 0.5, n + 2
        dt = (b2 - c2) / n2
        se = math.sqrt(max((b2 + c2) - (b2 - c2) ** 2 / n2, 0.0)) / n2
        lo, hi = max(-1.0, dt - z * se), min(1.0, dt + z * se)
    elif method == "newcombe":
        l1, u1 = wilson_ci(a + b, n, conf)
        l2, u2 = wilson_ci(a + c, n, conf)
        den = (a + b) * (c + d) * (a + c) * (b + d)
        phi = (a * d - b * c) / math.sqrt(den) if den > 0 else 0.0
        dl = math.sqrt(max((p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2, 0.0))
        du = math.sqrt(max((u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2, 0.0))
        lo, hi = max(-1.0, diff - dl), min(1.0, diff + du)
    elif method == "bootstrap":
        dd = xb.astype(float) - yb.astype(float)
        lo, hi = _percentile_ci(_bootstrap(dd, np.mean, n_boot, np.random.default_rng(seed)), conf)
    else:
        raise ValueError("method must be 'agresti_min', 'newcombe' or 'bootstrap'")
    return {**out, "rate_x": p1, "rate_y": p2, "difference": diff, "lo": lo, "hi": hi}


def difference_of_differences_ci(a1, a0, b1, b0, *, n_boot: int = 10_000, conf: float = 0.95,
                                 seed: int = 0) -> dict:
    """Interaction as a difference of paired differences, paired bootstrap over seeds (protocol §9.3).

    All four arrays are indexed by the same seed. Per seed ``dd = (a1 − a0) − (b1 − b0)``; the estimate is
    the mean of dd (for success: a difference of rate differences), the interval a percentile bootstrap of
    the seeds. For the synergy of §9.3 pass ``a1 = flexible·adaptive``, ``a0 = flexible·fixed``,
    ``b1 = locked·adaptive``, ``b0 = locked·fixed``. Booleans are treated as 0/1.
    """
    (A1, A0, B1, B0), dropped = _float_pairs(a1, a0, b1, b0)
    dd = (A1 - A0) - (B1 - B0)
    if dd.size == 0:
        return {"estimate": _NAN, "lo": _NAN, "hi": _NAN, "n": 0, "n_dropped": dropped,
                "effect_a": _NAN, "effect_b": _NAN}
    lo, hi = _percentile_ci(_bootstrap(dd, np.mean, n_boot, np.random.default_rng(seed)), conf)
    return {"estimate": float(dd.mean()), "lo": lo, "hi": hi, "n": int(dd.size), "n_dropped": dropped,
            "effect_a": float(np.mean(A1 - A0)), "effect_b": float(np.mean(B1 - B0)), "n_boot": n_boot, "conf": conf}


# ------------------------------------------------------------------------------- logistic regression
@dataclass
class _Design:
    """Treatment-coded design: intercept, numeric factors as-is, categorical factors as indicators of
    every level but the reference (the first of ``levels``), interactions as products of the columns."""

    factors: list[str]
    kinds: dict[str, str]
    levels: dict[str, list]
    interactions: list[tuple[str, ...]]

    def blocks(self, data: Mapping[str, Sequence]) -> tuple[dict[str, np.ndarray], dict[str, list[str]]]:
        cols, names = {}, {}
        for f in self.factors:
            v = np.asarray(data[f]).reshape(-1)
            if self.kinds[f] == "numeric":
                cols[f], names[f] = v.astype(float)[:, None], [f]
            else:
                lv = self.levels[f]
                unknown = set(v.tolist()) - set(lv)
                if unknown:
                    raise ValueError(f"factor {f!r}: unknown levels {sorted(map(str, unknown))}")
                cols[f] = np.stack([(v == L).astype(float) for L in lv[1:]], axis=1) if len(lv) > 1 \
                    else np.zeros((v.size, 0))
                names[f] = [f"{f}[{L}]" for L in lv[1:]]
        return cols, names

    def matrix(self, data: Mapping[str, Sequence], n: int | None = None) -> tuple[np.ndarray, list[str]]:
        cols, names = self.blocks(data)
        if cols:
            n = next(iter(cols.values())).shape[0]
        X, nm = [np.ones((n, 1))], ["intercept"]
        for f in self.factors:
            X.append(cols[f])
            nm += names[f]
        for inter in self.interactions:
            for combo in itertools.product(*[range(cols[f].shape[1]) for f in inter]):
                X.append(np.prod([cols[f][:, i] for f, i in zip(inter, combo)], axis=0)[:, None])
                nm.append(":".join(names[f][i] for f, i in zip(inter, combo)))
        return np.hstack(X), nm


def _nll(beta, X, y):
    eta = X @ beta
    return float(np.sum(np.logaddexp(0.0, eta) - y * eta))


def _grad(beta, X, y):
    from scipy.special import expit

    return X.T @ (expit(X @ beta) - y)


def _hess(beta, X, y):
    from scipy.special import expit

    p = expit(X @ beta)
    return X.T @ (X * (p * (1 - p))[:, None])


def _fit(X, y, beta0=None) -> tuple[np.ndarray, bool]:
    """Maximum-likelihood logistic coefficients (scipy trust-region Newton; BFGS fallback)."""
    from scipy.optimize import minimize

    b0 = np.zeros(X.shape[1]) if beta0 is None else np.asarray(beta0, dtype=float)
    res = minimize(_nll, b0, args=(X, y), jac=_grad, hess=_hess, method="trust-exact",
                   options={"gtol": 1e-9, "maxiter": 500})
    if not res.success:
        res2 = minimize(_nll, res.x, args=(X, y), jac=_grad, method="BFGS", options={"gtol": 1e-8, "maxiter": 2000})
        if res2.fun <= res.fun:
            res = res2
    g = np.max(np.abs(_grad(res.x, X, y))) if X.size else 0.0
    return res.x, bool(res.success or g < 1e-6)


def _separated(X, y) -> bool:
    """Exact test for (quasi-)complete separation (Albert & Anderson 1984), by linear programming.

    The maximum-likelihood estimate does not exist iff some β ≠ 0 has ``s_i x_iᵀβ ≥ 0`` for every trial
    (``s_i = +1`` for a success, −1 for a failure) with at least one strict: maximise ``Σ s_i x_iᵀβ`` over
    ``|β_j| ≤ 1`` under those constraints; a positive optimum means separation.
    """
    from scipy.optimize import linprog

    if X.size == 0:
        return False
    A = np.where(np.asarray(y) > 0.5, 1.0, -1.0)[:, None] * X
    res = linprog(-A.sum(axis=0), A_ub=-A, b_ub=np.zeros(A.shape[0]), bounds=[(-1.0, 1.0)] * A.shape[1],
                  method="highs")
    return bool(res.status == 0 and -res.fun > 1e-9 * max(1.0, float(np.abs(A).sum())))


@dataclass
class LogisticFit:
    """Result of :func:`logistic_fit`: ``logit P(y=1) = X β`` (treatment coding, see ``names``)."""

    names: list[str]
    coef: np.ndarray
    se: np.ndarray
    wald_lo: np.ndarray
    wald_hi: np.ndarray
    p_wald: np.ndarray
    boot_lo: np.ndarray
    boot_hi: np.ndarray
    loglik: float
    n: int
    converged: bool
    separation: bool
    conf: float
    n_boot: int
    boot_failures: int = 0
    messages: list[str] = field(default_factory=list)
    boot_samples: np.ndarray | None = field(default=None, repr=False)
    design: _Design | None = field(default=None, repr=False)

    def predict(self, factors: Mapping[str, Sequence]) -> np.ndarray:
        """Predicted probabilities for new factor values (same factors and levels as the fit)."""
        from scipy.special import expit

        X, _ = self.design.matrix(factors)
        return expit(X @ self.coef)

    def coefficients(self) -> dict[str, dict]:
        return {n: {"coef": float(self.coef[i]), "se": float(self.se[i]), "wald_lo": float(self.wald_lo[i]),
                    "wald_hi": float(self.wald_hi[i]), "p_wald": float(self.p_wald[i]),
                    "boot_lo": float(self.boot_lo[i]), "boot_hi": float(self.boot_hi[i])}
                for i, n in enumerate(self.names)}

    def table(self) -> list[dict]:
        """One row per coefficient (pandas-free; ``pandas.DataFrame(fit.table())`` for a frame)."""
        return [{"term": n, **v} for n, v in self.coefficients().items()]

    def to_dict(self) -> dict:
        return {"coefficients": self.coefficients(), "loglik": self.loglik, "n": self.n, "converged": self.converged,
                "separation": self.separation, "conf": self.conf, "n_boot": self.n_boot,
                "boot_failures": self.boot_failures, "messages": list(self.messages)}


def _kind_of(v: np.ndarray) -> str:
    return "categorical" if v.dtype.kind in "OUSb" else "numeric"


def logistic_fit(y, factors: Mapping[str, Sequence], interactions: Sequence[Sequence[str]] = (), *,
                 levels: Mapping[str, Sequence] | None = None, n_boot: int = 1000, seed: int = 0,
                 conf: float = 0.95, groups: Sequence | None = None) -> LogisticFit:
    """Maximum-likelihood logistic regression of a binary outcome on factors (protocol §9.3).

    ``y``: outcomes (bool/0-1). ``factors``: name -> values per trial; strings/bools are categorical
    (treatment coding against the first level of ``levels[name]``, default the sorted unique values —
    pass e.g. ``levels={'body': ['locked', 'flexible', 'flexible+roll'], 'controller': ['fixed',
    'adaptive']}``), numbers are numeric. ``interactions``: tuples of factor names, e.g.
    ``[('body', 'controller')]``. Fitted with scipy (trust-region Newton on the exact log-likelihood).

    Intervals: Wald (``coef ± z·se``, se from the inverse observed information) and percentile bootstrap
    (``n_boot`` refits on resampled trials, fixed ``seed``; with ``groups`` — e.g. the seed — whole
    groups are resampled, which keeps the paired structure). Complete or quasi-complete separation (the
    MLE does not exist; some cell all-success or all-failure) is detected and flagged in ``separation``
    and ``messages`` (exact linear-programming test, :func:`_separated`): then the coefficients and Wald
    intervals are not meaningful.
    """
    from scipy.stats import norm

    yv = np.array([_NAN if v is None else float(v) for v in np.asarray(y, dtype=object).reshape(-1)])
    names = list(factors)
    vals = {f: np.asarray(factors[f]).reshape(-1) for f in names}
    for f, v in vals.items():
        if v.size != yv.size:
            raise ValueError(f"factor {f!r} has {v.size} values for {yv.size} outcomes")
    keep = np.isfinite(yv)
    for f, v in vals.items():
        if _kind_of(v) == "numeric":
            keep &= np.isfinite(v.astype(float))
    yv = yv[keep] > 0.5
    vals = {f: v[keep] for f, v in vals.items()}
    kinds = {f: _kind_of(v) for f, v in vals.items()}
    lv = {}
    for f in names:
        if kinds[f] == "categorical":
            lv[f] = list(levels[f]) if levels and f in levels else sorted(set(vals[f].tolist()), key=str)
    inter = [tuple(i) for i in interactions]
    for i in inter:
        for f in i:
            if f not in vals:
                raise ValueError(f"interaction factor {f!r} is not a factor")
    design = _Design(names, kinds, lv, inter)
    X, cols = design.matrix(vals, n=int(yv.size))
    yf = yv.astype(float)
    beta, conv = _fit(X, yf)
    H = _hess(beta, X, yf)
    try:
        cov = np.linalg.inv(H)
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        se = np.full(beta.size, math.inf)
    z = _z(conf)
    msgs = []
    separation = _separated(X, yf)
    if separation:
        msgs.append("(quasi-)complete separation: the maximum-likelihood estimate does not exist, coefficients "
                    "diverge and the Wald intervals are not valid; report rates per cell instead")
    if not conv:
        msgs.append("the optimiser did not converge")
    with np.errstate(divide="ignore", invalid="ignore"):
        p_wald = 2 * norm.sf(np.abs(beta / se))
    boot_lo = np.full(beta.size, _NAN)
    boot_hi = np.full(beta.size, _NAN)
    samples = None
    fails = 0
    if n_boot:
        rng = np.random.default_rng(seed)
        if groups is not None:
            g = np.asarray(groups).reshape(-1)[keep]
            ug = np.unique(g)
            members = [np.flatnonzero(g == u) for u in ug]
        samples = np.full((n_boot, beta.size), _NAN)
        for b in range(n_boot):
            if groups is not None:
                pick = rng.integers(0, len(members), len(members))
                idx = np.concatenate([members[i] for i in pick])
            else:
                idx = rng.integers(0, yf.size, yf.size)
            bb, ok = _fit(X[idx], yf[idx], beta)
            if ok:
                samples[b] = bb
            else:
                fails += 1
        for i in range(beta.size):
            boot_lo[i], boot_hi[i] = _percentile_ci(samples[:, i], conf)
        if fails:
            msgs.append(f"{fails} of {n_boot} bootstrap refits did not converge and were dropped")
    return LogisticFit(names=cols, coef=beta, se=se, wald_lo=beta - z * se, wald_hi=beta + z * se, p_wald=p_wald,
                       boot_lo=boot_lo, boot_hi=boot_hi, loglik=-_nll(beta, X, yf), n=int(yf.size),
                       converged=conv, separation=separation, conf=conf, n_boot=n_boot, boot_failures=fails,
                       messages=msgs, boot_samples=samples, design=design)


# ---------------------------------------------------------------------------------- recovery curve
def recovery_curve(impulse, recovered, *, conf: float = 0.95, n_boot: int = 2000, seed: int = 0) -> dict:
    """Recovery probability vs push impulse (protocol §7 item 3).

    Per impulse level: ``n``, ``k`` recovered, ``rate`` with its Wilson interval. A logistic fit
    ``logit P(recovered) = b0 + b1 · J`` gives ``j50_Ns = −b0 / b1``, the impulse at 50 % recovery, with a
    percentile bootstrap interval (``n_boot`` resamples stratified by impulse level — the design's trials
    per level are fixed — fixed ``seed``) and a delta-method Wald interval. If every trial recovered (or
    none did) J50 is not identified (nan); separation is flagged.
    """
    J = np.asarray(impulse, dtype=float).reshape(-1)
    r = np.array([_NAN if v is None else float(v) for v in np.asarray(recovered, dtype=object).reshape(-1)])
    keep = np.isfinite(J) & np.isfinite(r)
    J, r = J[keep], r[keep] > 0.5
    rows = []
    for L in np.unique(J):
        sel = J == L
        k, n = int(r[sel].sum()), int(sel.sum())
        lo, hi = wilson_ci(k, n, conf)
        rows.append({"impulse_Ns": float(L), "n": n, "k": k, "rate": k / n, "lo": lo, "hi": hi})
    out: dict[str, Any] = {"levels": rows, "intercept": _NAN, "slope": _NAN, "j50_Ns": _NAN, "j50_lo": _NAN,
                           "j50_hi": _NAN, "j50_wald_lo": _NAN, "j50_wald_hi": _NAN, "separation": False,
                           "messages": [], "fit": None}
    if r.size == 0 or r.all() or not r.any() or np.unique(J).size < 2:
        out["messages"].append("J50 not identified: all trials recovered, none did, or a single impulse level")
        return out
    fit = logistic_fit(r, {"impulse": J}, n_boot=0)
    b0, b1 = fit.coef
    j50 = -b0 / b1 if b1 != 0 else _NAN
    out.update({"intercept": float(b0), "slope": float(b1), "j50_Ns": float(j50), "separation": fit.separation,
                "fit": fit})
    out["messages"] += fit.messages
    # delta method: Var(J50) = (Var b0 + J50² Var b1 + 2 J50 Cov(b0, b1)) / b1²
    X = np.column_stack([np.ones(J.size), J])
    try:
        cov = np.linalg.inv(_hess(fit.coef, X, r.astype(float)))
        var = (cov[0, 0] + j50 ** 2 * cov[1, 1] + 2 * j50 * cov[0, 1]) / b1 ** 2
        z = _z(conf)
        out["j50_wald_lo"], out["j50_wald_hi"] = j50 - z * math.sqrt(max(var, 0)), j50 + z * math.sqrt(max(var, 0))
    except np.linalg.LinAlgError:
        pass
    if n_boot:
        rng = np.random.default_rng(seed)
        members = [np.flatnonzero(J == L) for L in np.unique(J)]
        js = np.full(n_boot, _NAN)
        for b in range(n_boot):
            idx = np.concatenate([m[rng.integers(0, m.size, m.size)] for m in members])
            rb = r[idx]
            if rb.all() or not rb.any():
                continue
            bb, ok = _fit(X[idx], rb.astype(float), fit.coef)
            if ok and bb[1] != 0:
                js[b] = -bb[0] / bb[1]
        out["j50_lo"], out["j50_hi"] = _percentile_ci(js, conf)
        out["j50_boot_valid"] = int(np.isfinite(js).sum())
    return out


# ------------------------------------------------------------------------------------ cells and onset
def cell_rates(success, factors: Mapping[str, Sequence], conf: float = 0.95) -> list[dict]:
    """Success rate with its Wilson interval for every combination of factor values (protocol §7: one row per
    cell, e.g. the speed × difficulty heatmaps). ``factors``: name -> value per trial. Rows are sorted by the
    factor values; missing outcomes are dropped."""
    s = np.array([_NAN if v is None else float(v) for v in np.asarray(success, dtype=object).reshape(-1)])
    names = list(factors)
    cols = [np.asarray(factors[f], dtype=object).reshape(-1) for f in names]
    for f, c in zip(names, cols):
        if c.size != s.size:
            raise ValueError(f"factor {f!r} has {c.size} values for {s.size} outcomes")
    def order(key):          # numbers numerically, everything else as text; never compares mixed types
        return tuple((0, float(x), "") if isinstance(x, (int, float, np.number)) and not isinstance(x, bool)
                     else (1, 0.0, str(x)) for x in key)

    keys = sorted({tuple(c[i] for c in cols) for i in range(s.size)}, key=order)
    rows = []
    for key in keys:
        sel = np.ones(s.size, bool)
        for c, v in zip(cols, key):
            sel &= c == v
        rows.append({**dict(zip(names, key)), **rate_ci(s[sel], conf)})
    return rows


def onset_speed(speed, value, threshold: float = 5.0, *, n_boot: int = 2000, seed: int = 0,
                conf: float = 0.95) -> dict:
    """Onset of a metric with speed (protocol §9.3 item 8: the speed at which the RMS body-yaw angle exceeds
    5°), from trials at fixed speed levels.

    Per level: ``n``, ``mean``, ``sd`` and the fraction of trials above ``threshold``. ``onset_level`` = the
    lowest level whose mean exceeds the threshold; ``onset`` = the speed where the per-level mean crosses
    the threshold, interpolated linearly between that level and the one below (nan when the threshold is
    already exceeded at the lowest level — ``below_range`` — or never — ``above_range``). Interval: percentile
    bootstrap of ``onset`` resampling trials within each level (``n_boot``, fixed ``seed``); resamples
    without a crossing are counted in ``boot_no_crossing``. Missing values are dropped.
    """
    v = np.asarray(speed, dtype=float).reshape(-1)
    x = np.asarray(value, dtype=float).reshape(-1)
    keep = np.isfinite(v) & np.isfinite(x)
    v, x = v[keep], x[keep]
    levels = np.unique(v)
    members = [np.flatnonzero(v == L) for L in levels]

    def crossing(means):
        above = np.flatnonzero(means > threshold)
        if above.size == 0:
            return _NAN, None, "above_range"
        i = int(above[0])
        if i == 0:
            return _NAN, float(levels[0]), "below_range"
        m0, m1 = means[i - 1], means[i]
        return float(levels[i - 1] + (threshold - m0) / (m1 - m0) * (levels[i] - levels[i - 1])), float(levels[i]), ""

    means = np.array([x[m].mean() for m in members]) if levels.size else np.zeros(0)
    onset, onset_level, flag = crossing(means)
    rows = [{"speed": float(L), "n": int(m.size), "mean": float(x[m].mean()),
             "sd": float(x[m].std(ddof=1)) if m.size > 1 else _NAN, "fraction_above": float(np.mean(x[m] > threshold))}
            for L, m in zip(levels, members)]
    out = {"levels": rows, "threshold": threshold, "onset": onset, "onset_level": onset_level, "flag": flag,
           "lo": _NAN, "hi": _NAN, "boot_no_crossing": 0}
    if n_boot and levels.size >= 2:
        rng = np.random.default_rng(seed)
        ons = np.full(n_boot, _NAN)
        for b in range(n_boot):
            mb = np.array([x[m[rng.integers(0, m.size, m.size)]].mean() for m in members])
            ons[b] = crossing(mb)[0]
        out["lo"], out["hi"] = _percentile_ci(ons, conf)
        out["boot_no_crossing"] = int(np.sum(~np.isfinite(ons)))
    return out
