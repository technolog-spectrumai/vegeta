"""chiron.stats: intervals, paired tests and logistic fits on hand-checkable data (protocol §7, §9.3)."""
import math

import numpy as np
import pytest
from scipy.special import expit, logit

from vegeta.chiron import stats as S


def test_wilson_ci():
    assert S.wilson_ci(5, 10) == pytest.approx((0.2366, 0.7634), abs=1e-4)
    z2 = 1.959964 ** 2
    assert S.wilson_ci(0, 10) == pytest.approx((0.0, z2 / (10 + z2)), abs=1e-6)
    assert S.wilson_ci(10, 10) == pytest.approx((10 / (10 + z2), 1.0), abs=1e-6)
    assert all(math.isnan(v) for v in S.wilson_ci(0, 0))
    r = S.rate_ci([True, False, None, True, np.nan, 1])
    assert (r["k"], r["n"], r["rate"]) == (3, 4, 0.75) and r["lo"] < 0.75 < r["hi"]


def test_mcnemar_exact_known_table():
    # discordant pairs 1 vs 9: p = 2 * P(Bin(10, 1/2) <= 1) = 2 * 11/1024
    tab = [[10, 1], [9, 10]]
    assert S.mcnemar_exact(tab)["p_value"] == pytest.approx(22 / 1024)
    x = [1] * 10 + [1] + [0] * 9 + [0] * 10
    y = [1] * 10 + [0] + [1] * 9 + [0] * 10
    r = S.mcnemar_exact(x, y)
    assert r["p_value"] == pytest.approx(22 / 1024) and (r["n10"], r["n01"], r["n"]) == (1, 9, 30)
    assert S.mcnemar_exact(1, 9)["p_value"] == pytest.approx(22 / 1024)
    assert S.mcnemar_exact(0, 6)["p_value"] == pytest.approx(2 / 64)
    assert S.mcnemar_exact(5, 5)["p_value"] == 1.0 and S.mcnemar_exact(0, 0)["p_value"] == 1.0
    assert S.mcnemar_exact([1, 0, None], [0, 0, 1])["n"] == 2                     # missing pair dropped


def test_paired_difference_ci():
    x = np.linspace(1, 2, 30)
    r = S.paired_difference_ci(x, x - 0.1)
    assert (r["estimate"], r["lo"], r["hi"]) == pytest.approx((0.1, 0.1, 0.1)) and r["n"] == 30
    rng = np.random.default_rng(1)
    a = rng.normal(1.0, 1.0, 30)
    b = a - 0.5 + rng.normal(0, 0.3, 30)
    r1 = S.paired_difference_ci(a, b)
    r2 = S.paired_difference_ci(a, b)
    assert r1 == r2                                                               # fixed seed: reproducible
    d = a - b
    assert r1["estimate"] == pytest.approx(d.mean()) and r1["lo"] < d.mean() < r1["hi"]
    se = d.std(ddof=1) / math.sqrt(30)
    assert (r1["hi"] - r1["lo"]) == pytest.approx(2 * 1.96 * se, rel=0.2)
    assert S.paired_difference_ci(a, b, seed=7)["lo"] != r1["lo"]
    a[3] = np.nan
    assert S.paired_difference_ci(a, b)["n_dropped"] == 1
    assert S.paired_difference_ci(a, b, statistic="median")["estimate"] == pytest.approx(np.nanmedian(a - b))


def test_success_difference_ci_paired():
    # 10 both, 10 x only, 2 y only, 8 neither: 20/30 vs 12/30
    x = [1] * 10 + [1] * 10 + [0] * 2 + [0] * 8
    y = [1] * 10 + [0] * 10 + [1] * 2 + [0] * 8
    r = S.success_difference_ci(x, y)
    assert r["rate_x"] == pytest.approx(20 / 30) and r["rate_y"] == pytest.approx(12 / 30)
    assert r["difference"] == pytest.approx(8 / 30) and r["lo"] < 8 / 30 < r["hi"] and r["lo"] > 0
    assert r["p_mcnemar"] == pytest.approx(S.mcnemar_exact(10, 2)["p_value"])
    sw = S.success_difference_ci(y, x)                                            # swap: mirrored interval
    assert (sw["lo"], sw["hi"]) == pytest.approx((-r["hi"], -r["lo"]))
    # Agresti-Min (2005): +1/2 in every cell
    n, a, b, c, d = 30, 10, 10, 2, 8
    se = math.sqrt((b + c + 1) - (b - c) ** 2 / (n + 2)) / (n + 2)
    assert (r["lo"], r["hi"]) == pytest.approx(((b - c) / (n + 2) - 1.959964 * se, (b - c) / (n + 2) + 1.959964 * se))
    # Newcombe (1998) method 10, written out
    r = S.success_difference_ci(x, y, method="newcombe")
    sw = S.success_difference_ci(y, x, method="newcombe")
    assert (sw["lo"], sw["hi"]) == pytest.approx((-r["hi"], -r["lo"]))
    p1, p2 = (a + b) / n, (a + c) / n
    (l1, u1), (l2, u2) = S.wilson_ci(a + b, n), S.wilson_ci(a + c, n)
    phi = (a * d - b * c) / math.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    lo = p1 - p2 - math.sqrt((p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2)
    hi = p1 - p2 + math.sqrt((u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2)
    assert (r["lo"], r["hi"]) == pytest.approx((lo, hi))
    same = S.success_difference_ci([1, 1, 0, 0], [1, 1, 0, 0])                    # no discordant pair
    assert same["difference"] == 0 and same["lo"] < 0 < same["hi"]
    bt = S.success_difference_ci(x, y, method="bootstrap")
    assert bt["lo"] < 8 / 30 < bt["hi"] and bt == S.success_difference_ci(x, y, method="bootstrap")


def test_difference_of_differences_ci():
    rng = np.random.default_rng(0)
    lf = rng.normal(1, 0.2, 30)
    la = lf + 0.1
    ff = lf + 0.3
    fa = ff + 0.1 + 0.25                                                          # synergy 0.25 for every seed
    r = S.difference_of_differences_ci(fa, ff, la, lf)
    assert (r["estimate"], r["lo"], r["hi"]) == pytest.approx((0.25, 0.25, 0.25))
    assert r["effect_a"] == pytest.approx(0.35) and r["effect_b"] == pytest.approx(0.1)
    # success: flexible gains with the adaptive controller on half the seeds, locked never changes
    a1 = np.ones(30, bool)
    a0 = np.arange(30) % 2 == 0
    b1 = b0 = np.arange(30) % 3 == 0
    r = S.difference_of_differences_ci(a1, a0, b1, b0)
    assert r["estimate"] == pytest.approx(0.5) and r["lo"] < 0.5 < r["hi"] and r["n"] == 30
    noisy = fa + rng.normal(0, 0.1, 30)
    r = S.difference_of_differences_ci(noisy, ff, la, lf)
    dd = (noisy - ff) - (la - lf)
    assert r["estimate"] == pytest.approx(dd.mean()) and r["lo"] < dd.mean() < r["hi"]


def test_logistic_fit_single_factor_matches_log_odds_ratio():
    x = np.array(["a"] * 200 + ["b"] * 200)
    y = np.r_[np.ones(50), np.zeros(150), np.ones(150), np.zeros(50)]
    fit = S.logistic_fit(y, {"x": x}, n_boot=200, seed=1)
    assert fit.names == ["intercept", "x[b]"]
    assert fit.coef == pytest.approx([-math.log(3), math.log(9)], abs=1e-6)
    se_or = math.sqrt(1 / 50 + 1 / 150 + 1 / 150 + 1 / 50)                       # Woolf's SE of a log odds ratio
    assert fit.se == pytest.approx([math.sqrt(1 / 50 + 1 / 150), se_or], rel=1e-5)
    assert fit.wald_lo[1] == pytest.approx(math.log(9) - 1.959964 * se_or, rel=1e-5)
    assert fit.boot_lo[1] < math.log(9) < fit.boot_hi[1] and not fit.separation and fit.converged
    assert fit.predict({"x": ["a", "b"]}) == pytest.approx([0.25, 0.75])
    row = fit.table()[1]
    assert row["term"] == "x[b]" and row["p_wald"] < 1e-10


def test_logistic_fit_body_controller_interaction_recovers_cell_logits():
    rates = {("locked", "fixed"): 40, ("locked", "adaptive"): 50, ("flexible", "fixed"): 60,
             ("flexible", "adaptive"): 85, ("flexible+roll", "fixed"): 55, ("flexible+roll", "adaptive"): 70}
    body, ctrl, y, seed = [], [], [], []
    for (b, c), k in rates.items():
        body += [b] * 100
        ctrl += [c] * 100
        y += [1] * k + [0] * (100 - k)
        seed += list(range(100))
    L = {cell: logit(k / 100) for cell, k in rates.items()}
    fit = S.logistic_fit(y, {"body": body, "controller": ctrl}, [("body", "controller")],
                         levels={"body": ["locked", "flexible", "flexible+roll"], "controller": ["fixed", "adaptive"]},
                         n_boot=100, groups=seed)
    co = fit.coefficients()
    assert fit.names == ["intercept", "body[flexible]", "body[flexible+roll]", "controller[adaptive]",
                         "body[flexible]:controller[adaptive]", "body[flexible+roll]:controller[adaptive]"]
    assert co["intercept"]["coef"] == pytest.approx(L["locked", "fixed"], abs=1e-6)
    assert co["body[flexible]"]["coef"] == pytest.approx(L["flexible", "fixed"] - L["locked", "fixed"], abs=1e-6)
    assert co["controller[adaptive]"]["coef"] == pytest.approx(L["locked", "adaptive"] - L["locked", "fixed"], abs=1e-6)
    syn = (L["flexible", "adaptive"] - L["flexible", "fixed"]) - (L["locked", "adaptive"] - L["locked", "fixed"])
    assert co["body[flexible]:controller[adaptive]"]["coef"] == pytest.approx(syn, abs=1e-6)
    syn_r = (L["flexible+roll", "adaptive"] - L["flexible+roll", "fixed"]) - (L["locked", "adaptive"] - L["locked", "fixed"])
    assert co["body[flexible+roll]:controller[adaptive]"]["coef"] == pytest.approx(syn_r, abs=1e-6)
    for v in co.values():
        assert v["wald_lo"] < v["coef"] < v["wald_hi"] and v["boot_lo"] < v["coef"] < v["boot_hi"]
    assert fit.loglik == pytest.approx(sum(k * math.log(k / 100) + (100 - k) * math.log(1 - k / 100)
                                           for k in rates.values()), rel=1e-9)


def test_logistic_fit_flags_separation():
    x = np.array(["a"] * 20 + ["b"] * 20)
    y = np.r_[np.ones(20), np.r_[np.ones(10), np.zeros(10)]]                      # group a: all successes
    fit = S.logistic_fit(y, {"x": x}, n_boot=0)
    assert fit.separation and any("separation" in m for m in fit.messages)
    # complete separation on a numeric factor: recovered for J <= 3, never for J >= 4
    J = np.repeat([0.0, 1, 2, 3, 4, 6, 8], 10)
    assert S.logistic_fit(J <= 3, {"J": J}, n_boot=0).separation
    assert S.recovery_curve(J, J <= 3, n_boot=0)["separation"]
    # a steep but overlapping curve with a large coefficient (small units) is not separation
    v = np.repeat([0.10, 0.12, 0.14, 0.16], 50)
    ok = np.r_[np.ones(48), np.zeros(2), np.ones(35), np.zeros(15), np.ones(15), np.zeros(35), np.ones(2), np.zeros(48)]
    fit = S.logistic_fit(ok, {"v": v}, n_boot=0)
    assert not fit.separation and fit.coef[1] < -50


def test_recovery_curve_recovers_j50():
    levels = np.array([0, 1, 2, 3, 4, 6, 8], float)
    p = expit(4.0 - 1.0 * levels)                                                 # J50 = 4 N s
    J, rec = [], []
    for L, pi in zip(levels, p):
        k = int(round(1000 * pi))
        J += [L] * 1000
        rec += [1] * k + [0] * (1000 - k)
    r = S.recovery_curve(J, rec, n_boot=200)
    assert r["j50_Ns"] == pytest.approx(4.0, abs=0.03) and r["slope"] == pytest.approx(-1.0, abs=0.03)
    assert r["j50_lo"] < r["j50_Ns"] < r["j50_hi"] and r["j50_hi"] - r["j50_lo"] < 0.3
    assert r["j50_wald_lo"] < 4.0 < r["j50_wald_hi"]
    lv = {row["impulse_Ns"]: row for row in r["levels"]}
    assert lv[4.0]["k"] == 500 and lv[4.0]["rate"] == 0.5 and (lv[4.0]["lo"], lv[4.0]["hi"]) == S.wilson_ci(500, 1000)
    # the study's size: 30 seeds per level
    rng = np.random.default_rng(3)
    J = np.repeat(levels, 30)
    rec = rng.random(J.size) < expit(4.0 - J)
    r = S.recovery_curve(J, rec, n_boot=300)
    assert 2.5 < r["j50_Ns"] < 5.5 and r["j50_lo"] < r["j50_Ns"] < r["j50_hi"]
    allrec = S.recovery_curve([0, 1, 2], [1, 1, 1])
    assert math.isnan(allrec["j50_Ns"]) and allrec["messages"] and len(allrec["levels"]) == 3


def test_cell_rates():
    speed = [0.1] * 4 + [0.2] * 4 + [0.1] * 2
    body = ["locked"] * 8 + ["flexible"] * 2
    ok = [1, 1, 0, 1, 0, 0, 1, None, 1, 1]
    rows = S.cell_rates(ok, {"body": body, "speed": speed})
    cell = {(r["body"], r["speed"]): r for r in rows}
    assert len(rows) == 3 and (cell["locked", 0.1]["k"], cell["locked", 0.1]["n"]) == (3, 4)
    assert (cell["locked", 0.2]["k"], cell["locked", 0.2]["n"]) == (1, 3)              # None dropped
    assert (cell["flexible", 0.1]["lo"], cell["flexible", 0.1]["hi"]) == S.wilson_ci(2, 2)


def test_onset_speed():
    speeds = np.repeat([0.1, 0.2, 0.3, 0.4, 0.5], 10)
    rms = np.repeat([1.0, 2.0, 4.0, 8.0, 12.0], 10) + np.tile(np.linspace(-0.5, 0.5, 10), 5)
    r = S.onset_speed(speeds, rms, 5.0, n_boot=300)
    assert r["onset_level"] == 0.4 and r["onset"] == pytest.approx(0.3 + (5 - 4) / (8 - 4) * 0.1)
    assert r["lo"] <= r["onset"] <= r["hi"] and r["flag"] == ""
    assert [row["fraction_above"] for row in r["levels"]] == [0.0, 0.0, 0.0, 1.0, 1.0]
    assert S.onset_speed(speeds, rms + 10, 5.0, n_boot=0)["flag"] == "below_range"
    never = S.onset_speed(speeds, rms * 0.1, 5.0, n_boot=0)
    assert never["flag"] == "above_range" and math.isnan(never["onset"])
