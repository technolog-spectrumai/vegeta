"""stability_plots: the study's comparative plots on a synthetic run table with known answers (rates, Wilson and
bootstrap intervals, McNemar p, the §11.6 verdicts, the interaction, the undulation onset), every function returning a
Figure (Agg), and one real paired Cleopatra run through chiron's trial runner and the trace plot.

Run: cd /home/user/vegeta/notebooks/designs && python3 -m pytest -q tests/test_stability_plots.py
"""
import math

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

import stability_plots as sp  # noqa: E402
from vegeta.chiron import stats  # noqa: E402

SEEDS = range(20)
CTRL = {"fixed": ("designs/myropod_controller.py:fixed", math.nan),
        "adaptive": ("designs/myropod_controller.py:adaptive", 0.6)}

#: success rule per (terrain, level, controller, treatment): seed -> success
SUCCESS = {
    ("rough", 0.15, "fixed", "spring"): lambda s: s < 10,             # 10/20
    ("rough", 0.15, "fixed", "spring_damper"): lambda s: s < 18,      # 18/20: 8 discordant, all for damping -> helps
    ("rough", 0.15, "adaptive", "spring"): lambda s: s < 15,          # identical -> no clear difference
    ("rough", 0.15, "adaptive", "spring_damper"): lambda s: s < 15,
    ("rough", 0.25, "fixed", "spring"): lambda s: s < 12,             # 12/20
    ("rough", 0.25, "fixed", "spring_damper"): lambda s: s < 4,       # 4/20: 8 discordant against damping -> hurts
    ("rough", 0.25, "adaptive", "spring"): lambda s: s % 2 == 0,      # 10 vs 10 discordant -> no clear difference
    ("rough", 0.25, "adaptive", "spring_damper"): lambda s: s % 2 == 1,
    ("alt_bumps", 0.15, "fixed", "spring"): lambda s: s < 10,         # 5 discordant: interval excludes 0 but
    ("alt_bumps", 0.15, "fixed", "spring_damper"): lambda s: s < 15,  # McNemar p = 0.0625 -> no clear difference
    ("alt_bumps", 0.15, "adaptive", "spring"): lambda s: True,
    ("alt_bumps", 0.15, "adaptive", "spring_damper"): lambda s: True,
}
H = {"rough": "terrain.rms", "alt_bumps": "terrain.height"}


def run_row(terrain, level, controller, treat, seed, v=0.2, **extra):
    """One run_trials-style row (column names as chiron.experiments / chiron.metrics write them)."""
    factory, sigma = CTRL[controller]
    noise = 0.01 * (seed - 9.5) / 9.5                      # zero mean over the 20 seeds
    sign = 1.0 if seed % 2 == 0 else -1.0
    treatment = treat
    success = bool(SUCCESS.get((terrain, level, controller, treatment), lambda s: True)(seed))
    spring = treatment == "spring"
    row = {
        "trial_id": f"{terrain}-{level}-{controller}-{treatment}-{seed}-{v}", "name": "",
        "robot_factory": "designs/myropod_robot.py:cleopatra", "robot_kwargs.body_connection": treatment,
        "controller_factory": factory, "controller_kwargs.sigma": sigma,
        "terrain.kind": terrain, "terrain.level": level, "terrain.seed": seed, "seed": seed, "v_target": v,
        "course_m": 1.5, "lab_kwargs.timestep": 0.00025, "info.treatment": treatment, "status": "ok", "error": "",
        "success": success, "reason": "success" if success else "fall", "distance_m": 1.5 if success else 0.7,
        "robot": f"cleopatra {treatment}", "controller": controller, "treatment": treatment,
        "achieved_speed_m_s": 0.8 * v + noise,
        "cot_mech": 2.0 + noise * 10 - (0.0 if spring else 0.3),              # damping: exactly −0.3 -> helps
        "payload_roll_rate_rms_rad_s": 0.5 + noise + (0.0 if spring else 0.2),  # damping: exactly +0.2 -> hurts
        "payload_pitch_rate_rms_rad_s": 0.4 + noise, "payload_tilt_p95_deg": 3.0 + (level or 0.0) * 10 + noise,
        "payload_body": "head",
        "slip_per_m": 0.05 + (0.0 if spring else 0.01 * sign),                # mean 0 difference -> no clear
        "body_yaw_rms_max_deg": 1.0,
    }
    for body, scale in (("head", 1.0), ("segment 1", 1.0), ("segment 2", 2.0), ("segment 3", 1.5)):
        row[f"roll_rate_rms_rad_s@{body}"] = scale * (0.5 + noise)
        row[f"pitch_rate_rms_rad_s@{body}"] = scale * (0.4 + noise)
        row[f"tilt_p95_deg@{body}"] = scale * (3.0 + noise)
    if terrain in H:
        row[H[terrain]] = level * 0.18
        row["terrain.spacing"] = 0.17
    row.update(extra)
    return row


def study_table():
    rows = [run_row(t, L, c, tr, s) for (t, L, c, tr) in SUCCESS for s in SEEDS]
    rows.append(run_row("rough", 0.15, "fixed", "spring", 99, **{"treatment": "legacy:rigid",      # never pooled
                                                                  "robot_kwargs.body_connection": "rigid",
                                                                  "robot_kwargs.legacy": True}))
    rows.append(run_row("rough", 0.15, "fixed", "spring", 98, status="error", success=None))    # a crashed sim
    return pd.DataFrame(rows)


def wilson(k, n, z=1.959963984540054):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def fig_text(fig):
    return " ".join([t.get_text() for t in fig.texts] + [fig._suptitle.get_text() if fig._suptitle else ""])


@pytest.fixture(scope="module")
def df():
    return study_table()


# ----------------------------------------------------------------------------------------------- preparation
def test_prepare_runs_drops_legacy_and_errors(df):
    d, notes = sp.prepare_runs(df)
    assert len(d) == len(df) - 2
    assert set(d["_treatment"]) == {"spring", "spring_damper"}
    assert set(d["_ctrl"]) == {"fixed (σ = 0)", "adaptive (σ = 0.6)"}
    assert any("legacy (pre-Amendment D)" in n for n in notes) and any("status" in n for n in notes)
    alt = sp.prepare_runs(df.drop(columns=["treatment"]))[0]           # falls back to robot_kwargs.body_connection
    assert len(alt) == len(df) - 2
    # a factor column named like a log field: run_trials keeps the log's value as 'metric.<name>', which wins
    shadow = df.rename(columns={"treatment": "metric.treatment", "controller": "metric.controller"})
    shadow["treatment"], shadow["controller"] = "my label", "my controller"
    d2, _ = sp.prepare_runs(shadow)
    assert d2["_treatment"].tolist() == d["_treatment"].tolist() and d2["_ctrl"].tolist() == d["_ctrl"].tolist()


def test_classify_rule():
    assert sp.classify(0.1, 0.5, "higher", 0.01) == "helps"
    assert sp.classify(0.1, 0.5, "higher", 0.06) == "no clear difference"       # McNemar part of §11.6
    assert sp.classify(-0.5, -0.1, "higher", 0.01) == "hurts"
    assert sp.classify(-0.5, -0.1, "lower") == "helps"
    assert sp.classify(0.1, 0.5, "lower") == "hurts"
    assert sp.classify(-0.1, 0.5, "higher", 0.001) == "no clear difference"
    assert sp.classify(math.nan, 0.5, "higher") == "no clear difference"
    with pytest.raises(ValueError):
        sp.classify(0.1, 0.2, "up")


# ----------------------------------------------------------------------------------------------- known answers
def test_success_vs_roughness_rates_and_wilson(df):
    fig, tab = sp.success_vs_roughness(df, "rough")
    assert isinstance(fig, Figure)
    assert len(tab) == 8 and set(tab["n"]) == {20}
    r = tab.set_index(["level", "treatment", "controller"])
    for (L, c, t), k in {(0.15, "fixed", "spring"): 10, (0.15, "fixed", "spring_damper"): 18,
                         (0.25, "fixed", "spring_damper"): 4, (0.25, "adaptive", "spring"): 10}.items():
        row = r.loc[(L, t, c)]
        assert row["k"] == k and row["rate"] == pytest.approx(k / 20)
        assert (row["lo"], row["hi"]) == pytest.approx(wilson(k, 20), abs=1e-12)
    assert (r.loc[(0.15, "spring", "fixed"), "lo"], r.loc[(0.15, "spring", "fixed"), "hi"]) == \
        pytest.approx((0.2993, 0.7007), abs=1e-4)
    assert r.loc[(0.15, "spring", "fixed"), "difficulty_si"] == pytest.approx(0.027)
    assert r.loc[(0.15, "spring", "fixed"), "difficulty_si_unit"] == "m"
    text = fig_text(fig)
    assert "legacy" in text and "n = runs per point" in text
    # n is written next to every point
    assert sum(1 for t in fig.axes[0].texts if t.get_text() == "n=20") == 8


def test_paired_forest_success_verdicts(df):
    fig, tab = sp.paired_difference_forest(df, "success")
    assert isinstance(fig, Figure) and len(tab) == 6
    r = tab.set_index(["terrain", "level", "controller"])
    helps = r.loc[("rough", 0.15, "fixed")]
    assert helps["n"] == 20 and (helps["n10"], helps["n01"]) == (8, 0)
    assert helps["estimate"] == pytest.approx(0.4) and helps["p_mcnemar"] == pytest.approx(2 * 0.5 ** 8)
    assert helps["lo"] > 0 and helps["verdict"] == "helps"
    assert (helps["mean_spring_damper"], helps["mean_spring"]) == pytest.approx((0.9, 0.5))
    ref = stats.success_difference_ci([s < 18 for s in SEEDS], [s < 10 for s in SEEDS], method="bootstrap")
    assert (helps["lo"], helps["hi"]) == pytest.approx((ref["lo"], ref["hi"]))
    hurts = r.loc[("rough", 0.25, "fixed")]
    assert hurts["estimate"] == pytest.approx(-0.4) and hurts["hi"] < 0 and hurts["verdict"] == "hurts"
    same = r.loc[("rough", 0.15, "adaptive")]
    assert same["estimate"] == 0 and same["p_mcnemar"] == 1.0 and same["verdict"] == "no clear difference"
    swap = r.loc[("rough", 0.25, "adaptive")]
    assert (swap["n10"], swap["n01"]) == (10, 10) and swap["verdict"] == "no clear difference"
    border = r.loc[("alt_bumps", 0.15, "fixed")]             # bootstrap interval excludes 0, McNemar p = 0.0625
    assert border["lo"] > 0 and border["p_mcnemar"] == pytest.approx(0.0625)
    assert border["verdict"] == "no clear difference"
    assert set(tab["n_unpaired"]) == {0}
    assert "higher is better" in fig.axes[0].get_xlabel()


def test_paired_forest_metrics_directions(df):
    _, cot = sp.paired_difference_forest(df, "cot_mech")
    assert set(cot["verdict"]) == {"helps"} and cot["estimate"].to_numpy() == pytest.approx(-0.3)
    assert cot["lo"].to_numpy() == pytest.approx(-0.3) and cot["hi"].to_numpy() == pytest.approx(-0.3)
    fig, roll = sp.paired_difference_forest(df, "payload_roll_rate_rms_rad_s")
    assert set(roll["verdict"]) == {"hurts"} and roll["estimate"].to_numpy() == pytest.approx(0.2)
    assert "lower is better" in fig.axes[0].get_xlabel() and "rad/s" in fig.axes[0].get_xlabel()
    _, slip = sp.paired_difference_forest(df, "slip_per_m")
    assert set(slip["verdict"]) == {"no clear difference"}
    assert slip["estimate"].to_numpy() == pytest.approx(0.0, abs=1e-15)
    assert (slip["lo"] < 0).all() and (slip["hi"] > 0).all() and slip["p_mcnemar"].isna().all()
    _, only = sp.paired_difference_forest(df, "cot_mech", which="success", terrain="rough")
    assert only.set_index(["level", "controller"]).loc[(0.15, "fixed"), "n"] == 10       # both succeeded
    with pytest.raises(ValueError, match="better"):
        sp.paired_difference_forest(df, "body_yaw_rms_max_deg")
    _, und = sp.paired_difference_forest(df, "body_yaw_rms_max_deg", better="lower")
    assert set(und["verdict"]) == {"no clear difference"}


def test_factorial_interaction_known_answer(df):
    fig, tab = sp.factorial_interaction(df, "rough", 0.15)
    assert isinstance(fig, Figure)
    cells = tab.loc[tab["kind"] == "cell"].set_index(["treatment", "controller"])
    assert cells["rate"].to_dict() == pytest.approx({("spring", "fixed"): 0.5, ("spring", "adaptive"): 0.75,
                                                     ("spring_damper", "fixed"): 0.9,
                                                     ("spring_damper", "adaptive"): 0.75})
    lo, hi = wilson(15, 20)
    assert (cells.loc[("spring", "adaptive"), "lo"], cells.loc[("spring", "adaptive"), "hi"]) == pytest.approx((lo, hi))
    eff = tab.loc[tab["kind"] == "effect"].set_index("controller")
    assert eff.loc["fixed", "estimate"] == pytest.approx(0.4) and eff.loc["fixed", "verdict"] == "helps"
    assert eff.loc["adaptive", "estimate"] == 0 and eff.loc["adaptive", "verdict"] == "no clear difference"
    inter = tab.loc[tab["kind"] == "interaction"].iloc[0]
    # (sd·adaptive − sd·fixed) − (spring·adaptive − spring·fixed) = (0.75 − 0.9) − (0.75 − 0.5) = −0.4
    assert inter["estimate"] == pytest.approx(-0.4) and inter["n"] == 20
    a1, a0 = [s < 15 for s in SEEDS], [s < 18 for s in SEEDS]
    b1, b0 = [s < 15 for s in SEEDS], [s < 10 for s in SEEDS]
    ref = stats.difference_of_differences_ci(a1, a0, b1, b0)
    assert (inter["lo"], inter["hi"]) == pytest.approx((ref["lo"], ref["hi"])) and inter["hi"] < 0


def test_achieved_vs_commanded_speed_means():
    rows = [run_row("flat", None, c, t, s, v=v) for v in (0.1, 0.2, 0.3) for c in CTRL for t in sp.TREATMENTS
            for s in SEEDS]
    for r in rows:
        r["success"] = r["seed"] % 4 != 0
    fig, tab = sp.achieved_vs_commanded_speed(pd.DataFrame(rows))
    assert isinstance(fig, Figure) and len(tab) == 12 and set(tab["n"]) == {20} and set(tab["n_success"]) == {15}
    assert tab["mean"].to_numpy() == pytest.approx(0.8 * tab["v_target"].to_numpy())
    assert (tab["lo"] < tab["mean"]).all() and (tab["hi"] > tab["mean"]).all()
    ref = stats.paired_difference_ci([0.8 * 0.2 + 0.01 * (s - 9.5) / 9.5 for s in SEEDS], np.zeros(20))
    row = tab.loc[np.isclose(tab["v_target"], 0.2)].iloc[0]
    assert (row["lo"], row["hi"]) == pytest.approx((ref["lo"], ref["hi"]))
    ax = fig.axes[0]
    from matplotlib.collections import PathCollection

    hollow = [c for c in ax.collections if isinstance(c, PathCollection) and len(c.get_offsets())
              and np.all(np.asarray(c.get_facecolors()).reshape(-1, 4)[:, 3] == 0)]
    assert sum(len(c.get_offsets()) for c in hollow) == 3 * 4 * 5          # every failed run drawn, hollow


def test_angular_motion_worst_segment_and_which(df):
    fig, tab = sp.angular_motion_and_slip_vs_roughness(df, "rough")
    assert isinstance(fig, Figure)
    assert set(tab["metric"]) == {k for k, _, _ in sp.ANGULAR_PANELS} and set(tab["which"]) == {"all"}
    assert set(tab["n"]) == {20}
    w = tab.loc[tab["metric"] == "worst_segment:roll_rate_rms_rad_s"]
    assert w["mean"].to_numpy() == pytest.approx(2.0 * 0.5)                # segment 2 is the worst, noise mean 0
    assert "segment 1, segment 2, segment 3" in w["label"].iloc[0]
    p = tab.loc[tab["metric"] == "payload_roll_rate_rms_rad_s"].set_index(["level", "treatment", "controller"])
    assert p.loc[(0.15, "spring_damper", "fixed"), "mean"] == pytest.approx(0.7)
    assert "failed runs included" in fig_text(fig)
    fig, tab = sp.angular_motion_and_slip_vs_roughness(df, "rough", which="success")
    s = tab.set_index(["metric", "level", "treatment", "controller"])
    assert s.loc[("slip_per_m", 0.15, "spring", "fixed"), "n"] == 10 and set(tab["which"]) == {"success"}
    assert "successful runs only" in fig_text(fig)


def test_cost_vs_speed_marks_failures(df):
    fig, tab = sp.cost_vs_speed(df)
    assert isinstance(fig, Figure) and len(fig.axes) == 2                   # one panel per terrain kind
    assert len(tab) == len(df) - 2 and tab["success"].dtype == bool
    assert "covers its distance before the failure" in fig_text(fig)
    fig, tab = sp.cost_vs_speed(df, terrain="rough")
    labels = [t.get_text() for t in fig.axes[-1].get_legend().get_texts()]
    # spring-only · fixed on rough: 10 + 12 successes of 40 runs; spring–damper · adaptive: 15 + 10
    assert any(t.startswith("spring-only") and "fixed" in t and "n = 22 ok + 18 failed" in t for t in labels)
    assert any(t.startswith("spring–damper") and "adaptive" in t and "n = 25 ok + 15 failed" in t for t in labels)
    assert any("hollow" in t for t in labels)
    assert tab.loc[~tab["success"], "cot_mech"].notna().all()               # failed runs keep their CoT


def test_undulation_onset():
    rows = []
    slope = {(4.0, "spring"): 40.0, (8.0, "spring"): 20.0, (4.0, "spring_damper"): 20.0, (8.0, "spring_damper"): 10.0}
    for (k, t), a in slope.items():
        for c in CTRL:
            for v in (0.1, 0.2, 0.3, 0.4, 0.5):
                for s in range(10):
                    yaw = a * v + (0.5 if s % 2 else -0.5)                # deg; zero-mean noise
                    rows.append(run_row("flat", None, c, t, s, v=v,
                                        **{"robot_kwargs.body_k_yaw": k, "body_yaw_rms_max_deg": yaw}))
    fig, tab = sp.undulation_vs_speed(pd.DataFrame(rows))
    assert isinstance(fig, Figure) and set(tab["n"]) == {10}
    on = tab.drop_duplicates(["controller", "k_yaw", "treatment"]).set_index(["controller", "k_yaw", "treatment"])
    assert on.loc[("fixed", 8.0, "spring"), "onset_speed_m_s"] == pytest.approx(0.3)      # means 2, 4, 6, ...
    assert on.loc[("fixed", 8.0, "spring"), "onset_interp_m_s"] == pytest.approx(0.25)
    assert on.loc[("fixed", 4.0, "spring"), "onset_speed_m_s"] == pytest.approx(0.2)
    assert on.loc[("fixed", 4.0, "spring_damper"), "onset_speed_m_s"] == pytest.approx(0.3)
    assert math.isnan(on.loc[("adaptive", 8.0, "spring_damper"), "onset_speed_m_s"])      # means 1 ... 5: never > 5
    assert on.loc[("adaptive", 8.0, "spring_damper"), "onset_flag"] == "above_range"
    m = tab.set_index(["controller", "k_yaw", "treatment", "v_target"])["mean"]
    assert m.loc[("fixed", 8.0, "spring", 0.4)] == pytest.approx(8.0)
    # the default stiffness fills a missing column (protocol §12.1: 8 N·m/rad)
    _, t2 = sp.undulation_vs_speed(pd.DataFrame([r for r in rows if r["robot_kwargs.body_k_yaw"] == 8.0])
                                   .drop(columns=["robot_kwargs.body_k_yaw"]))
    assert set(t2["k_yaw"]) == {8.0}


# ----------------------------------------------------------------------------------------------- guards
def test_never_pools_configurations_or_duplicates(df):
    mixed = df.copy()
    mixed["robot_kwargs.body_k_pitch"] = np.where(mixed["seed"] % 2 == 0, 8.0, 4.0)
    with pytest.raises(ValueError, match="mix configurations"):
        sp.success_vs_roughness(mixed, "rough")
    terrain_mix = df.copy()
    terrain_mix.loc[(terrain_mix["seed"] == 0) & (terrain_mix["terrain.kind"] == "rough"), "terrain.rms"] = 0.05
    for fn in (sp.success_vs_roughness, sp.angular_motion_and_slip_vs_roughness):
        with pytest.raises(ValueError, match="terrain.rms"):
            fn(terrain_mix, "rough")
    with pytest.raises(ValueError, match="terrain.rms"):
        sp.paired_difference_forest(terrain_mix, "success")
    dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="not unique"):
        sp.paired_difference_forest(dup, "success")
    two_speeds = pd.concat([df, df.assign(v_target=0.3)], ignore_index=True)
    with pytest.raises(ValueError, match="v_target"):
        sp.angular_motion_and_slip_vs_roughness(two_speeds, "rough")
    fig, tab = sp.success_vs_roughness(two_speeds, "rough")                  # one panel per speed
    assert sorted(tab["v_target"].unique()) == [0.2, 0.3] and len(fig.axes) == 2
    _, forest = sp.paired_difference_forest(two_speeds, "success")
    assert len(forest) == 12
    with pytest.raises(ValueError, match="terrain"):
        sp.success_vs_roughness(df, "steps")
    unpaired = df.loc[~((df["treatment"] == "spring") & (df["seed"] == 3))]
    _, f = sp.paired_difference_forest(unpaired, "success", terrain="rough")
    assert set(f["n"]) == {19} and set(f["n_unpaired"]) == {1}


# ----------------------------------------------------------------------------------------------- traces
def cleo_like_log(T=60, dt=0.01, wobble=1.0):
    """A Cleopatra-shaped log: head + 3 segments, 12 feet (4 per segment), 6 body hinges, 3 leg joints."""
    t = np.arange(T) * dt
    bodies = ["head", "segment 1", "segment 2", "segment 3"]
    feet = [f"s{s} {k}" for s in (1, 2, 3) for k in ("front-left", "front-right", "rear-left", "rear-right")]
    body_joints = [f"body {i}-{i + 1} {a}" for i in (1, 2) for a in ("yaw", "pitch", "roll")]
    joints = ["s1 front-left hip_yaw", "s1 front-left hip_pitch", "s1 front-left knee"] + body_joints
    kinds = ["hip_yaw", "hip_pitch", "knee"] + [f"body_{j.rsplit(' ', 1)[1]}" for j in body_joints]
    roll = wobble * 0.05 * np.sin(2 * np.pi * t)[:, None] * np.arange(1, 5)
    quat = np.zeros((T, 4, 4))
    quat[..., 0], quat[..., 1] = np.cos(roll / 2), np.sin(roll / 2)
    q = np.zeros((T, len(joints)))
    q[:, 3:] = wobble * 0.02 * np.sin(2 * np.pi * t)[:, None]
    force = np.zeros((T, 12, 3))
    force[..., 2] = 60.5 / 12
    normal = np.tile([0.0, 0.0, 1.0], (T, 12, 1))
    return {"t": t, "bodies": bodies, "body_pos": np.zeros((T, 4, 3)), "body_quat": quat,
            "body_angvel": np.zeros((T, 4, 3)), "belly_contact": np.zeros((T, 4), bool),
            "com": np.zeros((T, 3)), "com_vel": np.zeros((T, 3)), "total_mass": 6.17,
            "gravity": np.array([0, 0, -9.81]), "feet": feet, "foot_body": np.repeat([1, 2, 3], 4),
            "foot_force": force, "foot_normal": normal, "foot_pos": np.zeros((T, 12, 3)), "joints": joints,
            "joint_kind": kinds, "joint_active": np.array([True] * 3 + [False] * 6), "q": q, "qd": np.zeros_like(q),
            "tau": np.zeros_like(q), "tau_stall": np.r_[[6.0] * 3, [np.nan] * 6],
            "qd_noload": np.r_[[5.76] * 3, [np.nan] * 6], "robot": "cleopatra test", "seed": 3, "v_target": 0.2}


def test_synchronized_traces_synthetic():
    a, b = cleo_like_log(), cleo_like_log(wobble=2.0)
    fig, tab = sp.synchronized_traces(a, b, ("spring", "spring_damper"), q0=0.0,
                                      outcomes=({"reason": "success", "t_end": 0.5}, None))
    assert isinstance(fig, Figure)
    counts = tab.groupby(["run", "panel"])["entity"].nunique().to_dict()
    assert counts[("spring", "roll")] == 3 and counts[("spring", "pitch")] == 3          # segments, not the head
    assert counts[("spring", "normal_force")] == 12 and counts[("spring_damper", "deflection")] == 6
    d = tab.loc[(tab["run"] == "spring_damper") & (tab["panel"] == "deflection") & (tab["entity"] == "body 1-2 yaw")]
    assert d["value"].to_numpy() == pytest.approx(np.degrees(0.04 * np.sin(2 * np.pi * np.arange(60) * 0.01)))
    r = tab.loc[(tab["run"] == "spring") & (tab["panel"] == "roll") & (tab["entity"] == "segment 3")]
    assert r["value"].to_numpy() == pytest.approx(np.degrees(0.2 * np.sin(2 * np.pi * np.arange(60) * 0.01)))
    assert set(tab.loc[tab["panel"] == "normal_force", "unit"]) == {"N"}
    assert len(fig.axes) == 2 * (2 + 3 + 3)                                             # 8 rows, 2 runs
    shared = fig.axes[0].get_shared_x_axes()
    assert all(shared.joined(fig.axes[0], ax) for ax in fig.axes)
    assert "success at 0.50 s" in fig.axes[0].get_title()
    assert "into the ground" not in fig_text(fig)
    b["foot_force"][10, 4] = [0.0, 0.0, -30.0]                                          # a trapped pad
    fig, _ = sp.synchronized_traces(a, b, ("spring", "spring_damper"), q0=0.0)
    assert "spring_damper: the net contact force on a foot points into the ground (N < 0, down to -30 N) in 1 of" \
        in fig_text(fig)


# ----------------------------------------------------------------------------------------------- a real study row
def test_real_cleopatra_runs_through_the_trial_runner(tmp_path):
    """Two treatments × two controllers × two seeds of a short real course through chiron.experiments.run_trials:
    every plot accepts the real columns, and the paired runs' traces come from rerun_with_log."""
    pytest.importorskip("mujoco")
    import myropod_robot as mr
    from vegeta.chiron import experiments as ex

    designs = sp.__file__.rsplit("/", 1)[0]

    def make(body, controller, seed):
        kw = {"sigma": 0.6} if controller == "adaptive" else {}
        return ex.Trial(robot=f"{designs}/myropod_robot.py:cleopatra", robot_kwargs={"body_connection": body},
                        controller=f"{designs}/myropod_controller.py:{controller}", controller_kwargs=kw,
                        seed=seed, v_target=0.2, course_m=0.04, settle=0.2,
                        terrain={"kind": "rough", "rms": 0.002, "correlation_length": 0.0425, "start": 0.3,
                                 "level": 0.011},
                        lab_kwargs=mr.LAB_OPTIONS, info={"payload": "head"}, rules={"timeout": 0.6},
                        metrics_kwargs={"feasibility_every": 0, "steady_from_s": 0.1})

    trials = ex.paired_trials(make, {"body": list(sp.TREATMENTS), "controller": ["fixed", "adaptive"]}, [0, 1])
    df = ex.run_trials(trials, processes=1, progress=False)
    assert (df["status"] == "ok").all() and set(df["treatment"]) == set(sp.TREATMENTS)
    for fn, args in ((sp.success_vs_roughness, ("rough",)), (sp.achieved_vs_commanded_speed, ()),
                     (sp.angular_motion_and_slip_vs_roughness, ("rough",)), (sp.cost_vs_speed, ()),
                     (sp.paired_difference_forest, ("success",)), (sp.factorial_interaction, ("rough", 0.011)),
                     (sp.undulation_vs_speed, ())):
        kw = {"terrain": "rough"} if fn is sp.undulation_vs_speed else {}
        fig, tab = fn(df, *args, **kw)
        assert isinstance(fig, Figure) and len(tab), fn.__name__
        fig.savefig(tmp_path / f"{fn.__name__}.png", dpi=60)
    _, forest = sp.paired_difference_forest(df, "success")
    assert set(forest["n"]) == {2} and set(forest["controller"]) == {"fixed", "adaptive"}
    _, inter = sp.factorial_interaction(df, "rough", 0.011)
    assert inter.loc[inter["kind"] == "interaction", "n"].iloc[0] == 2
    a, b = (ex.rerun_with_log(t) for t in trials[:2])
    fig, tab = sp.synchronized_traces(a, b, ("spring", "spring_damper"))
    assert set(tab["panel"]) == {"roll", "pitch", "normal_force", "deflection"}
    assert tab.loc[tab["panel"] == "deflection", "entity"].nunique() == 6
    fig.savefig(tmp_path / "traces.png", dpi=60)
