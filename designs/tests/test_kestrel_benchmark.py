"""The benchmark reads the three design files into one table and ranks; missing files leave rows out, not errors."""
import kestrel_benchmark as kb


def test_benchmark_loads_ranks_and_tolerates_missing(tmp_path):
    df, notes = kb.load()
    assert "Kestrel-4Q-2" in df.index and any(i.startswith("Kestrel-X-") for i in df.index) and "Kestrel-Q4T1N-10" in df.index
    assert set(kb.MEASURES) <= set(df.columns)
    r = kb.ranks(df)
    assert r.loc["Kestrel-4Q-2", "hover pitch [rad/s²]"] == 1.0 and r["measures ranked"].max() >= 8
    df2, notes2 = kb.load({"45": tmp_path / "nope.json"})
    assert not any(i.startswith("Kestrel-X-") for i in df2.index) and any("45:" in n and "missing" in n for n in notes2)


def test_scores_are_anchored_and_missions_weight_them():
    import pandas as pd
    df, _ = kb.load()
    sc = kb.scores(df)
    assert set(sc.columns) == set(kb.CATEGORIES) and ((sc.fillna(5) >= 1) & (sc.fillna(5) <= 10)).all().all()
    assert sc.loc["Kestrel-Q4T1N-10", "structure"] == 10.0 and sc.loc["Kestrel-4Q-2", "hover authority"] > sc.loc["Kestrel-Q4T1N-10", "hover authority"]
    for m, w in kb.MISSIONS.items():
        assert abs(sum(w.values()) - 1) < 1e-9 and set(w) == set(kb.CATEGORIES)
    wt = kb.weighted(sc)
    assert ((wt[list(kb.MISSIONS)].astype(float) >= 1) & (wt[list(kb.MISSIONS)].astype(float) <= 10)).all().all()
    assert wt.loc["Kestrel-4Q-2", "medical delivery (hover at the site)"] > wt.loc["Kestrel-Q4T1N-10", "medical delivery (hover at the site)"]
    assert abs(kb._scale(120.0, 120.0, 50.0) - 1) < 1e-9 and abs(kb._scale(50.0, 120.0, 50.0) - 10) < 1e-9 and kb._scale(None, 0, 1) is None


def test_three_civilian_missions_score_as_one():
    df, _ = kb.load()
    civ = kb.group_scores(kb.weighted(kb.scores(df)))
    assert "3-civilian-missions" in civ.columns and civ.index[0] == "Kestrel-4Q-2"           # the biplane leads the set as the notebooks stand
    assert abs(sum(kb.MISSION_GROUPS["3-civilian-missions"].values()) - 1) < 1e-9
    assert set(kb.CIVILIAN) == {"maritime rescue", "urgent industrial logistics", "emergency medical delivery"}
