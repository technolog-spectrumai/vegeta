import numpy as np

from aero_kit import plain


def test_plain_makes_json_values():
    assert plain(np.float64(1.5)) == 1.5 and plain(np.nan) is None and plain(np.int32(3)) == 3
    assert plain(np.bool_(True)) is True and plain("x") == "x"


def test_save_table_writes_and_returns(tmp_path):
    import pandas as pd

    from aero_kit import save_table
    (tmp_path / "tables").mkdir()
    df = pd.DataFrame({"a": [1, 2]})
    assert save_table(df, "t", tmp_path) is df and (tmp_path / "tables" / "t.csv").exists()


def test_propeller_keywords():
    from aero_kit import cad_kw, pod_p
    blade = {"diameter_mm": 254, "pitch_mm": 152, "chord_root_mm": 20, "chord_max_mm": 25, "chord_tip_mm": 10,
             "thickness": 0.1, "camber": 0.04}
    hub = {"hub_diameter": 20, "hub_height": 10}
    kw = cad_kw(3, blade, hub)
    assert kw["blades"] == 3 and kw["diameter"] == 254 and kw["hub_diameter"] == 20 and kw["stations"] == 8
    p = pod_p("pusher", {"length": 100}, hub, {"pusher": 5, "tractor": 4})
    assert p == {"length": 100, "layout": "pusher", "hub_height": 10, "hub_diameter": 20, "gap": 5}
