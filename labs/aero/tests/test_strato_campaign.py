"""Strato campaign plumbing (Stage V8.14): the fast stages run, write their JSON and plots, and the report is built
from whatever results exist (and says QUICK when they came from a shrunk run)."""
import json
import sys
from pathlib import Path

import pytest

CAMPAIGN = Path(__file__).resolve().parents[1] / "campaigns" / "strato"


@pytest.fixture(scope="module")
def run():
    sys.path.insert(0, str(CAMPAIGN))
    import run as r
    r.style()
    return r


def test_fast_stages_and_report(run, tmp_path):
    for name in ("atmosphere", "wind"):
        res = run.STAGES[name](tmp_path, True)
        res["quick"] = True
        run.jdump(res, tmp_path / f"{name}.json")
        for p in res["plots"]:
            assert (tmp_path / p).stat().st_size > 10_000
    atm = json.loads((tmp_path / "atmosphere.json").read_text())
    assert atm["table"][3]["rho"] == pytest.approx(0.08891, rel=2e-4)     # 20 km
    assert atm["isa_error_at_20km"] > 0.05                                 # the troposphere model is ~7 % dense up there
    import report
    text = report.write(tmp_path).read_text()
    assert "QUICK RUN" in text and "What is verified and what is estimated" in text


def test_json_has_no_nan(run, tmp_path):
    run.jdump({"a": float("nan"), "b": [1.0, float("inf")]}, tmp_path / "x.json")
    assert json.loads((tmp_path / "x.json").read_text()) == {"a": None, "b": [1.0, None]}
