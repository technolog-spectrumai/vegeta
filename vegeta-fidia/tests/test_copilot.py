import json

import pytest

from vegeta.fidia import DesignSession, Proposal
from vegeta.fidia.proposer import user_message as _user_message
from _helpers import DESIGN, DESIGN_WITH_HOLE, FakeProposer, answer


def test_source_proposal_is_validated_and_only_written_on_accept(plate):
    fake = FakeProposer(answer("source", source=DESIGN_WITH_HOLE, parameters={"hole": 5.0}, summary="add hole"))
    s = DesignSession(f"{plate}:Plate", fake, parameters={"width": 40.0})
    prop = s.ask("add a central hole")
    assert prop.kind == "source" and prop.ok, prop.validation.messages
    assert prop.validation.parameters_added == ["hole"]
    assert prop.validation.parameters_after == {"width": 40.0, "thickness": 2.0, "hole": 5.0}
    before, after = prop.validation.changes()["volume"]
    assert after < before                                  # the hole removed material
    assert "+                  Parameter(\"hole\"" in prop.diff or "hole" in prop.diff
    assert plate.read_text() == DESIGN                     # untouched until accept
    assert fake.calls[0]["context"]["parameters"][0] == {"name": "width", "value": 40.0, "units": "mm", "min": 5,
                                                          "max": None, "description": ""}
    s.accept(prop, note="good")
    assert plate.read_text() == DESIGN_WITH_HOLE and (plate.parent / "plate.py.1.bak").read_text() == DESIGN
    assert s.parameters == {"width": 40.0, "hole": 5.0}
    events = [json.loads(l)["event"] for l in s.log.read_text().splitlines()]
    assert events == ["proposal", "accept"]


def test_broken_proposal_fails_validation_and_cannot_be_accepted(plate):
    bad = DESIGN.replace('box(p["width"], 10, p["thickness"])', 'box(p["width"], 10, p["thickness"]).edges().fillet(50)')
    s = DesignSession(f"{plate}:Plate", FakeProposer(answer("source", source=bad)))
    prop = s.ask("round all edges")
    assert not prop.ok and prop.validation.messages
    with pytest.raises(ValueError, match="did not validate"):
        s.accept(prop)
    assert plate.read_text() == DESIGN


def test_parameter_proposal_and_bounds(plate):
    s = DesignSession(f"{plate}:Plate", FakeProposer(answer("parameters", parameters={"thickness": 4.0}),
                                                     answer("parameters", parameters={"thickness": 0.1})))
    ok = s.ask("make it stiffer")
    assert ok.ok and ok.validation.measurements_after["volume"] == pytest.approx(30 * 10 * 4)
    bad = s.ask("paper thin")
    assert not bad.ok and "below minimum" in bad.validation.messages[0]


def test_answer_kind_changes_nothing_and_history_flows(plate):
    fake = FakeProposer(answer("answer", summary="need the load direction"), answer("answer"))
    s = DesignSession(f"{plate}:Plate", fake)
    a1 = s.ask("optimise it")
    assert a1.kind == "answer" and a1.ok and a1.diff == ""
    s.reject(a1, "not helpful")
    s.ask("vertical load, 50 N")
    assert [h["user"] for h in fake.calls[1]["history"]] == ["optimise it", "(rejected proposal p1: not helpful)"]


def test_geometry_preview_without_accept(plate):
    s = DesignSession(f"{plate}:Plate", FakeProposer(answer("source", source=DESIGN_WITH_HOLE)))
    prop = s.ask("hole")
    g = s.geometry(prop)
    assert g.measure()["n_faces"] > 6 and plate.read_text() == DESIGN


def test_user_message_contains_state():
    ctx = {"path": "x.py", "source": "src", "parameters": [{"name": "w", "value": 1, "units": "mm", "min": 0,
            "max": None, "description": "d"}], "measurements": {"volume": 1.0}, "notes": "keep holes"}
    msg = _user_message(ctx, "do it")
    for needle in ("x.py", "src", "- w: 1 mm", "volume", "keep holes", "## Instruction", "do it"):
        assert needle in msg


def test_rejects_non_python_spec(tmp_path):
    with pytest.raises(ValueError):
        DesignSession("vegeta.dedalus.examples:Cube", FakeProposer())


def test_proposed_code_never_runs_in_this_process(plate, monkeypatch):
    import vegeta.dedalus.loading as loading

    def forbidden(*a, **k):
        raise AssertionError("design code was loaded in the test process")
    monkeypatch.setattr(loading, "load_module", forbidden)
    evil = "import os\n" + DESIGN.replace("return", "os.system('true')\n        return")
    s = DesignSession(f"{plate}:Plate", FakeProposer(answer("source", source=evil), answer("source", source=DESIGN_WITH_HOLE)))
    bad = s.ask("do something")
    assert not bad.ok and any("rejected" in m for m in bad.validation.messages)
    good = s.ask("add a hole")
    assert good.ok and good.validation.measurements_after["valid"]
