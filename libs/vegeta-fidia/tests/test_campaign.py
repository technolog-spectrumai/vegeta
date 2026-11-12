import json
import shutil

import pytest

from vegeta import core
from vegeta.fidia import Analysis, Budget, Campaign, Criterion, Objective
from _helpers import DESIGN, FakeProposer, answer


@pytest.fixture
def ws(tmp_path):
    (tmp_path / "plate.py").write_text(DESIGN)
    w = core.Workspace.create(tmp_path / "ws", name="t")
    d = w.add_design("plate", str(tmp_path / "plate.py") + ":Plate")
    return w, d.new_revision(width=30.0, thickness=2.0, note="start")


def params(**p):
    return answer("parameters", parameters=p, summary=f"try {p}")


def test_loop_finds_feasible_design_and_records_everything(ws):
    w, r0 = ws
    # volume = width * 10 * thickness; require >= 900 mm^3, minimise surface area
    fake = FakeProposer(params(thickness=4.0), params(thickness=3.0), params(width=40.0),
                        answer("answer", summary="done"))
    c = Campaign(w, r0, analyses=[], criteria=[Criterion("geometry.volume", ">=", 900.0)],
                 objective=Objective("geometry.surface_area", "min"), proposer=fake, approval="auto",
                 budget=Budget(max_iterations=10, patience=5), name="vol")
    c.run(progress=False)
    t = c.table()
    assert list(t["status"]) == ["evaluated"] * 4 + ["proposer_stopped"]
    assert list(t["feasible"])[:4] == [False, True, True, True]
    assert c.best_id == t.loc[2, "revision"]                    # 30 x 10 x 3: feasible, smallest area
    assert c.stopped.startswith("the proposer ended")
    # the proposer saw the table of earlier steps and the incumbent
    ctx = fake.calls[2]["context"]
    assert "step 1" in ctx["notes"] and ctx["campaign"]["incumbent"] == t.loc[2, "revision"]
    assert fake.calls[0]["context"]["parameters"][1]["value"] == 2.0
    # every candidate is an ordinary revision branched from the incumbent; nothing is labelled
    r3 = w.revision(t.loc[3, "revision"])
    assert r3.parent == t.loc[2, "revision"] and r3.params["width"] == 40.0
    assert all(r.current_label == "unclassified" for r in w.revisions())
    state = json.loads((c.dir / "campaign.json").read_text())
    assert state["best"] == c.best_id and len(state["iterations"]) == 5
    events = [json.loads(l)["event"] for l in (c.dir / "events.jsonl").read_text().splitlines()]
    assert events.count("step") == 5 and events[-1] == "stopped"


def test_refusals_budget_and_patience(ws):
    w, r0 = ws
    fake = FakeProposer(answer("source", source=DESIGN, summary="edit source"),
                        params(color=1.0), params(width=31.0), params(thickness=0.1),
                        params(thickness=2.0), params(width=32.0), params(width=33.0))
    c = Campaign(w, r0, analyses=[], criteria=[Criterion("geometry.volume", ">=", 1e9)],
                 objective=Objective("geometry.volume", "min"), proposer=fake, approval="auto",
                 free=["thickness"], budget=Budget(max_iterations=20, patience=5), name="ref")
    c.run(progress=False)
    reasons = [r for r in c.table()["status"]]
    assert reasons == ["evaluated", "refused", "refused", "refused", "refused", "refused"]
    msgs = [it["reason"] for it in c.iterations[1:]]
    assert "only accepts parameter changes" in msgs[0]
    assert "not free" in msgs[2]                                 # width is fixed in this campaign
    assert "thickness" in msgs[3]                                # below the design's min=0.5
    assert "already evaluated" in msgs[4]
    assert c.stopped.startswith("no new best") and c.best_id is None
    assert len(w.revisions()) == 1                               # nothing was built


def test_declined_proposal_stops_and_campaign_resumes(ws):
    w, r0 = ws
    seen = []

    def policy(p):
        seen.append(p)
        return p["changes"].get("thickness", 0) <= 3.0          # a written approval policy

    kw = dict(analyses=[], criteria=[Criterion("geometry.volume", ">=", 700.0)],
              objective=Objective("geometry.volume", "min"), approval=policy, name="pol")
    c = Campaign(w, r0, proposer=FakeProposer(params(thickness=2.5), params(thickness=6.0)), **kw)
    c.run(progress=False)
    assert c.stopped.startswith("declined") and [it["status"] for it in c.iterations] == \
        ["evaluated", "evaluated", "declined"]
    assert seen[0]["values"]["thickness"] == 2.5
    again = Campaign(w, r0, proposer=FakeProposer(params(thickness=2.4)), budget=Budget(max_iterations=1), **kw)
    again.run(progress=False)                                    # continues the recorded campaign
    assert [it["step"] for it in again.iterations] == [0, 1, 2, 3] and again.best_id == again.iterations[3]["revision"]


def test_stop_file_and_token_budget(ws):
    w, r0 = ws
    kw = dict(analyses=[], criteria=[], objective=Objective("geometry.volume", "min"), approval="auto")
    c = Campaign(w, r0, proposer=FakeProposer(params(thickness=3.0)), budget=Budget(max_tokens=2), name="tok", **kw)
    c.run(progress=False)
    assert c.stopped.startswith("token budget") and len(c.iterations) == 2
    c2 = Campaign(w, r0, proposer=FakeProposer(), name="stop", **kw)
    c2.dir.mkdir(parents=True)
    (c2.dir / "STOP").touch()
    c2.run(progress=False)
    assert c2.stopped == "STOP file found" and len(c2.iterations) == 1


def test_validation_of_arguments(ws):
    w, r0 = ws
    with pytest.raises(ValueError, match="op"):
        Criterion("geometry.volume", "==", 1)
    with pytest.raises(ValueError, match="factory"):
        Analysis("fea", "static")
    with pytest.raises(ValueError, match="unknown parameters"):
        Campaign(w, r0, analyses=[], criteria=[], objective=Objective("geometry.volume"), proposer=FakeProposer(),
                 free=["nope"])


@pytest.mark.skipif(shutil.which("ccx") is None, reason="needs CalculiX (ccx)")
def test_fea_campaign(ws):
    from vegeta import talos

    w, r0 = ws

    def bend(rev):
        return talos.StructuralModel(
            rev.step, "mm-N-MPa", talos.Material("Al", youngs_modulus=70000, poissons_ratio=0.33, yield_strength=250),
            regions=[talos.SurfacesOnPlane("fixed", "x", -rev.params["width"] / 2),
                     talos.SurfacesOnPlane("tip", "x", rev.params["width"] / 2)],
            supports=[talos.FixedSupport("fixed")], loads=[talos.Force("tip", fz=-20.0)],
            mesh_settings=talos.MeshSettings(element_size=2.0))

    c = Campaign(w, r0, analyses=[Analysis("fea", "bend", bend)],
                 criteria=[Criterion("fea.bend.safety_factor_yield", ">=", 2.0)],
                 objective=Objective("geometry.volume", "min"), proposer=FakeProposer(params(thickness=4.0)),
                 approval="auto", budget=Budget(max_iterations=1), name="fea")
    c.run(progress=False)
    t = c.table()
    assert t.loc[1, "fea.bend.safety_factor_yield >= 2"] > t.loc[0, "fea.bend.safety_factor_yield >= 2"]
    assert w.revision(t.loc[1, "revision"]).evaluation("fea", "bend").ok
