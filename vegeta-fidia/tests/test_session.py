import json
import threading

import pytest
from _helpers import TWO_PART

from vegeta.ai import ScriptedProvider
from vegeta.fidia.agent import ModelAgent
from vegeta.fidia.demo import DEMO_PROMPT, DemoProvider, demo_agent
from vegeta.fidia.revision import Revision, RevisionSealed
from vegeta.fidia.sandbox import Sandbox
from vegeta.fidia.schemas import GENERATION_SCHEMA, PLAN_SCHEMA, REVIEW_SCHEMA
from vegeta.fidia.session import Limits, Session

pytestmark = pytest.mark.slow

PROMPT = "a post on a base"
PLAN = {"object": "post on a base", "description": "a square post standing on a plate", "size_mm": [100, 60, 60],
        "parts": [{"name": "base", "shape": "box", "color": [0.8, 0.1, 0.1], "connects_to": ["post"]},
                  {"name": "post", "shape": "box", "color": [0.1, 0.2, 0.9], "connects_to": ["base"]}],
        "floating_ok": False, "acceptance": ["a post stands on a base"], "assumptions": ""}
BROKEN = TWO_PART.replace("        base = ", '        raise ValueError("boom")\n        base = ')
LOOP = TWO_PART.replace("        base = ", "        while True:\n            pass\n        base = ")


def gen(source=TWO_PART, summary="s"):
    return {"summary": summary, "addresses": [], "source": source}


def review(score=9, verdict="accept", ok=True):
    return {"score": score, "verdict": verdict, "summary": "r", "issues": [], "next_instruction": "",
            "acceptance": [{"item": "a post stands on a base", "ok": ok, "note": ""}]}


def make(tmp_path, *answers, name="run", **kw):
    prov = ScriptedProvider(*answers, tokens_per_call=(100, 50))
    return Session(PROMPT, tmp_path / name, agent=ModelAgent(prov), progress=False, **kw), prov


def schemas(prov):
    names = {id(PLAN_SCHEMA): "plan", id(GENERATION_SCHEMA): "generate", id(REVIEW_SCHEMA): "review"}
    return [names[id(c["schema"])] for c in prov.calls]


def test_demo_reaches_done_in_two_revisions(tmp_path):
    s = Session(DEMO_PROMPT, tmp_path / "run", agent=demo_agent(), progress=False).run()
    assert s.done and s.stopped.startswith("done")
    rows = s.state["revisions"]
    assert [r["score"] for r in rows] == [5, 9] and rows[0]["warnings"] == 1 and rows[1]["done"]
    assert s.best.name == "rev-002" and (s.dir / "best" / "BEST").read_text().strip() == "rev-002"
    for f in ("model.glb", "model.gltf", "model.bin", "model.obj", "model.mtl", "model.stl", "model.step"):
        assert (s.dir / "best" / "export" / f).is_file(), f
    assert json.loads((s.dir / "best" / "export" / "reimport.json").read_text())["ok"]
    assert (s.dir / "rev-001" / "renders" / "sheet.png").is_file()
    state = json.loads((s.dir / "run.json").read_text())
    assert state["usage"]["tokens"] == 5 * 3700 and state["status"] == "done"
    events = [json.loads(line)["event"] for line in (s.dir / "events.jsonl").read_text().splitlines()]
    assert events[0] == "created" and events[-1] == "stopped" and events.count("revision") == 2


def test_review_skipped_when_the_build_fails(tmp_path):
    s, prov = make(tmp_path, PLAN, gen(BROKEN), gen(), review())
    s.run()
    assert schemas(prov) == ["plan", "generate", "generate", "review"]
    r1, r2 = s.revisions
    assert r1.status == "build_error" and not r1.valid and r1.score is None and r1.review is None
    assert "boom" in prov.calls[2]["messages"][0]["content"]  # the error went back to the modeller
    assert r2.done and s.best.name == "rev-002"


def test_iteration_token_and_time_limits(tmp_path):
    s, _ = make(tmp_path, PLAN, gen(), review(5, "revise"), limits=Limits(max_iterations=1))
    assert s.run().stopped == "max_iterations (1) reached"
    s, prov = make(tmp_path, PLAN, name="tokens", limits=Limits(max_tokens=100))
    s.agent.max_tokens = 500  # a call may use 500 tokens: over budget before the first call
    assert s.run().stopped.startswith("token budget") and prov.calls == [] and not s.revisions
    s, prov = make(tmp_path, PLAN, name="minutes", limits=Limits(max_minutes=0))
    assert s.run().stopped.startswith("time limit") and prov.calls == []


def test_token_budget_is_checked_before_each_call(tmp_path):
    s, prov = make(tmp_path, PLAN, gen(), review(), limits=Limits(max_tokens=400))
    s.agent.max_tokens = 150  # plan (150) + generate (300) fit, the review would not
    s.run()
    assert schemas(prov) == ["plan", "generate"]
    assert s.revisions[0].status == "cancelled" and s.revisions[0].record["cancelled_during"] == "review"
    assert s.stopped.startswith("token budget")


def test_patience(tmp_path):
    s, _ = make(tmp_path, PLAN, gen(), review(6, "revise"), gen(), review(6, "revise"), gen(), review(5, "revise"),
                limits=Limits(patience=2))
    s.run()
    assert len(s.revisions) == 3 and "patience 2" in s.stopped
    assert s.best.name == "rev-001"  # equal score: the earlier one stays


def test_stop_file_and_cancel(tmp_path):
    s, prov = make(tmp_path, PLAN)
    (s.dir / "STOP").write_text("")
    assert s.run().stopped.startswith("STOP file") and prov.calls == []

    def cancel_after(call):
        s2.cancel()
        return gen()
    s2, _ = make(tmp_path, PLAN, cancel_after, name="cancel")
    s2.run()
    rev = s2.revisions[0]
    assert rev.status == "cancelled" and rev.record["cancelled_during"] == "execute"
    assert s2.stopped == "cancelled by the user" and not (rev.path / "execution.json").exists()


def test_cancel_kills_a_running_build(tmp_path):
    s, _ = make(tmp_path, PLAN, gen(LOOP), sandbox=Sandbox(timeout_s=120))
    threading.Timer(4.0, s.cancel).start()
    s.run()
    rev = s.revisions[0]
    assert rev.status == "cancelled" and rev.record["duration_s"] < 10
    assert s.stopped == "cancelled by the user"


def test_feedback_and_replan(tmp_path):
    s, prov = make(tmp_path, PLAN, gen(), review(), PLAN, gen(summary="red"), review())
    s.run()
    assert s.done and len(s.revisions) == 1
    s.feedback("make the post taller", replan=True)
    assert not s.done
    s.run()
    assert schemas(prov) == ["plan", "generate", "review", "plan", "generate", "review"]
    assert "make the post taller" in prov.calls[3]["messages"][0]["content"]
    assert "make the post taller" in prov.calls[4]["messages"][0]["content"]
    assert s.state["plan_version"] == 2 and (s.dir / "plans" / "plan-002.json").is_file()
    assert s.revisions[1].record["feedback"] == ["make the post taller"] and s.state["feedback"] == []
    assert s.best.name == "rev-002"  # it saw the feedback, rev-001 did not


def test_feedback_waits_if_the_revision_is_cancelled_before_generating(tmp_path):
    s, prov = make(tmp_path, PLAN)
    s.feedback("thinner")
    s.cancel()
    s.step()
    assert s.revisions[0].status == "cancelled" and s.state["feedback"][0]["text"] == "thinner"


def test_invalid_revisions_never_become_best_and_pinning(tmp_path):
    s, _ = make(tmp_path, PLAN, gen(), review(6, "revise"), gen(BROKEN), gen(), review(9))
    s.run()
    assert [r.status for r in s.revisions] == ["ok", "build_error", "ok"]
    assert s.best.name == "rev-003"
    with pytest.raises(ValueError, match="not valid"):
        s.pin_best(2)
    s.pin_best("rev-001")
    assert s.best.name == "rev-001" and (s.dir / "best" / "BEST").read_text().strip() == "rev-001"
    s.pin_best(None)
    assert s.best.name == "rev-003"


def test_resume_continues_numbering_and_tokens(tmp_path):
    s = Session(DEMO_PROMPT, tmp_path / "run", agent=demo_agent(), progress=False).run()
    tokens = s.tokens
    with pytest.raises(ValueError, match="another prompt"):
        Session("something else", tmp_path / "run", agent=demo_agent())
    again = Session.open(tmp_path / "run", agent=ModelAgent(DemoProvider()), progress=False)
    assert again.run().stopped.startswith("done") and len(again.revisions) == 2  # nothing to do
    again.feedback("make the seat red")
    again.run()
    assert [r.name for r in again.revisions] == ["rev-001", "rev-002", "rev-003"]
    assert again.tokens > tokens and again.best.name == "rev-003"


def test_revisions_are_write_once(tmp_path):
    s, _ = make(tmp_path, PLAN, gen(), review())
    s.run()
    with pytest.raises(RevisionSealed):
        s.revisions[0].seal({"status": "rewritten"})
    assert Revision(s.dir / "rev-001").status == "ok"


def test_interrupt_saves_state(tmp_path):
    def interrupt(call):
        raise KeyboardInterrupt
    s, _ = make(tmp_path, PLAN, interrupt)
    s.run()
    assert s.stopped.startswith("interrupted")
    assert s.revisions[0].status == "cancelled" and s.revisions[0].sealed
    assert json.loads((s.dir / "run.json").read_text())["status"] == "stopped"


def test_approval_give_up_and_provider_failure(tmp_path):
    seen = []
    s, _ = make(tmp_path, PLAN, gen(), approve=lambda info: seen.append(info) or False)
    s.run()
    assert s.revisions[0].status == "declined" and s.stopped.startswith("declined")
    assert seen[0]["source"] == TWO_PART.strip() + "\n" and not (s.revisions[0].path / "execution.json").exists()

    s, _ = make(tmp_path, PLAN, gen(), review(2, "give_up"), name="give_up")
    assert s.run().stopped.startswith("the reviewer gave up")

    s, _ = make(tmp_path, PLAN, name="failure")  # the script runs dry at generate: a provider error
    s.run()
    assert s.revisions[0].status == "agent_error" and "model call failed during generate" in s.stopped


def test_on_iteration_feedback_and_stop(tmp_path):
    calls = []

    def hook(rev):
        calls.append(rev.name)
        return "wider base" if len(calls) == 1 else False
    s, prov = make(tmp_path, PLAN, gen(), review(), gen(), review(), on_iteration=hook)
    s.run()
    assert calls == ["rev-001", "rev-002"] and s.stopped == "stopped by on_iteration"
    assert "wider base" in prov.calls[3]["messages"][0]["content"]


def test_table_and_report(tmp_path):
    s, _ = make(tmp_path, PLAN, gen(), review())
    s.run()
    t = s.table()
    rows = t if isinstance(t, list) else t.reset_index().to_dict("records")
    assert rows[0]["name"] == "rev-001" and rows[0]["best"]
    assert "best: rev-001" in s.report()
