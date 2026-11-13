import pytest

from vegeta.ai import ScriptedProvider
from vegeta.fidia.agent import ModelAgent, generate_message, normalize_plan, normalize_review
from vegeta.fidia.schemas import GENERATION_SCHEMA, PLAN_SCHEMA, REVIEW_SCHEMA

PLAN = {"object": "stool", "description": "d", "size_mm": [300, 300, 450], "floating_ok": False, "assumptions": "",
        "parts": [{"name": "Seat Top", "shape": "disc", "color": [2, 0.5, -1], "connects_to": ["Leg 1"]},
                  {"name": "seat top", "shape": "disc", "color": [0.1], "connects_to": []}],
        "acceptance": ["a", "b"]}


def test_schemas_are_strict():
    for schema in (PLAN_SCHEMA, GENERATION_SCHEMA, REVIEW_SCHEMA):
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"])


def test_normalize_plan():
    plan = normalize_plan(PLAN)
    assert [p["name"] for p in plan["parts"]] == ["seat_top", "seat_top_2"]
    assert plan["parts"][0]["color"] == [1.0, 0.5, 0.0] and plan["parts"][1]["color"] == [0.6, 0.6, 0.6]
    assert plan["parts"][0]["connects_to"] == ["leg_1"]
    assert normalize_plan({**PLAN, "size_mm": [1, -2, 3]})["size_mm"] is None


def test_normalize_review_aligns_acceptance():
    r = normalize_review({"score": 14.6, "verdict": "great", "acceptance": [{"item": "a", "ok": True, "note": ""}],
                          "issues": [], "summary": "s", "next_instruction": ""}, normalize_plan(PLAN))
    assert r["score"] == 10 and r["verdict"] == "revise"
    assert [a["ok"] for a in r["acceptance"]] == [True, False] and r["acceptance"][1]["note"] == "not answered"


def test_roles_call_the_provider_with_schemas_and_image():
    fence = "```python\nimport cadquery as cq\n```"
    review = {"score": 9, "verdict": "accept", "summary": "ok", "acceptance": [], "issues": [], "next_instruction": ""}
    prov = ScriptedProvider(PLAN, {"summary": "s", "addresses": [], "source": fence}, review)
    agent = ModelAgent(prov, review_effort="low", max_tokens=5000)
    plan = agent.plan("a stool", feedback=["taller"]).data
    assert prov.calls[0]["schema"] is PLAN_SCHEMA and "taller" in prov.calls[0]["messages"][0]["content"]
    gen = agent.generate({"prompt": "a stool", "plan": plan, "revision": 1, "feedback": [], "previous": None,
                          "best": None}).data
    assert gen["source"] == "import cadquery as cq\n"  # the fence is stripped
    png = b"\x89PNG\r\n\x1a\n" + b"0" * 20
    brief = {"prompt": "a stool", "plan": plan, "revision": 1, "feedback": [], "problems": [],
             "measured": {"size_mm": [1, 2, 3], "parts": ["seat_top"], "triangles": 12}}
    agent.review(brief, png)
    last = prov.calls[-1]
    assert last["schema"] is REVIEW_SCHEMA and last["images"] == [png] and last["effort"] == "low"
    assert last["max_tokens"] == 5000 and agent.reserve == 5000
    assert "1. a" in last["messages"][0]["content"]


def test_generate_message_carries_errors_review_and_feedback():
    brief = {"prompt": "p", "plan": {"parts": []}, "revision": 3, "feedback": ["make it red"],
             "previous": {"name": "rev-002", "why": "", "status": "build_error", "error": "ValueError: boom",
                          "traceback": "line 9", "problems": ["WARN floating: x"], "source": "SRC",
                          "review": {"score": 4, "verdict": "revise", "summary": "meh", "next_instruction": "fix",
                                     "issues": [{"severity": "major", "part": "leg", "problem": "short", "fix": "longer"}],
                                     "acceptance": [{"item": "legs", "ok": False, "note": "n"}]}},
             "best": {"name": "rev-001", "score": 6}}
    text = generate_message(brief)
    for piece in ("make it red", "ValueError: boom", "line 9", "WARN floating", "SRC", "short -> longer",
                  "acceptance not met: legs", "rev-001", "revision 3"):
        assert piece in text, piece
