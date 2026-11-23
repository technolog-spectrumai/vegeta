"""The model's three roles in the loop — plan, generate, review — over any ``vegeta.ai`` provider.

``ModelAgent(provider)`` holds the prompts and schemas; the provider (``ClaudeProvider`` live, ``ScriptedProvider``
offline) holds the connection. Every answer is normalised here (names, ranges, list lengths) because structured
output guarantees the shape of the JSON, not its sense. The review is advice: it never makes a revision valid.
"""
from __future__ import annotations

import json
import re
import threading
from typing import Any, Sequence

from .contract import CONTRACT, EXAMPLE_SOURCE
from .schemas import GENERATION_SCHEMA, PLAN_SCHEMA, REVIEW_SCHEMA

PLAN_SYSTEM = """You are the planner of Fidia, a prompt-to-3D tool for engineers. Turn the user's request into a
buildable plan for a CadQuery model: the object's overall size in millimetres (Z up, resting on z = 0), and 1 to 12
named parts, each a closed solid made from primitives, booleans and modest fillets. Give each part a colour that
makes the model readable. Say which parts touch (the model must hang together unless parts are meant to be
separate). Write 3 to 6 acceptance checks that someone can confirm by looking at front, right, top and two iso
views (proportions, counts, arrangement, key features) — not things only a measurement could show.
Prefer a recognisable, well-proportioned simple model over a detailed fragile one. If the request is vague, pick
typical real-world dimensions and say so in `assumptions`."""

GENERATE_SYSTEM = """You are the modeller of Fidia. You write the Python/CadQuery code for a plan, and revise it from
what went wrong: build errors (fix the exact line), failed geometry checks (facts measured by Vegeta), the
reviewer's issues and the user's feedback (the user's feedback comes first).

""" + CONTRACT + """
An example of a valid file:
```python
""" + EXAMPLE_SOURCE + """```
Return the COMPLETE file in `source` every time. Keep what already works; change what the feedback asks for."""

REVIEW_SYSTEM = """You are the reviewer of Fidia. You see one image: five rendered views of a CAD model (front,
right, top, iso, rear iso; orthographic front/right/top) and a legend with the overall size and the part colours.
Judge how well the model matches the request and the plan: shape, proportions, part count and arrangement, colours.
Vegeta's measurements are facts; do not contradict them. Score 0 to 10 (8 = a user would accept it as is). Answer
every acceptance check in order. For each issue give a concrete fix (part, dimension, direction). Use verdict
"accept" only when the score is at least 8, every acceptance check passes and no major issue is left; "give_up"
only if the request cannot be modelled with this tool."""


def _snake(name: str) -> str:
    s = re.sub(r"[^a-z0-9_]+", "_", str(name).strip().lower()).strip("_") or "part"
    return "p_" + s if s[0].isdigit() else s


def normalize_plan(data: dict) -> dict:
    """Sanitised plan: snake_case unique part names, colours in 0..1, a positive size or None, 1-8 checks."""
    plan = dict(data or {})
    size = plan.get("size_mm")
    plan["size_mm"] = ([float(v) for v in size] if isinstance(size, (list, tuple)) and len(size) == 3
                       and all(isinstance(v, (int, float)) and v > 0 for v in size) else None)
    parts, taken = [], set()
    for p in (plan.get("parts") or [])[:12]:
        name = _snake(p.get("name", "part"))
        base, n = name, 2
        while name in taken:
            name, n = f"{base}_{n}", n + 1
        taken.add(name)
        rgb = [min(max(float(c), 0.0), 1.0) for c in (p.get("color") or [])[:3]]
        parts.append({"name": name, "shape": str(p.get("shape", "")), "color": rgb if len(rgb) == 3 else [0.6, 0.6, 0.6],
                      "connects_to": [_snake(c) for c in p.get("connects_to") or []]})
    plan["parts"] = parts
    plan["acceptance"] = [str(a) for a in (plan.get("acceptance") or [])][:8]
    plan["floating_ok"] = bool(plan.get("floating_ok", False))
    for key in ("object", "description", "assumptions"):
        plan[key] = str(plan.get(key) or "")
    return plan


def normalize_review(data: dict, plan: dict) -> dict:
    """Score clamped to 0..10, one acceptance entry per plan check (missing ones count as not ok)."""
    r = dict(data or {})
    try:
        r["score"] = max(0, min(10, int(round(float(r.get("score", 0))))))
    except (TypeError, ValueError):
        r["score"] = 0
    if r.get("verdict") not in ("accept", "revise", "give_up"):
        r["verdict"] = "revise"
    got = list(r.get("acceptance") or [])
    items = plan.get("acceptance") or []
    r["acceptance"] = [{"item": item, "ok": bool(got[i].get("ok")) if i < len(got) else False,
                        "note": str(got[i].get("note", "")) if i < len(got) else "not answered"}
                       for i, item in enumerate(items)]
    r["issues"] = [dict(i) for i in r.get("issues") or []]
    r["summary"] = str(r.get("summary") or "")
    r["next_instruction"] = str(r.get("next_instruction") or "")
    return r


def plan_message(prompt: str, feedback: Sequence[str] = (), previous: dict | None = None) -> str:
    lines = ["## Request", prompt.strip()]
    if previous:
        lines += ["## The current plan (revise it)", json.dumps(previous, indent=1)]
    if feedback:
        lines += ["## The user's feedback (follow it)"] + [f"- {f}" for f in feedback]
    lines += ["## Task", "Write the plan."]
    return "\n".join(lines)


def generate_message(brief: dict) -> str:
    """What the modeller reads: request, plan, feedback, and the previous revision with everything that went wrong."""
    lines = ["## Request", brief["prompt"].strip(), "## Plan", json.dumps(brief["plan"], indent=1)]
    if brief.get("feedback"):
        lines += ["## The user's feedback (most important; follow it)"] + [f"- {f}" for f in brief["feedback"]]
    prev = brief.get("previous")
    if prev:
        lines.append(f"## Previous revision {prev['name']}" + (f" ({prev['why']})" if prev.get("why") else ""))
        lines.append(f"status: {prev['status']}" + (f" — {prev['error']}" if prev.get("error") else ""))
        if prev.get("traceback"):
            lines += ["traceback (last lines):", "```", prev["traceback"], "```"]
        if prev.get("problems"):
            lines += ["Vegeta's geometry checks (facts):"] + [f"- {p}" for p in prev["problems"]]
        rv = prev.get("review")
        if rv:
            lines.append(f"Reviewer: score {rv.get('score')}/10, {rv.get('verdict')}: {rv.get('summary', '')}")
            lines += [f"- [{i.get('severity')}] {i.get('part')}: {i.get('problem')} -> {i.get('fix')}" for i in rv.get("issues", [])]
            lines += [f"- acceptance not met: {a['item']} ({a.get('note', '')})" for a in rv.get("acceptance", []) if not a.get("ok")]
            if rv.get("next_instruction"):
                lines.append(f"Reviewer's instruction: {rv['next_instruction']}")
        lines += ["Its code:", "```python", prev["source"], "```"]
    if brief.get("best") and (not prev or brief["best"]["name"] != prev["name"]):
        lines.append(f"(The best valid revision so far is {brief['best']['name']} with score {brief['best'].get('score')}.)")
    lines += ["## Task", f"Write revision {brief['revision']}: the complete design file."]
    return "\n".join(lines)


def review_message(brief: dict) -> str:
    plan = brief["plan"]
    lines = ["## Request", brief["prompt"].strip(),
             "## Plan", f"object: {plan.get('object')}", f"description: {plan.get('description')}",
             f"planned size (x, y, z mm): {plan.get('size_mm')}",
             "planned parts: " + ", ".join(f"{p['name']} ({p['shape']})" for p in plan.get("parts", [])),
             "## Acceptance checks (answer each, in order)"]
    lines += [f"{i + 1}. {a}" for i, a in enumerate(plan.get("acceptance", []))]
    m = brief["measured"]
    lines += ["## Measured by Vegeta (facts)", f"size (x, y, z mm): {m['size_mm']}", f"parts: {', '.join(m['parts'])}",
              f"triangles: {m['triangles']}"]
    lines += [f"- {p}" for p in brief.get("problems", [])] or ["- all geometry checks passed"]
    if brief.get("feedback"):
        lines += ["## The user's feedback for this revision"] + [f"- {f}" for f in brief["feedback"]]
    lines += ["## Task", f"Review revision {brief['revision']} from the attached image."]
    return "\n".join(lines)


class ModelAgent:
    """Plan, generate and review through ``provider.call`` (structured output; the review sees the contact sheet)."""

    def __init__(self, provider, *, plan_effort: str | None = None, generate_effort: str | None = None,
                 review_effort: str | None = "medium", max_tokens: int | None = None, extra_system: str = ""):
        self.provider = provider
        self.plan_effort, self.generate_effort, self.review_effort = plan_effort, generate_effort, review_effort
        self.max_tokens = max_tokens
        self.extra_system = extra_system

    @property
    def reserve(self) -> int:
        """Tokens one call may use at most (for the budget check before each call)."""
        cfg = getattr(self.provider, "config", None)
        return int(self.max_tokens or getattr(cfg, "max_tokens", 0) or 0)

    def describe(self) -> dict:
        return {"agent": "model", **self.provider.describe(), "efforts": {
            "plan": self.plan_effort, "generate": self.generate_effort, "review": self.review_effort}}

    def _system(self, text: str) -> str:
        return text + ("\n\n" + self.extra_system if self.extra_system else "")

    def plan(self, prompt: str, *, feedback: Sequence[str] = (), previous: dict | None = None,
             cancel: threading.Event | None = None):
        reply = self.provider.call(self._system(PLAN_SYSTEM), [{"role": "user", "content": plan_message(prompt, feedback, previous)}],
                                   schema=PLAN_SCHEMA, effort=self.plan_effort, max_tokens=self.max_tokens, cancel=cancel)
        reply.data = normalize_plan(reply.data)
        return reply

    def generate(self, brief: dict, *, cancel: threading.Event | None = None):
        reply = self.provider.call(self._system(GENERATE_SYSTEM), [{"role": "user", "content": generate_message(brief)}],
                                   schema=GENERATION_SCHEMA, effort=self.generate_effort, max_tokens=self.max_tokens,
                                   cancel=cancel)
        d = dict(reply.data or {})
        reply.data = {"summary": str(d.get("summary", "")), "addresses": [str(a) for a in d.get("addresses") or []],
                      "source": _strip_fences(str(d.get("source", "")))}
        return reply

    def review(self, brief: dict, sheet_png: bytes, *, cancel: threading.Event | None = None):
        reply = self.provider.call(self._system(REVIEW_SYSTEM), [{"role": "user", "content": review_message(brief)}],
                                   schema=REVIEW_SCHEMA, images=[sheet_png], effort=self.review_effort,
                                   max_tokens=self.max_tokens, cancel=cancel)
        reply.data = normalize_review(reply.data, brief["plan"])
        return reply


def _strip_fences(source: str) -> str:
    """The file itself, if the model wrapped it in a Markdown code fence anyway."""
    m = re.match(r"^\s*```(?:python)?\s*\n(.*?)\n```\s*$", source, re.S)
    return (m.group(1) if m else source).strip() + "\n"


def agent_from_config(model: str = "claude-opus-5", effort: str = "high", *, api_key: str | None = None,
                      review_effort: str = "medium", max_tokens: int = 16000, log: str | None = None) -> ModelAgent:
    """A live agent on Claude (raises ``ValueError`` without an API key)."""
    from vegeta.ai import ClaudeProvider, ProviderConfig

    cfg = ProviderConfig(api_key=api_key, model=model, effort=effort, max_tokens=max_tokens)
    if not cfg.has_key():
        raise ValueError("no API key: set ANTHROPIC_API_KEY (or use the offline demo agent)")
    return ModelAgent(ClaudeProvider(cfg, log=log), review_effort=review_effort)


def describe_agent(agent: Any) -> dict:
    try:
        return agent.describe()
    except Exception:  # an agent without describe() still runs
        return {"agent": type(agent).__name__}
