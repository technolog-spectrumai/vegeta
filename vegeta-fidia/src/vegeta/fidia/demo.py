"""An offline stand-in for the model, so the loop, the notebook and the CLI run without an API key.

``demo_agent()`` answers from a script through ``vegeta.ai``'s ``ScriptedProvider``: it plans a three-legged stool;
its first revision leaves the legs 20 mm short, so the seat does not touch them (Vegeta's floating-part check
warns, the scripted review asks for a fix); the second revision fixes it and is accepted. It also follows
simple feedback — a seat colour ("make the seat blue"), a leg count ("four legs"), "taller"/"lower" — so the
feedback cell of the notebook works offline. The review reacts to Vegeta's measured checks, not to the pictures.
"""
from __future__ import annotations

import re

from vegeta.ai import ScriptedProvider

from .agent import ModelAgent
from .schemas import GENERATION_SCHEMA, PLAN_SCHEMA, REVIEW_SCHEMA

DEMO_PROMPT = "a small stool with three legs"

COLOURS = {"red": (0.75, 0.15, 0.12), "blue": (0.15, 0.3, 0.7), "green": (0.2, 0.55, 0.25), "black": (0.1, 0.1, 0.1),
           "white": (0.92, 0.92, 0.9), "yellow": (0.9, 0.75, 0.15), "grey": (0.55, 0.55, 0.57), "wood": (0.55, 0.36, 0.2)}
NUMBERS = {"three": 3, "four": 4, "five": 5, "six": 6, "3": 3, "4": 4, "5": 5, "6": 6}

SOURCE = '''import math

import cadquery as cq
from vegeta.dedalus import Design, Parameter


class Model(Design):
    """A small stool: a round seat on {legs} legs joined by a ring."""

    parameters = [
        Parameter("seat_diameter", 300.0, "mm", min=150, description="seat diameter"),
        Parameter("seat_thickness", 25.0, "mm", min=10),
        Parameter("height", {height:.1f}, "mm", min=200, description="floor to the top of the seat"),
        Parameter("leg_diameter", 30.0, "mm", min=10),
        Parameter("legs", {legs}, "", min=3, max=6),
    ]

    def build(self, p):
        r = p["seat_diameter"] / 2 - 40
        leg_length = p["height"] - p["seat_thickness"]{shortfall}
        seat = (cq.Workplane("XY").workplane(offset=p["height"] - p["seat_thickness"])
                .circle(p["seat_diameter"] / 2).extrude(p["seat_thickness"]).edges(">Z").fillet(6))
        assy = cq.Assembly(name="stool")
        assy.add(seat, name="seat", color=cq.Color{seat})
        for i in range(p["legs"]):
            a = 2 * math.pi * i / p["legs"]
            leg = cq.Workplane("XY").center(r * math.cos(a), r * math.sin(a)).circle(p["leg_diameter"] / 2).extrude(leg_length)
            assy.add(leg, name=f"leg_{{i + 1}}", color=cq.Color(0.25, 0.2, 0.18))
        ring = cq.Workplane("XY").workplane(offset=140).circle(r + 6).circle(r - 6).extrude(12)
        assy.add(ring, name="ring", color=cq.Color(0.6, 0.6, 0.62))
        return assy
'''


def _feedback(text: str) -> list[str]:
    m = re.search(r"## The user's feedback[^\n]*\n((?:- [^\n]*\n?)+)", text)
    return [line[2:].strip() for line in m.group(1).splitlines()] if m else []


class DemoProvider(ScriptedProvider):
    """Routes each call by its schema to a scripted planner, modeller or reviewer, and never runs dry."""

    def __init__(self, tokens_per_call=(2500, 1200)):
        super().__init__(tokens_per_call=tokens_per_call, model="fidia-demo")
        self.state = {"legs": 3, "seat": COLOURS["wood"], "height": 450.0, "fixed": False}

    def describe(self) -> dict:
        return {"provider": "demo (scripted, offline)", "model": self.model}

    def call(self, system, messages, *, schema=None, **kwargs):
        route = {id(PLAN_SCHEMA): self._plan, id(GENERATION_SCHEMA): self._generate, id(REVIEW_SCHEMA): self._review}
        self.answers = [route[id(schema)]]
        return super().call(system, messages, schema=schema, **kwargs)

    def _apply(self, feedback: list[str]) -> list[str]:
        done = []
        for f in feedback:
            low = f.lower()
            for word, rgb in COLOURS.items():
                if re.search(rf"\b{word}\b", low):
                    self.state["seat"] = rgb
                    done.append(f"seat colour {word}")
            m = re.search(r"\b(three|four|five|six|[3-6])\s+legs?\b", low)
            if m:
                self.state["legs"] = NUMBERS[m.group(1)]
                done.append(f"{self.state['legs']} legs")
            if re.search(r"\b(taller|higher)\b", low):
                self.state["height"] += 100
                done.append(f"height {self.state['height']:.0f} mm")
            if re.search(r"\b(lower|shorter)\b", low):
                self.state["height"] = max(250.0, self.state["height"] - 100)
                done.append(f"height {self.state['height']:.0f} mm")
        return done

    def _plan(self, call) -> dict:
        self._apply(_feedback(call["messages"][-1]["content"]))
        n, h = self.state["legs"], self.state["height"]
        legs = [{"name": f"leg_{i + 1}", "shape": f"cylinder d30 x {h - 25:.0f}, at radius 110 mm, {360 / n:.0f} deg apart",
                 "color": [0.25, 0.2, 0.18], "connects_to": ["seat", "ring"]} for i in range(n)]
        return {"object": "stool", "description": f"a small round stool on {n} straight legs with a ring between them",
                "size_mm": [300, 300, h],
                "parts": [{"name": "seat", "shape": "disc d300 x 25 with a 6 mm top fillet", "color": list(self.state["seat"]),
                           "connects_to": [leg["name"] for leg in legs]}] + legs
                + [{"name": "ring", "shape": "flat ring r104-116 x 12 at z=140", "color": [0.6, 0.6, 0.62],
                    "connects_to": [leg["name"] for leg in legs]}],
                "floating_ok": False,
                "acceptance": [f"{n} legs, evenly spaced", "a round seat on top of the legs",
                               "a ring joins the legs part-way down", "the legs stand on the floor"],
                "assumptions": f"seat height {h:.0f} mm, seat diameter 300 mm, wooden seat"}

    def _generate(self, call) -> dict:
        text = call["messages"][-1]["content"]
        done = self._apply(_feedback(text))
        first = "## Previous revision" not in text
        if not first:
            self.state["fixed"] = True
        s = self.state
        source = SOURCE.format(legs=s["legs"], height=s["height"], seat=tuple(round(c, 3) for c in s["seat"]),
                               shortfall="" if s["fixed"] else " - 20")
        if first:
            return {"summary": f"seat on {s['legs']} legs with a ring", "addresses": [], "source": source}
        return {"summary": "legs reach the seat" + (f"; {', '.join(done)}" if done else ""),
                "addresses": ["the seat did not touch the legs"] + done, "source": source}

    def _review(self, call) -> dict:
        text = call["messages"][-1]["content"]
        items = re.findall(r"^\d+\. (.+)$", text.split("## Acceptance checks", 1)[-1].split("## Measured", 1)[0], re.M)
        if "not attached" in text:
            return {"score": 5, "verdict": "revise", "summary": "The seat floats above the legs; the proportions are right.",
                    "acceptance": [{"item": a, "ok": "top of the legs" not in a, "note": "seat not on the legs"
                                    if "top of the legs" in a else ""} for a in items],
                    "issues": [{"severity": "major", "part": "seat", "problem": "a 20 mm gap between the seat and the legs",
                                "fix": "make the legs height - seat_thickness long so they reach the seat"}],
                    "next_instruction": "Extend the legs by 20 mm so the seat rests on them."}
        return {"score": 9, "verdict": "accept", "summary": "A clean stool: seat on the legs, ring between them.",
                "acceptance": [{"item": a, "ok": True, "note": ""} for a in items], "issues": [],
                "next_instruction": ""}


def demo_agent() -> ModelAgent:
    """A ``ModelAgent`` on the offline ``DemoProvider``."""
    return ModelAgent(DemoProvider())
