"""The JSON the model must return at each stage (structured output: every field required, no extras).

Constraints the API does not enforce (list lengths, number ranges) are checked in ``agent.py``.
"""
from __future__ import annotations

_STR = {"type": "string"}
_STRS = {"type": "array", "items": _STR}

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "object": {"type": "string", "description": "what is modelled, in a few words"},
        "description": {"type": "string", "description": "what it should look like: proportions, style, key features"},
        "size_mm": {"type": "array", "items": {"type": "number"},
                    "description": "overall bounding box [x (width), y (depth), z (height)] in millimetres, Z up"},
        "parts": {
            "type": "array",
            "description": "the named parts, 1 to 12; every part a closed solid",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "short snake_case name, used in the code and files"},
                    "shape": {"type": "string", "description": "how to build it from primitives, with dimensions in mm"},
                    "color": {"type": "array", "items": {"type": "number"}, "description": "[r, g, b] in 0..1"},
                    "connects_to": {"type": "array", "items": _STR, "description": "names of the parts it touches"},
                },
                "required": ["name", "shape", "color", "connects_to"],
                "additionalProperties": False,
            },
        },
        "floating_ok": {"type": "boolean", "description": "true only if some parts are meant to be separate"},
        "acceptance": {"type": "array", "items": _STR,
                       "description": "3 to 6 checks a reviewer can confirm from the rendered views"},
        "assumptions": {"type": "string", "description": "what you assumed where the prompt was not specific"},
    },
    "required": ["object", "description", "size_mm", "parts", "floating_ok", "acceptance", "assumptions"],
    "additionalProperties": False,
}

GENERATION_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string", "description": "one sentence: what this version builds or changes"},
        "addresses": {"type": "array", "items": _STR, "description": "the problems from the last revision this fixes"},
        "source": {"type": "string", "description": "the COMPLETE design file (Python), following the contract"},
    },
    "required": ["summary", "addresses", "source"],
    "additionalProperties": False,
}

REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer", "description": "0 (unrelated) to 10 (exactly what was asked, well made)"},
        "verdict": {"type": "string", "enum": ["accept", "revise", "give_up"],
                    "description": "accept: good enough to hand over; revise: fixable issues; give_up: cannot be done"},
        "summary": {"type": "string", "description": "one or two sentences on what you see"},
        "acceptance": {
            "type": "array",
            "description": "one entry per acceptance check of the plan, in order",
            "items": {"type": "object",
                      "properties": {"item": _STR, "ok": {"type": "boolean"}, "note": _STR},
                      "required": ["item", "ok", "note"], "additionalProperties": False},
        },
        "issues": {
            "type": "array",
            "items": {"type": "object",
                      "properties": {"severity": {"type": "string", "enum": ["major", "minor"]},
                                     "part": {"type": "string", "description": "part name, or 'model'"},
                                     "problem": _STR,
                                     "fix": {"type": "string", "description": "concrete change: part, dimension, direction"}},
                      "required": ["severity", "part", "problem", "fix"], "additionalProperties": False},
        },
        "next_instruction": {"type": "string", "description": "what the next revision should change (empty if accept)"},
    },
    "required": ["score", "verdict", "summary", "acceptance", "issues", "next_instruction"],
    "additionalProperties": False,
}
