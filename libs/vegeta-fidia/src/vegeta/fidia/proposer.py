"""The copilot's side of the conversation: the CAD system prompt, the design-state message, and
``ProviderProposer``, which turns any ``vegeta.ai`` provider into a ``Proposer`` for ``DesignSession`` and
``Campaign``. The connection itself (keys, model, retries, images, cancellation) is ``vegeta.ai``'s job."""
from __future__ import annotations

import json

from .proposals import PROPOSAL_SCHEMA

SYSTEM_PROMPT = """You are a CAD copilot for an engineer using Dedalus, a thin layer over CadQuery.

Rules you must follow:
- A design is a Python file with a class deriving from vegeta.dedalus.Design: a `parameters` list of
  Parameter(name, default, units, min, max, description) and a `build(self, p)` method that returns a
  CadQuery Workplane or Shape. Keep this contract. Do not add file, network or subprocess access.
- When you change the source, return the COMPLETE new file content in `source` (kind="source").
  Keep existing parameter names and meanings unless the engineer asks to change them; new parameters
  need sensible defaults, units and ranges. Keep the class name.
- When only parameter values should change, use kind="parameters" with `source` = null.
- If the request is unclear or would change the engineering intent (loads, materials, function),
  use kind="answer" and explain or ask instead of guessing.
- Lengths are millimetres. Prefer simple, robust CadQuery operations (box, cylinder, extrude, cut,
  hole, fillet with modest radii). Fillets must be smaller than adjacent feature sizes.
- Be concrete in `expected_effects` (what the engineer should see in volume/dimensions) and honest
  in `risks` (what could fail to build or what you are unsure about).
The engineer decides; you propose."""


class ProviderProposer:
    """A ``Proposer`` over a ``vegeta.ai`` provider (``ClaudeProvider`` live, ``ScriptedProvider`` offline):
    one structured call per proposal, the last ``history_turns`` turns replayed, ``extra_system`` appended."""

    def __init__(self, provider, *, extra_system: str = "", history_turns: int = 6, effort: str | None = None):
        self.provider = provider
        self.extra_system = extra_system
        self.history_turns = history_turns
        self.effort = effort
        self.name = provider.describe().get("provider", "provider")

    def describe(self) -> dict:
        return self.provider.describe()

    def propose(self, context: dict, instruction: str, history: list[dict]) -> tuple[dict, dict]:
        messages = []
        for turn in history[-self.history_turns:]:
            messages.append({"role": "user", "content": turn["user"]})
            messages.append({"role": "assistant", "content": turn["assistant"]})
        messages.append({"role": "user", "content": user_message(context, instruction)})
        system = SYSTEM_PROMPT + ("\n\n" + self.extra_system if self.extra_system else "")
        reply = self.provider.call(system, messages, schema=PROPOSAL_SCHEMA, effort=self.effort)
        u = reply.usage
        return reply.data, {"input_tokens": u.input_tokens, "output_tokens": u.output_tokens, "model": reply.model}


def user_message(context: dict, instruction: str) -> str:
    """The design state and the instruction as the model reads them."""
    parts = [
        "## Current design file",
        f"path: {context['path']}",
        "```python",
        context["source"],
        "```",
        "## Parameters (name: value, units, range)",
    ]
    for p in context["parameters"]:
        rng = f"[{p.get('min')}, {p.get('max')}]"
        parts.append(f"- {p['name']}: {p['value']} {p.get('units') or ''} {rng} {p.get('description') or ''}".rstrip())
    if context.get("measurements"):
        parts.append("## Measurements of the current geometry (mm, mm^2, mm^3)")
        parts.append(json.dumps(context["measurements"], indent=None, default=str))
    if context.get("notes"):
        parts.append("## Engineer's notes")
        parts.append(context["notes"])
    parts += ["## Instruction", instruction]
    return "\n".join(parts)
