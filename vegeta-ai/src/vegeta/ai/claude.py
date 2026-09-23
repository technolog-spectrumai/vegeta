"""Claude adapter: turns an instruction plus the current design state into a typed Proposal."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

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


@dataclass
class ClaudeConfig:
    """How to call Claude. The API key comes from the argument or the ANTHROPIC_API_KEY environment
    variable (an external service connection). Everything else is a plain argument."""

    api_key: str | None = None
    model: str = "claude-opus-5"
    effort: str = "high"          # low | medium | high | xhigh | max
    max_tokens: int = 16000
    timeout: float = 600.0
    max_retries: int = 2
    base_url: str | None = None
    extra_system: str = ""        # company rules appended to the system prompt
    history_turns: int = 6        # how many previous proposal turns to keep in context

    def client(self):
        import anthropic

        kwargs: dict[str, Any] = {"timeout": self.timeout, "max_retries": self.max_retries}
        if self.api_key:
            kwargs["api_key"] = self.api_key
        elif not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("ANTHROPIC_AUTH_TOKEN"):
            raise ValueError("no API key: pass ClaudeConfig(api_key=...) or set ANTHROPIC_API_KEY")
        if self.base_url:
            kwargs["base_url"] = self.base_url
        return anthropic.Anthropic(**kwargs)


class ClaudeProposer:
    """Proposer backed by the Anthropic SDK with structured (JSON-schema) output."""

    name = "claude"

    def __init__(self, config: ClaudeConfig | None = None, **kwargs):
        self.config = config or ClaudeConfig(**kwargs)
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = self.config.client()
        return self._client

    def describe(self) -> dict:
        return {"provider": "anthropic", "model": self.config.model, "effort": self.config.effort,
                "max_tokens": self.config.max_tokens}

    def propose(self, context: dict, instruction: str, history: list[dict]) -> tuple[dict, dict]:
        """Return ``(proposal_json, usage)``. ``context`` is the design state built by the session."""
        messages = []
        for turn in history[-self.config.history_turns:]:
            messages.append({"role": "user", "content": turn["user"]})
            messages.append({"role": "assistant", "content": turn["assistant"]})
        messages.append({"role": "user", "content": _user_message(context, instruction)})
        system = SYSTEM_PROMPT + ("\n\n" + self.config.extra_system if self.config.extra_system else "")
        response = self.client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            system=system,
            messages=messages,
            thinking={"type": "adaptive"},
            output_config={"effort": self.config.effort,
                           "format": {"type": "json_schema", "schema": PROPOSAL_SCHEMA}},
        )
        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            raise RuntimeError(f"Claude declined the request ({getattr(details, 'category', None)}): "
                               f"{getattr(details, 'explanation', '')}")
        if response.stop_reason == "max_tokens":
            raise RuntimeError("Claude's answer was cut off by max_tokens; raise ClaudeConfig.max_tokens")
        text = next(b.text for b in response.content if b.type == "text")
        data = json.loads(text)
        usage = {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens,
                 "model": response.model}
        return data, usage


def _user_message(context: dict, instruction: str) -> str:
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
