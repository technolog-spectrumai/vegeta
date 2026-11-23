"""The provider-neutral part: what a call returns, how usage is counted, the ``Provider`` protocol, and a
scripted provider for tests and offline demos. Nothing here knows about geometry or engineering."""
from __future__ import annotations

import base64
import json
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

# USD per million tokens (input, output) — list prices, for a cost estimate only; extend as needed.
PRICES = {
    "claude-opus-5": (5.0, 25.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-haiku-4-5": (1.0, 5.0),
}


class ProviderError(RuntimeError):
    """The provider could not produce a usable answer (refusal, cut off, not JSON, no key...)."""


class Cancelled(RuntimeError):
    """The call was abandoned because its ``cancel`` event was set."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    model: str = ""

    @property
    def tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(self.input_tokens + other.input_tokens, self.output_tokens + other.output_tokens,
                     self.calls + other.calls, other.model or self.model)

    def cost_usd(self, prices: dict | None = None) -> float | None:
        """List-price estimate for this model, or None when the model is not in the price table."""
        p = (prices or PRICES).get(self.model)
        return None if p is None else (self.input_tokens * p[0] + self.output_tokens * p[1]) / 1e6

    def to_dict(self) -> dict:
        return {**asdict(self), "tokens": self.tokens, "cost_usd": self.cost_usd()}


@dataclass
class Reply:
    """One answer: ``data`` when a JSON schema was requested, else ``text``."""

    data: Any = None
    text: str = ""
    usage: Usage = field(default_factory=Usage)
    model: str = ""
    stop_reason: str = ""
    duration_s: float = 0.0


class Provider(Protocol):
    """Anything that answers ``call``. ``messages`` are ``{"role": "user"|"assistant", "content": str}``;
    ``images`` (PNG/JPEG bytes) are attached to the last user message; with ``schema`` the answer is JSON
    that matches it (``Reply.data``); ``cancel`` (a ``threading.Event``) abandons the call."""

    def call(self, system: str, messages: Sequence[dict], *, schema: dict | None = None, images: Sequence[bytes] = (),
             effort: str | None = None, max_tokens: int | None = None, cancel: threading.Event | None = None) -> Reply: ...

    def describe(self) -> dict: ...


def media_type(image: bytes) -> str:
    if image[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if image[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    raise ValueError("images must be PNG or JPEG bytes")


def image_block(image: bytes) -> dict:
    return {"type": "image", "source": {"type": "base64", "media_type": media_type(image),
                                        "data": base64.b64encode(image).decode("ascii")}}


def run_cancellable(fn: Callable[[], Any], cancel: threading.Event | None, poll_s: float = 0.2) -> Any:
    """Run ``fn`` in a worker thread; return its value, re-raise its exception, or raise ``Cancelled`` as soon
    as ``cancel`` is set (the worker is abandoned — a blocking HTTP call cannot be interrupted safely)."""
    if cancel is None:
        return fn()
    box: dict = {}

    def work():
        try:
            box["value"] = fn()
        except BaseException as exc:  # handed to the caller
            box["error"] = exc

    t = threading.Thread(target=work, daemon=True, name="vegeta-ai-call")
    t.start()
    while t.is_alive():
        if cancel.is_set():
            raise Cancelled("call abandoned: cancelled")
        t.join(poll_s)
    if "error" in box:
        raise box["error"]
    return box.get("value")


class TranscriptLog:
    """Append every call (system prompt, messages, reply, usage) to a JSONL file; images are recorded by size."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, *, provider: dict, system: str, messages: Sequence[dict], images: Sequence[bytes], schema: dict | None,
              reply: Reply | None, error: str | None = None) -> None:
        rec = {"at": utc_now(), "provider": provider, "system": system,
               "messages": [{"role": m["role"], "content": m["content"] if isinstance(m["content"], str) else "<blocks>"} for m in messages],
               "images": [len(i) for i in images], "schema": bool(schema), "error": error,
               "reply": None if reply is None else {"data": reply.data, "text": reply.text, "stop_reason": reply.stop_reason,
                                                    "usage": reply.usage.to_dict(), "duration_s": reply.duration_s}}
        with open(self.path, "a") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")


class ScriptedProvider:
    """Answers from a script, in order: each entry is the ``data`` dict (or a str for text, an Exception to
    raise, or a callable ``(call) -> dict`` that sees the request). Records every call in ``calls``. For tests
    and offline demos; counts ``tokens_per_call`` input and output tokens per answer."""

    def __init__(self, *answers, tokens_per_call: tuple[int, int] = (1, 1), model: str = "scripted", log: str | Path | None = None):
        self.answers = list(answers)
        self.calls: list[dict] = []
        self.tokens_per_call = tokens_per_call
        self.model = model
        self.log = TranscriptLog(log) if log else None

    def describe(self) -> dict:
        return {"provider": "scripted", "model": self.model, "remaining": len(self.answers)}

    def call(self, system, messages, *, schema=None, images=(), effort=None, max_tokens=None, cancel=None) -> Reply:
        if cancel is not None and cancel.is_set():
            raise Cancelled("call abandoned: cancelled")
        call = {"system": system, "messages": [dict(m) for m in messages], "schema": schema, "images": list(images),
                "effort": effort, "max_tokens": max_tokens}
        self.calls.append(call)
        if not self.answers:
            raise ProviderError("scripted provider has no answers left")
        a = self.answers.pop(0)
        if isinstance(a, BaseException):
            raise a
        if callable(a):
            a = a(call)
        t0 = time.monotonic()
        usage = Usage(self.tokens_per_call[0], self.tokens_per_call[1], 1, self.model)
        reply = Reply(data=a if not isinstance(a, str) else None, text=a if isinstance(a, str) else json.dumps(a),
                      usage=usage, model=self.model, stop_reason="end_turn", duration_s=time.monotonic() - t0)
        if self.log:
            self.log.write(provider=self.describe(), system=system, messages=messages, images=images, schema=schema, reply=reply)
        return reply
