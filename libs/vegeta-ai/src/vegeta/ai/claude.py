"""Claude through the Anthropic SDK: structured (JSON-schema) or text answers, image input, cancellable calls."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .provider import Cancelled, ProviderError, Reply, TranscriptLog, Usage, image_block, run_cancellable


@dataclass
class ProviderConfig:
    """How to reach Claude. The key comes from the argument or ``ANTHROPIC_API_KEY`` / ``ANTHROPIC_AUTH_TOKEN``
    (an external service connection); everything else is a plain argument."""

    api_key: str | None = None
    model: str = "claude-opus-5"
    effort: str = "high"          # low | medium | high | xhigh | max
    max_tokens: int = 16000
    timeout: float = 600.0
    max_retries: int = 2
    base_url: str | None = None

    def has_key(self) -> bool:
        return bool(self.api_key or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))

    def client(self):
        import anthropic

        kwargs: dict[str, Any] = {"timeout": self.timeout, "max_retries": self.max_retries}
        if self.api_key:
            kwargs["api_key"] = self.api_key
        elif not self.has_key():
            raise ProviderError("no API key: pass ProviderConfig(api_key=...) or set ANTHROPIC_API_KEY")
        if self.base_url:
            kwargs["base_url"] = self.base_url
        return anthropic.Anthropic(**kwargs)

    def describe(self) -> dict:
        return {"provider": "anthropic", "model": self.model, "effort": self.effort, "max_tokens": self.max_tokens,
                "key": "set" if self.has_key() else "missing"}


class ClaudeProvider:
    """``Provider`` backed by Claude. ``log`` appends every call to a JSONL transcript."""

    def __init__(self, config: ProviderConfig | None = None, *, client=None, log: str | Path | None = None, **kwargs):
        self.config = config or ProviderConfig(**kwargs)
        self._client = client
        self.log = TranscriptLog(log) if log else None

    @property
    def client(self):
        if self._client is None:
            self._client = self.config.client()
        return self._client

    def describe(self) -> dict:
        return self.config.describe()

    def request(self, system: str, messages, *, schema=None, images=(), effort=None, max_tokens=None) -> dict:
        """The keyword arguments of ``messages.create`` for this call (images go on the last user message)."""
        msgs = [dict(m) for m in messages]
        if images:
            last = max(i for i, m in enumerate(msgs) if m["role"] == "user")
            content = msgs[last]["content"]
            blocks = [{"type": "text", "text": content}] if isinstance(content, str) else list(content)
            msgs[last]["content"] = [image_block(im) for im in images] + blocks
        output_config: dict[str, Any] = {"effort": effort or self.config.effort}
        if schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": schema}
        return {"model": self.config.model, "max_tokens": max_tokens or self.config.max_tokens, "system": system,
                "messages": msgs, "thinking": {"type": "adaptive"}, "output_config": output_config}

    def call(self, system, messages, *, schema=None, images=(), effort=None, max_tokens=None, cancel=None) -> Reply:
        kwargs = self.request(system, messages, schema=schema, images=images, effort=effort, max_tokens=max_tokens)
        t0 = time.monotonic()
        reply, error = None, None
        try:
            response = run_cancellable(lambda: self.client.messages.create(**kwargs), cancel)
            reply = self._parse(response, schema, time.monotonic() - t0)
            return reply
        except (Cancelled, ProviderError) as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        except Exception as exc:  # SDK errors (auth, rate limit, network): one error type for the caller
            error = f"{type(exc).__name__}: {exc}"
            raise ProviderError(error) from exc
        finally:
            if self.log:
                self.log.write(provider=self.describe(), system=system, messages=messages, images=images, schema=schema,
                               reply=reply, error=error)

    def _parse(self, response, schema, duration: float) -> Reply:
        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            raise ProviderError(f"Claude declined the request ({getattr(details, 'category', None)}): "
                                f"{getattr(details, 'explanation', '')}")
        if response.stop_reason == "max_tokens":
            raise ProviderError("the answer was cut off by max_tokens; raise ProviderConfig.max_tokens")
        text = "".join(b.text for b in response.content if getattr(b, "type", None) == "text")
        usage = Usage(response.usage.input_tokens, response.usage.output_tokens, 1, response.model)
        data = None
        if schema is not None:
            try:
                data = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ProviderError(f"the answer is not the requested JSON: {exc}") from exc
        return Reply(data=data, text=text, usage=usage, model=response.model, stop_reason=response.stop_reason,
                     duration_s=duration)
