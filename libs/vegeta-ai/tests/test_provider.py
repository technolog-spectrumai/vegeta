"""The provider layer without the network: request shape, parsing, errors, usage, cancellation, scripting."""
import json
import threading
import time
from types import SimpleNamespace

import pytest

from vegeta.ai import (Cancelled, ClaudeProvider, ProviderConfig, ProviderError, Reply, ScriptedProvider, Usage,
                       image_block, run_cancellable)

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"], "additionalProperties": False}


class FakeMessages:
    def __init__(self, response, delay=0.0):
        self.response, self.delay, self.kwargs = response, delay, None

    def create(self, **kwargs):
        self.kwargs = kwargs
        time.sleep(self.delay)
        return self.response


def response(text='{"ok": true}', stop="end_turn", model="claude-opus-5"):
    return SimpleNamespace(stop_reason=stop, content=[SimpleNamespace(type="thinking"), SimpleNamespace(type="text", text=text)],
                           usage=SimpleNamespace(input_tokens=100, output_tokens=20), model=model,
                           stop_details=SimpleNamespace(category="policy", explanation="no"))


def provider(resp, delay=0.0, **cfg):
    fake = FakeMessages(resp, delay)
    return ClaudeProvider(ProviderConfig(api_key="k", **cfg), client=SimpleNamespace(messages=fake)), fake


def test_structured_call_with_image_request_shape_and_reply():
    p, fake = provider(response())
    r = p.call("sys", [{"role": "user", "content": "look"}], schema=SCHEMA, images=[PNG], effort="medium")
    k = fake.kwargs
    assert k["system"] == "sys" and k["thinking"] == {"type": "adaptive"} and k["model"] == "claude-opus-5"
    assert k["output_config"] == {"effort": "medium", "format": {"type": "json_schema", "schema": SCHEMA}}
    blocks = k["messages"][-1]["content"]
    assert blocks[0]["type"] == "image" and blocks[0]["source"]["media_type"] == "image/png" and blocks[1] == {"type": "text", "text": "look"}
    assert isinstance(r, Reply) and r.data == {"ok": True} and r.usage.tokens == 120 and r.model == "claude-opus-5"


def test_text_call_has_no_format_and_keeps_messages():
    p, fake = provider(response(text="hello"))
    r = p.call("sys", [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}, {"role": "user", "content": "c"}])
    assert "format" not in fake.kwargs["output_config"] and fake.kwargs["output_config"]["effort"] == "high"
    assert [m["content"] for m in fake.kwargs["messages"]] == ["a", "b", "c"] and r.text == "hello" and r.data is None


@pytest.mark.parametrize("resp, match", [(response(stop="refusal"), "declined"), (response(stop="max_tokens"), "max_tokens"),
                                         (response(text="not json"), "not the requested JSON")])
def test_errors_are_provider_errors(resp, match):
    p, _ = provider(resp)
    with pytest.raises(ProviderError, match=match):
        p.call("s", [{"role": "user", "content": "x"}], schema=SCHEMA)


def test_sdk_exceptions_become_provider_errors_and_are_logged(tmp_path):
    class Boom:
        def create(self, **k):
            raise ConnectionError("down")
    p = ClaudeProvider(ProviderConfig(api_key="k"), client=SimpleNamespace(messages=Boom()), log=tmp_path / "t.jsonl")
    with pytest.raises(ProviderError, match="ConnectionError"):
        p.call("s", [{"role": "user", "content": "x"}])
    rec = json.loads((tmp_path / "t.jsonl").read_text())
    assert rec["error"].startswith("ConnectionError") and rec["reply"] is None


def test_cancel_abandons_a_slow_call():
    p, _ = provider(response(), delay=5.0)
    cancel = threading.Event()
    threading.Timer(0.3, cancel.set).start()
    t0 = time.monotonic()
    with pytest.raises(Cancelled):
        p.call("s", [{"role": "user", "content": "x"}], cancel=cancel)
    assert time.monotonic() - t0 < 1.5


def test_run_cancellable_passes_values_and_errors():
    assert run_cancellable(lambda: 3, threading.Event()) == 3
    with pytest.raises(KeyError):
        run_cancellable(lambda: {}["x"], threading.Event())


def test_usage_adds_and_costs():
    u = Usage(1_000_000, 100_000, 1, "claude-opus-5") + Usage(0, 0, 1, "claude-opus-5")
    assert u.calls == 2 and u.tokens == 1_100_000 and u.cost_usd() == pytest.approx(5.0 + 2.5)
    assert Usage(1, 1, 1, "unknown").cost_usd() is None


def test_key_lookup(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    cfg = ProviderConfig()
    assert not cfg.has_key() and cfg.describe()["key"] == "missing"
    with pytest.raises(ProviderError, match="no API key"):
        cfg.client()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    assert ProviderConfig().has_key()


def test_image_block_checks_the_format():
    assert image_block(b"\xff\xd8\xff" + b"0" * 8)["source"]["media_type"] == "image/jpeg"
    with pytest.raises(ValueError):
        image_block(b"GIF89a")


def test_scripted_provider_records_and_answers(tmp_path):
    s = ScriptedProvider({"ok": True}, "plain", lambda call: {"n": len(call["messages"])}, ValueError("boom"),
                         tokens_per_call=(5, 2), log=tmp_path / "log.jsonl")
    assert s.call("sys", [{"role": "user", "content": "a"}], schema=SCHEMA, images=[PNG]).data == {"ok": True}
    assert s.call("sys", [{"role": "user", "content": "b"}]).text == "plain"
    assert s.call("sys", [{"role": "user", "content": "c"}, {"role": "user", "content": "d"}]).data == {"n": 2}
    with pytest.raises(ValueError):
        s.call("sys", [{"role": "user", "content": "e"}])
    with pytest.raises(ProviderError, match="no answers left"):
        s.call("sys", [{"role": "user", "content": "f"}])
    assert len(s.calls) == 5 and s.calls[0]["images"] == [PNG] and s.calls[0]["schema"] == SCHEMA
    assert len((tmp_path / "log.jsonl").read_text().splitlines()) == 3
    ev = threading.Event(); ev.set()
    with pytest.raises(Cancelled):
        ScriptedProvider({"a": 1}).call("s", [], cancel=ev)
