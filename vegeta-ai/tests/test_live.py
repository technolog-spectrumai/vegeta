"""Calls the real API; skipped without ANTHROPIC_API_KEY."""
import pytest

from vegeta.ai import ClaudeProvider, ProviderConfig


@pytest.mark.requires_api_key
def test_live_structured_call():
    schema = {"type": "object", "properties": {"sum": {"type": "integer"}}, "required": ["sum"], "additionalProperties": False}
    r = ClaudeProvider(ProviderConfig(effort="low", max_tokens=2000)).call("You add numbers.", [{"role": "user", "content": "2 + 3?"}], schema=schema)
    assert r.data == {"sum": 5} and r.usage.tokens > 0
