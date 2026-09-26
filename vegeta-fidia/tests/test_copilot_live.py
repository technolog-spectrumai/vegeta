"""Calls the real API; skipped without ANTHROPIC_API_KEY."""
import pytest

from vegeta.ai import ClaudeProvider, ProviderConfig
from vegeta.fidia import DesignSession, ProviderProposer


@pytest.mark.requires_api_key
def test_live_proposal_builds(plate):
    s = DesignSession(f"{plate}:Plate", ProviderProposer(ClaudeProvider(ProviderConfig(effort="low"))))
    prop = s.ask("add a centred through-hole with a new parameter 'hole' (default 4 mm)")
    assert prop.kind in ("source", "answer")
    if prop.kind == "source":
        assert prop.ok, prop.validation.messages
        assert "hole" in prop.validation.parameters_after
