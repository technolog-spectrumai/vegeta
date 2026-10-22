from vegeta.ai import ScriptedProvider
from vegeta.fidia import DesignSession, PROPOSAL_SCHEMA, ProviderProposer
from _helpers import DESIGN_WITH_HOLE, answer


def test_provider_proposer_drives_the_copilot(plate):
    provider = ScriptedProvider(answer("source", source=DESIGN_WITH_HOLE, summary="hole"), tokens_per_call=(30, 7))
    s = DesignSession(f"{plate}:Plate", ProviderProposer(provider, extra_system="House rule: metric."))
    prop = s.ask("add a hole")
    call = provider.calls[0]
    assert prop.ok and prop.kind == "source" and prop.usage == {"input_tokens": 30, "output_tokens": 7, "model": "scripted"}
    assert call["schema"] == PROPOSAL_SCHEMA and call["system"].endswith("House rule: metric.")
    assert "## Instruction\nadd a hole" in call["messages"][-1]["content"]
