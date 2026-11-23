import pytest

from vegeta.fidia import Limits, Session, agent_from_config


@pytest.mark.requires_api_key
@pytest.mark.slow
def test_live_prompt_to_3d(tmp_path):
    agent = agent_from_config("claude-opus-5", "medium", review_effort="low", max_tokens=12000)
    s = Session("a simple coffee mug with a handle", tmp_path / "mug", agent=agent,
                limits=Limits(max_iterations=3, max_minutes=15, max_tokens=150_000), progress=False).run()
    assert s.revisions and s.plan and s.plan["parts"]
    assert s.tokens > 0 and s.tokens <= 150_000
    if s.best is not None:
        assert s.best.valid and (s.dir / "best" / "export" / "model.glb").is_file()
