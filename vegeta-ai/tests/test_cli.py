from vegeta.ai import cli


def test_check_without_key_reports_and_does_not_fail(monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    assert cli.main(["check"]) == 0
    assert "API key missing" in capsys.readouterr().out
    assert cli.main(["check", "--call"]) == 2
    assert "no API key" in capsys.readouterr().out


def test_models_lists_prices(capsys):
    assert cli.main(["models"]) == 0
    assert "claude-opus-5" in capsys.readouterr().out
