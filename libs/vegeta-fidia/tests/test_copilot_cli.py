import json

from vegeta.fidia import cli
from _helpers import DESIGN, DESIGN_WITH_HOLE, FakeProposer, answer


def test_cli_propose_save_and_accept(plate, tmp_path, monkeypatch, capsys):
    fake = FakeProposer(answer("source", source=DESIGN_WITH_HOLE, summary="hole"))
    monkeypatch.setattr(cli, "make_proposer", lambda args: fake)
    saved = tmp_path / "p.json"
    assert cli.main(["propose", f"{plate}:Plate", "add a hole", "--save", str(saved), "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["kind"] == "source" and out["validation"]["ok"] and plate.read_text() == DESIGN
    assert cli.main(["accept", f"{plate}:Plate", str(saved)]) == 0
    assert plate.read_text() == DESIGN_WITH_HOLE


def test_cli_needs_key(plate, monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    assert cli.main(["propose", f"{plate}:Plate", "x"]) == 2
    assert "no API key" in capsys.readouterr().err
