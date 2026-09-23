# Vegeta AI — a Claude copilot for Dedalus designs

`vegeta-ai` (`vegeta.ai`) lets you iterate on a CadQuery/Dedalus design with Claude. The model
**proposes** a new version of the design file or new parameter values; Vegeta **builds and measures**
the proposal in a temporary copy, shows the diff and the measured effect; the design file changes only
when **you accept**. Every proposal and decision is logged next to the design file.

Rules (from `docs/philosophy.md`): the AI never changes anything silently; it never decides loads,
materials or boundary conditions; a proposal is data until you accept it.

## Setup
```bash
pip install -e vegeta-ai            # done by install_local.sh
export ANTHROPIC_API_KEY=sk-ant-...  # or pass api_key=... explicitly
```

## Python
```python
from vegeta.ai import ClaudeConfig, ClaudeProposer, DesignSession

claude = ClaudeProposer(ClaudeConfig(
    api_key=None,            # default: ANTHROPIC_API_KEY
    model="claude-opus-5",   # any current Claude model id
    effort="high",           # low | medium | high | xhigh | max  (cost/quality)
    max_tokens=16000,
    extra_system="All parts are FDM-printed in PETG; wall thickness >= 1.6 mm.",  # company rules
))

s = DesignSession("designs/bracket.py:Bracket", claude,
                  parameters={"thickness": 6.0}, notes="clamped at x=-40, 200 N down at x=+40")

p1 = s.ask("add two ribs under the plate to increase bending stiffness without adding more than 15 % mass")
p1                       # summary, rationale, expected effects, risks, VALIDATION (built + measured), diff
s.geometry(p1)           # CadQuery view of the proposal — nothing written yet
s.accept(p1, note="ribs look right")      # writes designs/bracket.py (backup bracket.py.1.bak)

p2 = s.ask("the ribs are too tall, halve their height and round their ends")   # conversation continues
s.reject(p2, "rounding failed to build")
```
What validation checks: the proposed file loads as a `Design`; parameters resolve within their ranges;
the geometry builds and is a valid solid; measurements before/after (volume, area, dimensions,
face/solid counts) are compared; added/removed parameters are listed. A failed validation cannot be
accepted.

`Proposal.kind` is `source` (new file content), `parameters` (values only) or `answer` (no change:
an explanation or a question back to you).

With `vegeta-core`, accept a proposal and then record it as a revision as usual
(`design.new_revision(...)` / `rev.branch(...)`) — the workspace keeps the source snapshot and hash.

## CLI
```bash
vegeta ai propose designs/bracket.py:Bracket "add two stiffening ribs" -p thickness=6 --save p1.json
vegeta ai accept  designs/bracket.py:Bracket p1.json          # re-validates, then writes (backup kept)
vegeta ai --model claude-sonnet-5 --effort medium propose ... # cheaper model for quick iterations
vegeta ai propose ... --accept                                 # explicit one-step accept if it validates
```

## Safety and cost
- Validation **executes the proposed CadQuery code** in this Python process (that is what building a
  design means). Use trusted models and read the diff; the copilot is a programmer, not the engineer.
- Files: `<design>.ai.jsonl` (all proposals/decisions with token usage), `<design>.<n>.bak` backups.
- Cost: a proposal is a few thousand tokens; with Claude Opus 5 ($5 / $25 per MTok) typically a few
  cents. `effort="low"`/`medium` or `claude-sonnet-5` for cheap iterations.
- Structured output: the model must answer with the JSON schema in `vegeta.ai.PROPOSAL_SCHEMA`, so a
  malformed answer is an error, never a silent change.

Notebook: `notebooks/07_ai_design_copilot.ipynb`.
