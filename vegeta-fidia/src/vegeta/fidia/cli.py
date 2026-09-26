"""``fidia`` / ``vegeta fidia``: the design copilot (``propose``, ``accept``); the prompt-to-3D commands are
registered in ``_add_prompt_commands`` once available."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .copilot import DesignSession
from .proposals import Proposal, Validation


def _parse_params(items):
    out = {}
    for item in items or []:
        k, _, v = item.partition("=")
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def make_proposer(args):
    """The live copilot proposer from the CLI arguments (replaced in tests)."""
    from vegeta.ai import ClaudeProvider, ProviderConfig

    from .proposer import ProviderProposer

    cfg = ProviderConfig(api_key=args.api_key, model=args.model, effort=args.effort, max_tokens=args.max_tokens)
    if not cfg.has_key():
        raise ValueError("no API key: pass --api-key or set ANTHROPIC_API_KEY")
    return ProviderProposer(ClaudeProvider(cfg))


def _add_ai_options(p):
    p.add_argument("--model", default="claude-opus-5")
    p.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    p.add_argument("--api-key", default=None, help="default: ANTHROPIC_API_KEY")
    p.add_argument("--max-tokens", type=int, default=16000)


def _copilot(args, prog) -> int:
    if args.cmd == "propose":
        session = DesignSession(args.design, make_proposer(args), parameters=_parse_params(args.param), notes=args.notes)
        prop = session.ask(args.instruction)
        if args.save:
            Path(args.save).write_text(json.dumps(prop.to_dict(), indent=2, default=str))
        if args.accept and prop.ok and prop.kind != "answer":
            session.accept(prop, note="accepted from CLI --accept")
            prop.status = "accepted"
        print(json.dumps(prop.to_dict(), indent=2, default=str) if args.json else prop)
        if args.accept and prop.status == "accepted":
            print(f"accepted: {session.path} updated")
        return 0 if prop.ok else 1
    data = json.loads(Path(args.proposal_json).read_text())
    data.pop("diff", None)
    v = data.pop("validation", None)
    prop = Proposal(**data)
    prop.validation = Validation(**v) if v else None
    session = DesignSession(args.design, _NoProposer())
    prop.validation = session.validate(prop)
    session.proposals[prop.id] = prop
    if not prop.ok:
        print(f"{prog}: proposal no longer validates: {prop.validation.messages}", file=sys.stderr)
        return 1
    session.accept(prop, note="accepted from CLI")
    print(f"accepted {prop.id}: {session.path} updated")
    return 0


def build_parser(prog: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog=prog, description="Fidia: AI modelling for Vegeta")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("propose", help="copilot: ask for a change to a design; prints the validated proposal, writes nothing")
    _add_ai_options(p)
    p.add_argument("design", help="path/design.py:Name")
    p.add_argument("instruction")
    p.add_argument("-p", "--param", action="append", metavar="NAME=VALUE", help="current parameter values")
    p.add_argument("--notes", default="", help="context for the model (constraints, intent)")
    p.add_argument("--save", help="write the proposal JSON here (for 'accept' later)")
    p.add_argument("--accept", action="store_true", help="accept immediately if it validates (explicit opt-in)")
    p.add_argument("--json", action="store_true")
    a = sub.add_parser("accept", help="copilot: apply a saved proposal to the design file (backup kept)")
    a.add_argument("design")
    a.add_argument("proposal_json")
    _add_prompt_commands(sub)
    return ap


def _add_prompt_commands(sub) -> None:
    """Prompt-to-3D commands (run, resume, show); filled in by the session module."""


def main(argv=None, prog: str = "fidia") -> int:
    args = build_parser(prog).parse_args(argv)
    try:
        if args.cmd in ("propose", "accept"):
            return _copilot(args, prog)
        return args.func(args, prog)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"{prog}: error: {exc}", file=sys.stderr)
        return 2


class _NoProposer:
    name = "none"

    def propose(self, *a, **k):
        raise RuntimeError("no proposer configured")

    def describe(self):
        return {"provider": "none"}
