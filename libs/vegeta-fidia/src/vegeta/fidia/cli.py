"""``fidia`` / ``vegeta fidia``: prompt-to-3D (``run``, ``resume``, ``show``) and the design copilot (``propose``,
``accept``).

    fidia run "a small stool with three legs" --offline      # scripted demo agent, no API key
    fidia run "a desk lamp" --out runs/lamp --max-iterations 4 --confirm
    fidia resume runs/lamp --feedback "make the shade red"
    fidia show runs/lamp
"""
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


def make_agent(args):
    """The prompt-to-3D agent from the CLI arguments: the offline demo, or Claude (replaced in tests)."""
    if args.offline:
        from .demo import demo_agent

        return demo_agent()
    from .agent import agent_from_config

    return agent_from_config(args.model, args.effort, api_key=args.api_key, review_effort=args.review_effort,
                             max_tokens=args.max_tokens, log=args.transcript)


def _slug(prompt: str) -> str:
    import re

    return re.sub(r"[^a-z0-9]+", "-", prompt.lower()).strip("-")[:40] or "run"


def _session_options(p, resume: bool = False):
    from .session import Limits

    d = Limits()
    p.add_argument("--offline", action="store_true", help="use the scripted demo agent (no API key, no cost)")
    p.add_argument("--model", default="claude-opus-5")
    p.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    p.add_argument("--review-effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"])
    p.add_argument("--api-key", default=None, help="default: ANTHROPIC_API_KEY")
    p.add_argument("--max-tokens", type=int, default=16000, help="per model call")
    p.add_argument("--transcript", default=None, help="append every model call to this JSONL file")
    p.add_argument("--max-iterations", type=int, default=None if resume else d.max_iterations)
    p.add_argument("--max-minutes", type=float, default=None if resume else d.max_minutes)
    p.add_argument("--token-budget", type=int, default=None if resume else d.max_tokens, help="for the whole run")
    p.add_argument("--accept-score", type=int, default=None if resume else d.accept_score)
    p.add_argument("--timeout", type=float, default=120.0, help="seconds per build in the sandbox")
    p.add_argument("--memory-mb", type=int, default=4096, help="memory cap per build")
    p.add_argument("--confirm", action="store_true", help="ask before running each generated file")
    p.add_argument("--feedback", action="append", default=[], help="feedback for the next revision (repeatable)")
    p.add_argument("--replan", action="store_true", help="re-plan with the feedback first")
    p.add_argument("--quiet", action="store_true")


def _limits(args, base=None):
    from dataclasses import asdict

    from .session import Limits

    cur = asdict(base or Limits())
    for key, arg in (("max_iterations", "max_iterations"), ("max_minutes", "max_minutes"),
                     ("max_tokens", "token_budget"), ("accept_score", "accept_score")):
        if getattr(args, arg) is not None:
            cur[key] = getattr(args, arg)
    return Limits(**cur)


def _print_result(session) -> int:
    print(session.report())
    best = session.best
    if best is not None:
        for fmt, path in best.files().items():
            print(f"  {fmt:5s} {path}")
        print(f"  copy of the best revision: {session.best_dir}")
    return 0 if best is not None else 1


def _run(args, prog) -> int:
    from .sandbox import Sandbox
    from .session import Limits, Session

    out = Path(args.out) if args.out else Path("fidia_runs") / _slug(args.prompt)
    if args.offline and not args.quiet:
        print("offline: the scripted demo agent answers (it always builds the demo stool)")
    session = Session(args.prompt, out, agent=make_agent(args), limits=_limits(args, Limits()),
                      sandbox=Sandbox(timeout_s=args.timeout, memory_mb=args.memory_mb),
                      approve="ask" if args.confirm else "auto", progress=not args.quiet)
    for text in args.feedback:
        session.feedback(text, replan=args.replan)
    session.run()
    return _print_result(session)


def _resume(args, prog) -> int:
    import json

    from .sandbox import Sandbox
    from .session import Limits, Session

    state = json.loads((Path(args.run_dir) / "run.json").read_text())
    session = Session.open(args.run_dir, agent=make_agent(args), limits=_limits(args, Limits(**state["limits"])),
                           sandbox=Sandbox(timeout_s=args.timeout, memory_mb=args.memory_mb),
                           approve="ask" if args.confirm else "auto", progress=not args.quiet)
    for text in args.feedback:
        session.feedback(text, replan=args.replan)
    session.run()
    return _print_result(session)


def _show(args, prog) -> int:
    import json

    from .revision import Revision

    run = Path(args.run_dir)
    state = json.loads((run / "run.json").read_text())
    if args.json:
        print(json.dumps(state, indent=2))
        return 0
    print(f"prompt: {state['prompt']}\nstatus: {state['status']} ({state.get('stopped')})\n"
          f"best: {state['best']}  tokens: {state['usage']['tokens']}")
    for row in state["revisions"]:
        print(Revision(run / row["name"]).text())
    return 0


def _add_prompt_commands(sub) -> None:
    """Prompt-to-3D commands: run, resume, show."""
    r = sub.add_parser("run", help="prompt-to-3D: plan, generate, build, check, render, review, revise")
    r.add_argument("prompt")
    r.add_argument("--out", default=None, help="run directory (default fidia_runs/<prompt>); an existing run of the "
                                              "same prompt is resumed")
    _session_options(r)
    r.set_defaults(func=_run)
    c = sub.add_parser("resume", help="continue a run (optionally with --feedback)")
    c.add_argument("run_dir")
    _session_options(c, resume=True)
    c.set_defaults(func=_resume)
    s = sub.add_parser("show", help="print a run's revisions, checks and reviews")
    s.add_argument("run_dir")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=_show)


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
