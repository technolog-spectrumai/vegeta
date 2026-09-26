"""``vegeta ai ...``: check the connection to the AI provider; list the models it knows prices for."""
from __future__ import annotations

import argparse
import json

from .claude import ClaudeProvider, ProviderConfig
from .provider import PRICES, ProviderError


def main(argv=None, prog: str = "vegeta ai") -> int:
    ap = argparse.ArgumentParser(prog=prog, description="The AI provider layer: connection check and models")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="show the configuration; with --call, make one tiny call")
    c.add_argument("--model", default=ProviderConfig.model)
    c.add_argument("--api-key", default=None, help="default: ANTHROPIC_API_KEY")
    c.add_argument("--call", action="store_true", help="make one tiny call (costs a few tokens)")
    c.add_argument("--json", action="store_true")
    sub.add_parser("models", help="models with a price in the cost table (USD per million tokens)")
    args = ap.parse_args(argv)
    if args.cmd == "models":
        for model, (pin, pout) in PRICES.items():
            print(f"{model:20s} input {pin:6.2f}  output {pout:6.2f}")
        return 0
    cfg = ProviderConfig(api_key=args.api_key, model=args.model, effort="low", max_tokens=200)
    out = {"config": cfg.describe()}
    code = 0
    if args.call:
        if not cfg.has_key():
            out["call"] = "skipped: no API key (set ANTHROPIC_API_KEY)"
            code = 2
        else:
            try:
                r = ClaudeProvider(cfg).call("Answer with one word.", [{"role": "user", "content": "Say OK."}])
                out["call"] = {"text": r.text.strip()[:40], "model": r.model, "usage": r.usage.to_dict(),
                               "duration_s": round(r.duration_s, 2)}
            except ProviderError as exc:
                out["call"] = f"failed: {exc}"
                code = 1
    if args.json:
        print(json.dumps(out, indent=2, default=str))
    else:
        conf = out["config"]
        print(f"provider {conf['provider']}, model {conf['model']}, API key {conf['key']}")
        if "call" in out:
            print("call:", out["call"])
    return code
