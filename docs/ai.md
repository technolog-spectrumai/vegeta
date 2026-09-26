# Vegeta AI — the connection to an AI provider

`vegeta-ai` (`vegeta.ai`) is a thin layer: it connects to an AI provider and handles what surrounds a call —
keys, model, effort, retries, structured JSON output, image input, cancellation, token usage and cost, an optional
transcript. It knows nothing about geometry or engineering and imports nothing else from Vegeta.

What to ask and what to do with the answer lives in the tools that use it: **[Fidia](fidia.md)** (`vegeta-fidia`)
holds the design copilot, parameter campaigns and prompt-to-3D.

## Setup
```bash
pip install -e vegeta-ai             # done by install_local.sh
export ANTHROPIC_API_KEY=sk-ant-...  # or pass api_key=... explicitly
vegeta ai check                      # configuration and key status (no call, no cost)
vegeta ai check --call               # one tiny call (a few tokens); exit 2 without a key
vegeta ai models                     # models in the cost table (USD per million tokens)
```

## Python
```python
from vegeta.ai import ClaudeProvider, ProviderConfig, ScriptedProvider

claude = ClaudeProvider(ProviderConfig(
    api_key=None,            # default: ANTHROPIC_API_KEY
    model="claude-opus-5",   # any current Claude model id
    effort="high",           # low | medium | high | xhigh | max  (cost/quality)
    max_tokens=16000, timeout=600, max_retries=2,
), log="calls.jsonl")        # optional: every call appended as JSON (images recorded by size only)

reply = claude.call(
    "You are ...",                                   # system prompt
    [{"role": "user", "content": "..."}],            # messages
    schema={...},                                    # JSON schema -> reply.data (strict structured output)
    images=[png_bytes],                              # PNG/JPEG, attached to the last user message
    effort="medium",                                 # per call
    cancel=threading.Event(),                        # set it to abandon the call
)
reply.data, reply.text, reply.usage.tokens, reply.usage.cost_usd(), reply.model, reply.stop_reason
```
- **Errors:** anything that is not a usable answer (refusal, cut off at `max_tokens`, invalid JSON, no key,
  network or API errors after retries) raises `ProviderError`; a set `cancel` event raises `Cancelled`.
- **Cancellation:** the call runs in a worker thread that the caller stops waiting for within 0.2 s of the event
  being set (`run_cancellable`); the request itself finishes in the background and is discarded.
- **Usage:** `Usage(input_tokens, output_tokens, calls, model)` adds up with `+`; `cost_usd()` uses list prices in
  `PRICES` (an estimate; `None` for models not in the table).
- **Offline and tests:** `ScriptedProvider(*answers)` answers in order — a dict (JSON data), a string (text), an
  exception (raised) or a function of the call — and records every call (`.calls`: system, messages, schema,
  images, effort). It is how Fidia's tests and its offline demo run without a key.
- **The protocol:** any object with `call(system, messages, *, schema, images, effort, max_tokens, cancel) -> Reply`
  and `describe() -> dict` is a provider. An OpenAI provider is planned (`todo.md` 12.2).

Rules (from `docs/philosophy.md`): the AI never changes anything silently; it never decides loads, materials or
boundary conditions; its answers are data until the engineer, or a deterministic check, accepts them.
