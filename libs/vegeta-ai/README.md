# vegeta-ai

The connection between Vegeta and an AI provider, and nothing else: `ProviderConfig` (key from the
environment, model, effort, limits), `ClaudeProvider.call(...)` (structured JSON output, image input,
cancellable calls, usage and a cost estimate, an optional JSONL transcript) and `ScriptedProvider` for tests
and offline demos. What to ask and what to do with the answers lives in the tools: `vegeta-fidia`.
See `../docs/ai.md`.
