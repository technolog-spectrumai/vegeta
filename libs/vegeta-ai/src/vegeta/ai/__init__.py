"""vegeta.ai — the connection to an AI provider, and nothing else.

``ClaudeProvider(ProviderConfig(...)).call(system, messages, schema=..., images=..., cancel=...)`` returns a
``Reply`` with JSON ``data`` (or ``text``) and ``Usage``; ``ScriptedProvider`` answers from a script for tests and
offline demos. What to ask and what to do with the answer lives in the tools that use it (``vegeta.fidia``).
"""
from .claude import ClaudeProvider, ProviderConfig
from .provider import (PRICES, Cancelled, Provider, ProviderError, Reply, ScriptedProvider, TranscriptLog, Usage,
                       image_block, media_type, run_cancellable)

__version__ = "0.2.0"
__all__ = ["PRICES", "Cancelled", "ClaudeProvider", "Provider", "ProviderConfig", "ProviderError", "Reply",
           "ScriptedProvider", "TranscriptLog", "Usage", "image_block", "media_type", "run_cancellable"]
