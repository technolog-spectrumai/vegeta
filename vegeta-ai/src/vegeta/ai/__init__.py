"""vegeta.ai — a Claude copilot for Dedalus designs.

The model proposes; Vegeta builds and measures the proposal; the engineer accepts or rejects.
Nothing changes on disk without ``accept()``.
"""
from .claude import ClaudeConfig, ClaudeProposer
from .proposals import PROPOSAL_SCHEMA, Proposal, Validation
from .session import DesignSession, Proposer

__version__ = "0.1.0"
__all__ = ["PROPOSAL_SCHEMA", "ClaudeConfig", "ClaudeProposer", "DesignSession", "Proposal", "Proposer", "Validation"]
