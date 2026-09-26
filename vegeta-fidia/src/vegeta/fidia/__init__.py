"""vegeta.fidia — AI modelling for Vegeta.

- Prompt to 3D: ``Session(prompt, out_dir, agent=...)`` plans, writes CadQuery (a Dedalus design), runs it in a
  sandboxed subprocess, checks and renders the result, lets the model review the renders, and revises — bounded
  by iterations, time and tokens, with user feedback and cancellation, every revision on disk, the best valid one
  kept, exported as glTF/GLB and OBJ.
- The design copilot (``DesignSession``) and bounded parameter campaigns (``Campaign``).

The AI provider connection is ``vegeta.ai``. Heavy parts (pyvista, vegeta.core) are imported only when used.
"""
from .campaign import Analysis, Budget, Campaign, Criterion, Objective
from .copilot import DesignSession, Proposer
from .proposals import PROPOSAL_SCHEMA, Proposal, Validation
from .proposer import ProviderProposer

__version__ = "0.1.0"
__all__ = ["PROPOSAL_SCHEMA", "Analysis", "Budget", "Campaign", "Criterion", "DesignSession", "Objective", "Proposal",
           "Proposer", "ProviderProposer", "Validation"]
