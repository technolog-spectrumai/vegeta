"""vegeta.core — the Vegeta workbench: workspaces, designs, immutable revisions and evaluations.

Everything is stored as plain files in a workspace directory. Nothing runs automatically: geometry is
generated and each analysis is run only when the engineer calls it; missing analyses show as NOT RUN.
"""
from ._io import ImmutableError
from .design import Design
from .evaluation import Evaluation
from .revision import LABELS, Revision
from .status import FAILED, NOT_RUN, StatusTable
from .workspace import Workspace

__version__ = "0.1.0"

__all__ = ["Design", "Evaluation", "FAILED", "ImmutableError", "LABELS", "NOT_RUN", "Revision", "StatusTable",
           "Workspace"]
