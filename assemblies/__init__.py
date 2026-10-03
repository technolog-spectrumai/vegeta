"""Proven, notebook-free product code: components, workflows and the ``.vida`` assembly files they write.

Run a workflow from the repository root: ``python -m assemblies.workflows.microjet --help``. See ``assemblies/README.md``.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RUNS = ROOT.parent / "runs" / "assemblies"
