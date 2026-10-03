"""Workflows: one module per product. Each builds its assembly tree, runs the long calculations explicitly (top to
bottom in ``run``), saves the tree as a ``.vida`` in ``assemblies/data`` and writes the exports other code reads.

Run one from the repository root: ``python -m assemblies.workflows.<name> --help``.
"""
