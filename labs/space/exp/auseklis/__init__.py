"""Auseklis — a reusable civilian debris-removal servicer for LEO (exp/: a speculative design of the space lab).

Python-first engineering model: ``physics`` (closed-form relations), ``variants`` (three architectures and their
mass / power / Δv budgets), ``mission`` (per-removal ledgers), ``rendezvous`` (relative motion and approach safety),
``capture`` (momentum and contact estimates), ``cad`` (CadQuery design) and ``render`` (matplotlib views).
Heavy solvers (CalculiX, Gmsh, MuJoCo) are only *prepared* in ``external/``; nothing here runs them.
Read ``README.md`` for what is verified, calculated and assumed — and for the requirements that do not close."""
