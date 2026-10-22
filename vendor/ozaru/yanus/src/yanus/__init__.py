"""Yanus — is what a piece of work needs on this machine there? Requirement checks, nothing more.

    from yanus import AnyOf, Env, Package, Tool, check_needs, table
    rows = check_needs([Tool("ccx", version_cmd=("ccx", "-v")), Package("numpy"), Env("HOME"),
                        AnyOf("section backend", Package("neuralfoil"), Tool("xfoil"))])
    print(table(rows))

A tool is found on the PATH (or by a ``probe`` callable) and asked its version with a short timeout; a package is
found, not imported; an environment variable must be set. Yanus never installs, launches or decides anything else —
what a missing need means is the caller's business. Standard library only.
"""
from .needs import AnyOf, Env, Need, NeedResult, Package, Tool, check_need, check_needs, clear_cache, table

__all__ = ["AnyOf", "Env", "Need", "NeedResult", "Package", "Tool", "check_need", "check_needs", "clear_cache", "table"]
__version__ = "0.1.0"
