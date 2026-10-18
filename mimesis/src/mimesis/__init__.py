"""Mimesis — imitation of expensive APIs for end-to-end testing before the long runs.

A program that calls a slow or heavy API (a finite-element solver, an external tool) declares a **mode**
(``mock`` | ``off`` | ``real``) and registers **mimics**: stubs that return what the real API returns — the same
keys, types and shapes — filled with dumb but plausible numbers in milliseconds. Everything around the call stays
real, so whole pipelines and notebooks run against the stubs. Mimesis keeps the stubs honest:

- ``Mode``        the switch (an environment variable overrides the program's default; exported to child processes)
- ``mimic``/``call``  the registry: a name per real API, the real function or the stub chosen by the mode
- ``label``       every stub result says it is one (``"mock": True``, the mimic, a note)
- ``isolate``     mock outputs go to separate places (``runs`` → ``runs-mock``) and never mix with real ones
- ``schema``      a real result's shape recorded once; each stub checked against it (drift fails a test)
- ``banner``      a loud notice in a notebook or a terminal saying which mode is in force
- ``interpreter`` the external interpreter a real API needs, or this one under mock

Standard library only.
"""
from .banner import banner
from .interpreter import interpreter
from .isolate import isolate_env, isolated
from .label import NOTE, is_mock, label
from .mode import Mode
from .registry import MimicError, available, call, get, mimic
from .schema import SchemaError, check, record_schema, schema_of

__version__ = "0.1.0"
__all__ = ["Mode", "mimic", "call", "get", "available", "MimicError", "label", "is_mock", "NOTE", "isolated", "isolate_env",
           "schema_of", "record_schema", "check", "SchemaError", "banner", "interpreter", "__version__"]
