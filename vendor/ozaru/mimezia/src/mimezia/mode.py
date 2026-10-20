"""The mode switch: ``mock`` (stubs), ``off`` (launch nothing, read what exists), ``real`` (the real API)."""
from __future__ import annotations

import os
from dataclasses import dataclass

MOCK, OFF, REAL = "mock", "off", "real"


@dataclass(frozen=True)
class Mode:
    """``Mode("ALABASTRON_SOLVER", default="mock")``. ``resolve(value)``: the environment variable when it is set, else
    ``value``, else the default; validated and exported to the environment so child processes see the same mode."""
    env: str
    default: str = MOCK
    choices: tuple = (MOCK, OFF, REAL)

    def current(self, fallback: str | None = None) -> str:
        """The mode in force now, without exporting it (``fallback`` when the variable is unset)."""
        v = (os.environ.get(self.env) or fallback or self.default).strip().lower()
        if v not in self.choices:
            raise ValueError(f"{self.env} is one of {self.choices}, not {v!r}")
        return v

    def resolve(self, value: str | None = None) -> str:
        v = self.current(value)
        os.environ[self.env] = v
        return v

    def is_mock(self, fallback: str | None = None) -> bool:
        return self.current(fallback) == MOCK

    @staticmethod
    def launches(value: str) -> bool:
        """Whether a program in this mode may produce what it is missing (``off`` never launches)."""
        return value in (MOCK, REAL)
