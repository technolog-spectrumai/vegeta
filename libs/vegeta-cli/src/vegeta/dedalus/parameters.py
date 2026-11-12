"""Explicit, validated design parameters."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class Parameter:
    """One named design parameter.

    The type of ``default`` fixes the parameter type (int, float, bool or str). ``min``/``max``
    are inclusive bounds; ``choices`` restricts the allowed values.
    """

    name: str
    default: Any
    units: str | None = None
    min: float | None = None
    max: float | None = None
    description: str = ""
    choices: tuple | None = None

    def __post_init__(self):
        if not self.name.isidentifier():
            raise ValueError(f"parameter name must be a Python identifier, got {self.name!r}")
        if not isinstance(self.default, (int, float, bool, str)):
            raise ValueError(f"parameter {self.name!r}: default must be int, float, bool or str")
        self.validate(self.default)

    @property
    def type(self) -> type:
        return type(self.default)

    def validate(self, value: Any) -> Any:
        """Return ``value`` coerced to the parameter type, or raise ``ValueError``."""
        t = self.type
        if t is bool:
            if not isinstance(value, bool):
                raise ValueError(f"parameter {self.name!r} expects bool, got {value!r}")
        elif t in (int, float):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"parameter {self.name!r} expects a number, got {value!r}")
            if t is int and float(value) != int(value):
                raise ValueError(f"parameter {self.name!r} expects an integer, got {value!r}")
            value = t(value)
            if self.min is not None and value < self.min:
                raise ValueError(f"parameter {self.name!r}={value} is below minimum {self.min}")
            if self.max is not None and value > self.max:
                raise ValueError(f"parameter {self.name!r}={value} is above maximum {self.max}")
        elif not isinstance(value, str):
            raise ValueError(f"parameter {self.name!r} expects str, got {value!r}")
        if self.choices is not None and value not in self.choices:
            raise ValueError(f"parameter {self.name!r}={value!r} not in {self.choices}")
        return value

    def parse(self, text: str) -> Any:
        """Parse a command-line string into this parameter's type and validate it."""
        t = self.type
        if t is bool:
            low = text.strip().lower()
            if low not in ("true", "false", "1", "0", "yes", "no"):
                raise ValueError(f"parameter {self.name!r} expects true/false, got {text!r}")
            return self.validate(low in ("true", "1", "yes"))
        if t in (int, float):
            try:
                num = float(text)
            except ValueError:
                raise ValueError(f"parameter {self.name!r} expects a number, got {text!r}") from None
            return self.validate(num)
        return self.validate(text)


class ParameterSet:
    """Ordered collection of parameters with override resolution."""

    def __init__(self, parameters: Iterable[Parameter]):
        self._params: dict[str, Parameter] = {}
        for p in parameters:
            if p.name in self._params:
                raise ValueError(f"duplicate parameter {p.name!r}")
            self._params[p.name] = p

    def __iter__(self):
        return iter(self._params.values())

    def __len__(self):
        return len(self._params)

    def __contains__(self, name: str) -> bool:
        return name in self._params

    def __getitem__(self, name: str) -> Parameter:
        return self._params[name]

    @property
    def defaults(self) -> dict[str, Any]:
        return {p.name: p.default for p in self}

    def resolve(self, overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Defaults updated with validated ``overrides``; unknown names raise ``ValueError``."""
        overrides = dict(overrides or {})
        unknown = sorted(set(overrides) - set(self._params))
        if unknown:
            raise ValueError(f"unknown parameter(s) {unknown}; known: {list(self._params)}")
        values = self.defaults
        for name, value in overrides.items():
            values[name] = self._params[name].validate(value)
        return values

    def table(self) -> list[dict[str, Any]]:
        return [
            {"name": p.name, "default": p.default, "units": p.units, "min": p.min, "max": p.max,
             "description": p.description}
            for p in self
        ]
