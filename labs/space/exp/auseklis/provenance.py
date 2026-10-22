"""Every number in Auseklis says where it comes from: VERIFIED (a cited public source), CALCULATED (a script here
produces it) or ASSUMED (a design choice or class-typical value, with the reason). ``Tagged`` is a float that carries
its tag, so arithmetic works and the register can list every assumption for the technical specification."""
from __future__ import annotations

VERIFIED, CALCULATED, ASSUMED = "VERIFIED", "CALCULATED", "ASSUMED"
REGISTER: list["Tagged"] = []


class Tagged(float):
    def __new__(cls, value, unit="", tag=ASSUMED, note="", name=""):
        self = super().__new__(cls, value)
        self.unit, self.tag, self.note, self.name = unit, tag, note, name
        REGISTER.append(self)
        return self

    def __repr__(self):
        return f"{float(self):g} {self.unit} [{self.tag}]".replace("  ", " ")


def assumed(value, unit="", note="", name=""):
    return Tagged(value, unit, ASSUMED, note, name)


def verified(value, unit, source, name=""):
    if not source:
        raise ValueError("a VERIFIED value needs its source")
    return Tagged(value, unit, VERIFIED, source, name)


def register_table(tag=None):
    rows = [(t.name or "?", float(t), t.unit, t.tag, t.note) for t in REGISTER if tag in (None, t.tag)]
    return sorted(set(rows))
