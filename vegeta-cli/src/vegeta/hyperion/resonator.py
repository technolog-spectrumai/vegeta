"""Resonators: a linear cavity (a train from one end mirror to the other) or a ring (the elements around the loop),
their round-trip matrix, stability, eigenmode (the self-consistent Gaussian beam), the spot on every element, the
Gouy phase and the longitudinal/transverse mode spacing, the finesse and linewidth from the mirrors' reflectivities
and the round-trip loss, and a sweep of any parameter (a thermal lens, a spacing) for the stability window."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .beam import C_LIGHT, BeamTrace, GaussianBeam, propagate_q, spot_from, trace
from .elements import CurvedMirror, Element, FlatMirror, Train, is_mirror

PLANES = ("t", "s")


@dataclass
class Mode:
    """The eigenmode in one plane: ``stable``, ``m`` = (A+D)/2 of the round trip, the beam at the reference plane
    (``beam``), the waist ``w0`` and where it sits (``z0`` from the reference plane along the forward path), the
    spot ``spots`` on every element (name, z, w, R), the round-trip Gouy phase, the full ``trace`` for drawing."""
    plane: str
    stable: bool
    m: float
    beam: GaussianBeam | None = None
    w0: float = math.nan
    z0: float = math.nan
    gouy_round_trip: float = math.nan
    spots: list[dict] = field(default_factory=list)
    trace: BeamTrace | None = None
    note: str = ""

    def spot(self, name: str) -> float:
        """The spot radius on the element (or part of a composite: ``"rod in"``, ``"rod out"``) named ``name``."""
        for s in self.spots:
            if s["name"] == name:
                return s["w"]
        for s in self.spots:
            if s["element"] == name:
                return s["w"]
        raise KeyError(f"no element named {name!r}; have {[s['name'] for s in self.spots]}")

    def to_dict(self) -> dict:
        return {"plane": self.plane, "stable": self.stable, "m": self.m, "w0": self.w0, "z0": self.z0,
                "gouy_round_trip": self.gouy_round_trip, "spots": self.spots, "note": self.note,
                "beam": self.beam.to_dict() if self.beam else None}


@dataclass(frozen=True)
class Resonator:
    """``train``: for a linear cavity the elements from the first mirror to the last (both must be mirrors); for a
    ``ring`` every element around the loop in order, the reference plane just before the first. ``wavelength`` [m]
    the laser line; ``loss`` the fractional power loss per round trip from everything but the mirrors (scatter,
    absorption, a crystal's imperfection); ``name`` for the tables."""
    train: Train
    wavelength: float
    ring: bool = False
    loss: float = 0.0
    name: str = ""

    def __post_init__(self):
        els = self.train.elements
        if not els:
            raise ValueError("an empty train")
        if not self.ring and not (is_mirror(els[0]) and is_mirror(els[-1])):
            raise ValueError("a linear cavity's train must start and end with a mirror (FlatMirror / CurvedMirror)")
        self.train.check(self.wavelength)

    # ------------------------------------------------------------------------------------------------ the round trip
    def forward(self) -> Train:
        """The path a beam traces from the reference plane: the whole loop (ring) or mirror 1 → mirror 2 (linear, the
        first mirror excluded: the reference plane sits just after it)."""
        els = self.train.elements
        return Train(els if self.ring else els[1:], self.name)

    def round_trip_train(self) -> Train:
        els = self.train.elements
        if self.ring:
            return Train(els, self.name)
        inner = list(els[1:-1])
        return Train(inner + [els[-1]] + [e.reverse() for e in reversed(inner)] + [els[0]], self.name)

    def round_trip(self, plane: str = "t") -> np.ndarray:
        return self.round_trip_train().matrix(self.wavelength, plane)

    def half_trace(self, plane: str = "t") -> float:
        m = self.round_trip(plane)
        return float(m[0, 0] + m[1, 1]) / 2.0

    def stable(self, plane: str = "t") -> bool:
        return abs(self.half_trace(plane)) < 1.0

    def g_parameters(self) -> dict | None:
        """For a bare two-mirror cavity (mirrors and free space only): g₁ = 1 − L/R₁, g₂ = 1 − L/R₂ and their product."""
        els = self.train.elements
        if self.ring or any(not (is_mirror(e) or e.kind == "FreeSpace") for e in els):
            return None
        L = self.train.optical_length(self.wavelength)
        r = [e.R if isinstance(e, CurvedMirror) else math.inf for e in (els[0], els[-1])]
        g = [1.0 - L / ri if math.isfinite(ri) else 1.0 for ri in r]
        return {"g1": g[0], "g2": g[1], "g1g2": g[0] * g[1], "L": L}

    # ------------------------------------------------------------------------------------------------- the eigenmode
    def mode(self, plane: str = "t", samples: int = 40) -> Mode:
        """The self-reproducing beam at the reference plane and along the forward path."""
        m = self.round_trip(plane)
        (a, b), (c, d) = m
        half = (a + d) / 2.0
        if abs(half) >= 1.0:
            return Mode(plane, False, float(half), note="unstable: |(A+D)/2| ≥ 1")
        if abs(b) < 1e-15:
            return Mode(plane, True, float(half), note="degenerate round trip (B = 0): every beam reproduces itself")
        root = math.sqrt(4.0 - (a + d) ** 2)
        inv_q = complex((d - a) / (2.0 * b), -root / (2.0 * abs(b)))
        q = 1.0 / inv_q
        n0 = self.train.n_in(self.wavelength) if self.ring else self.train.elements[1].n_in(self.wavelength)
        beam = GaussianBeam.from_q(q, self.wavelength, n0, 1.0, self.name)
        tr = trace(beam, self.forward(), plane, samples)
        spots = [{"name": s["name"], "element": s["element"], "z": s["z"], "w": s["w"], "R": s["R"]} for s in tr.stations[1:]]
        first = self.train.elements[0]._label() if not self.ring else "reference"
        spots.insert(0, {"name": first, "element": first, "z": 0.0, "w": tr.stations[0]["w"], "R": tr.stations[0]["R"]})
        gouy = math.acos(max(-1.0, min(1.0, half)))
        if b < 0:
            gouy = 2.0 * math.pi - gouy
        return Mode(plane, True, float(half), beam, beam.w0, beam.z0, gouy, spots, tr)

    def modes(self, samples: int = 40) -> dict[str, Mode]:
        return {p: self.mode(p, samples) for p in PLANES}

    def astigmatic(self) -> bool:
        """Whether any element distinguishes the two planes (a tilted mirror, a Brewster plate)."""
        return any((isinstance(e, CurvedMirror) and e.angle != 0.0) or e.kind == "BrewsterPlate" for e in self.train.elements)

    # ---------------------------------------------------------------------------------------------------- spectrum
    def optical_length(self) -> float:
        """One-way optical length of a linear cavity, the loop's for a ring [m]."""
        return self.train.optical_length(self.wavelength)

    def round_trip_length(self) -> float:
        return self.optical_length() * (1.0 if self.ring else 2.0)

    def mirrors(self) -> list[Element]:
        return [e for e in self.train.elements if is_mirror(e)]

    def survival(self) -> float:
        """Power left after one round trip: Π R_mirror · (1 − loss)."""
        r = 1.0
        for e in self.mirrors():
            r *= e.reflectivity
        return r * (1.0 - self.loss)

    def spectrum(self) -> dict:
        """FSR, round-trip time, photon lifetime, finesse, linewidth, Q, the transverse mode spacing from the Gouy
        phase (tangential plane), the total round-trip loss."""
        L_rt = self.round_trip_length()
        fsr = C_LIGHT / L_rt
        t_rt = L_rt / C_LIGHT
        s = self.survival()
        delta = -math.log(s) if s > 0 else math.inf                  # the round-trip loss as an exponent
        tau_c = t_rt / delta if delta > 0 else math.inf
        r_amp = math.sqrt(s)
        finesse = math.pi * math.sqrt(r_amp) / (1.0 - r_amp) if r_amp < 1 else math.inf
        linewidth = fsr / finesse if math.isfinite(finesse) else 0.0
        nu = C_LIGHT / self.wavelength
        mode = self.mode("t", samples=2)
        out = {"optical_length": self.optical_length(), "round_trip_length": L_rt, "fsr": fsr, "round_trip_time": t_rt,
               "survival": s, "loss_round_trip": 1.0 - s, "photon_lifetime": tau_c, "finesse": finesse, "linewidth": linewidth,
               "Q": nu * 2 * math.pi * tau_c if math.isfinite(tau_c) else math.inf,
               "longitudinal_modes_per_nm": 1e-9 * nu * nu / C_LIGHT / fsr}
        if mode.stable:
            out["gouy_round_trip"] = mode.gouy_round_trip
            out["transverse_mode_spacing"] = fsr * mode.gouy_round_trip / (2 * math.pi)
        return out

    # ------------------------------------------------------------------------------------------------------- sweeps
    @staticmethod
    def sweep(builder: Callable[[float], "Resonator"], values, elements: list[str] | None = None, plane: str = "t") -> dict:
        """``builder(value)`` → a Resonator; over ``values`` the half-trace m, stability, waist and the spot on the named
        elements (all elements when None). Arrays, nan where unstable."""
        values = np.asarray(list(values), dtype=float)
        ms, w0s, rows = [], [], []
        for v in values:
            mode = builder(float(v)).mode(plane, samples=2)
            ms.append(mode.m)
            w0s.append(mode.w0 if mode.stable else math.nan)
            rows.append({s["name"]: s["w"] for s in mode.spots} if mode.stable else {})
        names = elements or sorted({k for r in rows for k in r}, key=lambda k: next(i for r in rows for i, kk in enumerate(r) if kk == k))
        spots = {nm: np.array([r.get(nm, math.nan) for r in rows]) for nm in names}
        return {"values": values, "m": np.array(ms), "stable": np.abs(np.array(ms)) < 1.0, "w0": np.array(w0s), "spots": spots}

    def summary(self) -> dict:
        """Stability and mode in both planes, the spectrum, the g-parameters where they exist — one dict for a table."""
        out = {"name": self.name, "ring": self.ring, "wavelength": self.wavelength, "elements": [e._label() for e in self.train.elements]}
        for p, mode in self.modes(samples=2).items():
            out[f"stable_{p}"] = mode.stable
            out[f"m_{p}"] = mode.m
            out[f"w0_{p}"] = mode.w0
            out[f"z0_{p}"] = mode.z0
            out[f"spots_{p}"] = {s["name"]: s["w"] for s in mode.spots} if mode.stable else {}
        g = self.g_parameters()
        if g:
            out.update(g)
        out.update(self.spectrum())
        return out

    def to_dict(self) -> dict:
        return {"train": self.train.to_dict(), "wavelength": self.wavelength, "ring": self.ring, "loss": self.loss, "name": self.name}

    @classmethod
    def from_dict(cls, d: dict) -> "Resonator":
        return cls(Train.from_dict(d["train"]), d["wavelength"], d.get("ring", False), d.get("loss", 0.0), d.get("name", ""))


def two_mirror(R1: float, R2: float, L: float, wavelength: float, reflectivity: tuple[float, float] = (1.0, 1.0), loss: float = 0.0,
               name: str = "") -> Resonator:
    """The textbook cavity: mirror 1 (radius R1, inf for flat), length L, mirror 2."""
    def mirror(R, refl, nm):
        return FlatMirror(refl, name=nm) if math.isinf(R) else CurvedMirror(R, 0.0, refl, name=nm)
    from .elements import FreeSpace
    return Resonator(Train([mirror(R1, reflectivity[0], "M1"), FreeSpace(L, name="cavity"), mirror(R2, reflectivity[1], "M2")]),
                     wavelength, False, loss, name)


def kogelnik_li(R1: float, R2: float, L: float, wavelength: float) -> dict:
    """The closed forms for a two-mirror cavity (Kogelnik & Li 1966): g₁, g₂, the waist and its position from mirror 1,
    the spots on the mirrors — for checking the matrices."""
    g1 = 1.0 - L / R1 if math.isfinite(R1) else 1.0
    g2 = 1.0 - L / R2 if math.isfinite(R2) else 1.0
    gg = g1 * g2
    if not 0 <= gg <= 1 or (g1 + g2 - 2 * gg) == 0:
        return {"g1": g1, "g2": g2, "g1g2": gg, "stable": False}
    lam = wavelength
    w0_sq = (lam * L / math.pi) * math.sqrt(gg * (1 - gg)) / abs(g1 + g2 - 2 * gg)
    z1 = g2 * (1 - g1) * L / (g1 + g2 - 2 * gg)
    w1_sq = (lam * L / math.pi) * math.sqrt(g2 / (g1 * (1 - gg))) if g1 != 0 else math.nan
    w2_sq = (lam * L / math.pi) * math.sqrt(g1 / (g2 * (1 - gg))) if g2 != 0 else math.nan
    return {"g1": g1, "g2": g2, "g1g2": gg, "stable": True, "w0": math.sqrt(w0_sq), "z0": z1,
            "w1": math.sqrt(w1_sq) if w1_sq == w1_sq else math.nan, "w2": math.sqrt(w2_sq) if w2_sq == w2_sq else math.nan}
