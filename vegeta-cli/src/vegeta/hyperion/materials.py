"""Optical materials: the refractive index from a Sellmeier equation, its temperature coefficient, the thermal
conductivity and expansion a laser crystal's thermal lens needs. A handful of named materials; any other is a
``Material`` the engineer writes down. Wavelengths are metres everywhere in Hyperion (the Sellmeier coefficients
below are in µm², as the catalogues print them, and converted inside ``n``)."""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

SELLMEIER = "sellmeier"
AIR = "air"


@dataclass(frozen=True)
class Material:
    """``n(wavelength)`` from Sellmeier's n² = 1 + Σ Bᵢλ²/(λ² − Cᵢ) (λ in µm, ``C`` in µm²); ``dn_dT`` [1/K] the
    absolute temperature coefficient of the index near 1 µm, ``kappa`` [W/m/K] the thermal conductivity,
    ``alpha_T`` [1/K] the expansion, ``nu`` Poisson's ratio — the numbers a thermal lens is made of. ``note`` says
    where the coefficients come from."""
    name: str
    B: tuple[float, ...]
    C: tuple[float, ...]
    dn_dT: float = 0.0
    kappa: float = math.nan
    alpha_T: float = math.nan
    nu: float = math.nan
    model: str = SELLMEIER
    note: str = ""

    def n(self, wavelength: float) -> float:
        """The refractive index at a vacuum wavelength [m]."""
        lam = wavelength * 1e6
        l2 = lam * lam
        if self.model == AIR:
            s2 = 1.0 / l2                                      # Ciddor's two-term form for dry air at 15 °C, 1013 hPa
            return 1.0 + self.B[0] / (self.C[0] - s2) + self.B[1] / (self.C[1] - s2)
        n2 = 1.0 + sum(b * l2 / (l2 - c) for b, c in zip(self.B, self.C))
        if n2 <= 0:
            raise ValueError(f"{self.name}: no real index at {lam:.4g} µm")
        return math.sqrt(n2)

    def group_index(self, wavelength: float, dl: float = 1e-10) -> float:
        """n − λ dn/dλ (a centred difference), for pulse timing and the cavity's optical length in a dispersive slab."""
        n0, nm, np_ = self.n(wavelength), self.n(wavelength - dl), self.n(wavelength + dl)
        return n0 - wavelength * (np_ - nm) / (2 * dl)

    def with_(self, **changes) -> "Material":
        """A copy with some numbers changed (a doped crystal's conductivity, a different dn/dT)."""
        return replace(self, **changes)

    def to_dict(self) -> dict:
        return {"name": self.name, "B": list(self.B), "C": list(self.C), "dn_dT": self.dn_dT, "kappa": self.kappa,
                "alpha_T": self.alpha_T, "nu": self.nu, "model": self.model, "note": self.note}

    @classmethod
    def from_dict(cls, d: dict) -> "Material":
        return cls(d["name"], tuple(d["B"]), tuple(d["C"]), d.get("dn_dT", 0.0), d.get("kappa", math.nan),
                   d.get("alpha_T", math.nan), d.get("nu", math.nan), d.get("model", SELLMEIER), d.get("note", ""))


MATERIALS: dict[str, Material] = {}


def _add(m: Material) -> Material:
    MATERIALS[m.name.lower()] = m
    return m


BK7 = _add(Material("BK7", (1.03961212, 0.231792344, 1.01046945), (0.00600069867, 0.0200179144, 103.560653),
                    dn_dT=1.4e-6, kappa=1.114, alpha_T=7.1e-6, nu=0.206, note="Schott N-BK7 catalogue Sellmeier; dn/dT absolute near 1 µm"))
FUSED_SILICA = _add(Material("fused silica", (0.6961663, 0.4079426, 0.8974794), (0.0684043**2, 0.1162414**2, 9.896161**2),
                             dn_dT=1.0e-5, kappa=1.38, alpha_T=0.55e-6, nu=0.17, note="Malitson 1965; dn/dT ≈ 1e-5/K"))
YAG = _add(Material("Nd:YAG", (2.28200, 3.27644), (0.01185, 282.734), dn_dT=7.3e-6, kappa=13.0, alpha_T=7.8e-6, nu=0.25,
                    note="Zelmon 1998 (undoped YAG Sellmeier); Nd:YAG dn/dT, κ, α as in Koechner"))
SAPPHIRE = _add(Material("sapphire", (1.4313493, 0.65054713, 5.3414021), (0.0726631**2, 0.1193242**2, 18.028251**2),
                         dn_dT=1.3e-5, kappa=33.0, alpha_T=5.3e-6, nu=0.29, note="Malitson 1962, ordinary ray"))
AIR_STP = _add(Material("air", (5.792105e-2, 1.67917e-3), (238.0185, 57.362), dn_dT=-1e-6, kappa=0.026, alpha_T=math.nan,
                        model=AIR, note="Ciddor 1996 two-term dry air, 15 °C, 1013 hPa"))
VACUUM = _add(Material("vacuum", (), (), note="n = 1"))


def get(name: str | Material) -> Material:
    """A named material (case-insensitive: ``"bk7"``, ``"Nd:YAG"``, ``"fused silica"``) or the Material itself."""
    if isinstance(name, Material):
        return name
    key = str(name).lower().replace("_", " ")
    if key in MATERIALS:
        return MATERIALS[key]
    raise ValueError(f"unknown material {name!r}; known: {', '.join(m.name for m in MATERIALS.values())}")


def index_of(medium, wavelength: float | None) -> float:
    """The index of ``medium`` — a number, a Material or a material's name — at ``wavelength`` [m]. A Material needs
    the wavelength; a number is taken as is."""
    if isinstance(medium, (int, float)):
        return float(medium)
    m = get(medium)
    if m.model == SELLMEIER and not m.B:
        return 1.0
    if wavelength is None:
        raise ValueError(f"{m.name}: the index depends on the wavelength; pass wavelength=")
    return m.n(wavelength)


def names() -> list[str]:
    return [m.name for m in MATERIALS.values()]
