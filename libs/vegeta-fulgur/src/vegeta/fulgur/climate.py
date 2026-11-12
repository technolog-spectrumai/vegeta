"""The design conditions: Warsaw's year (monthly means, the station Warszawa-Okęcie, 1991-2020 normals, rounded)
and dry indoor air. Each month is an ``Air``; ``worst`` picks the month that minimises the spark margin."""
from __future__ import annotations

from .gas import Air, INDOOR

# month: (mean temperature °C, mean relative humidity %)
WARSAW = {
    "jan": (-1.8, 86), "feb": (-0.6, 83), "mar": (3.1, 76), "apr": (9.2, 68), "may": (14.5, 68), "jun": (17.9, 69),
    "jul": (19.8, 70), "aug": (19.2, 70), "sep": (14.0, 76), "oct": (8.7, 82), "nov": (3.6, 87), "dec": (-0.3, 88),
}
WARSAW_HUMID_DAY = Air(T=291.15, rh=0.95)     # a rainy day: what a demonstrator must survive, not run at full power in
WARSAW_P = 100_600.0                           # Pa: 106 m above sea level


def warsaw(month: str) -> Air:
    t, rh = WARSAW[month]
    return Air(T=273.15 + t, p=WARSAW_P, rh=rh / 100.0)


def conditions() -> dict[str, Air]:
    """Every design condition by name: the twelve months, the humid day, the dry room."""
    out = {m: warsaw(m) for m in WARSAW}
    out["humid_day"] = WARSAW_HUMID_DAY
    out["indoor"] = INDOOR
    return out
