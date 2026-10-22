"""kestrel_keys — the Porunga keys of the Kestrel side-lab results (docs/cache_identities.md).

The side labs publish a completed run's summary under one of these keys; the notebooks derive the same key from the
same parameters and load it. Neither imports the other: both build the identity here, from the design data.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    import porunga
except ImportError:                                            # a source tree without the editable install
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "vendor" / "ozaru" / "porunga" / "src"))
    import porunga

BLOWN = ("blown", "cruise", "transitions", "all")               # campaigns with the rotor disks: the propeller calibration counts


def cache_dir() -> Path:
    """The local cache: ``$PORUNGA_DIR``, else ``<repo>/cache/porunga``."""
    return Path(os.environ.get("PORUNGA_DIR") or Path(__file__).resolve().parents[3] / "cache" / "porunga")


def _design(variant: str, campaign: str, fingerprint: dict | None) -> dict:
    if fingerprint is None:
        import kestrel_campaigns as kcp
        fingerprint = kcp.fingerprint(variant)
    d = {"params_sha256": fingerprint["params_sha256"]}
    if campaign in BLOWN:
        d["prop_lab"] = fingerprint.get("prop_lab")
    return d


def airframe_identity(variant: str, campaign: str, quality: str, *, mock: bool, fingerprint: dict | None = None) -> dict:
    return {"identity": "vegeta.kestrel.airframe-cfd", "version": 1, "variant": variant, "campaign": campaign,
            "quality": quality, "design": _design(variant, campaign, fingerprint), "mock": bool(mock)}


def structure_identity(variant: str, campaign: str, *, element_size: float = 2.0, n_modes: int = 6, mock: bool,
                       fingerprint: dict | None = None) -> dict:
    return {"identity": "vegeta.kestrel.structure-fea", "version": 1, "variant": variant, "campaign": campaign,
            "settings": {"element_size_mm": float(element_size), "n_modes": int(n_modes)},
            "design": _design(variant, campaign, fingerprint), "mock": bool(mock)}


def airframe(variant, campaign, quality, *, mock, fingerprint=None) -> str:
    return porunga.key(airframe_identity(variant, campaign, quality, mock=mock, fingerprint=fingerprint))


def structure(variant, campaign, *, element_size=2.0, n_modes=6, mock, fingerprint=None) -> str:
    return porunga.key(structure_identity(variant, campaign, element_size=element_size, n_modes=n_modes, mock=mock,
                                          fingerprint=fingerprint))
