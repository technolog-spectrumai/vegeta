"""Four drone race environments in MuJoCo for the reference 5-inch racing quad: hangar, industrial zone, farm, quarry.
See README.md. ``python -m races --help``."""
from . import courses, mock, path, refquad, replay, rules  # noqa: F401

__all__ = ["courses", "mock", "path", "refquad", "replay", "rules"]
from . import analysis, ingest  # noqa: E402,F401
