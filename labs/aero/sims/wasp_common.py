"""What the WASP run scripts share: the paths, the build by name, a result file that survives a stop."""
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "labs" / "aero" / "designs"))
import wasp  # noqa: E402

OUT = ROOT / "labs" / "aero" / "sims" / "output"
RUNS = ROOT / "labs" / "aero" / "_runs" / "wasp"
BUILDS = list(wasp.VARIANTS)
DEFAULT = "WASP-1F"


def slug(build: str) -> str:
    return build.lower().replace("-", "")


def load(path: Path, fresh: bool = False) -> dict | None:
    return None if fresh or not path.exists() else yaml.safe_load(path.read_text())


def save(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    tmp.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True))
    tmp.replace(path)


def num(x, n=4):
    return None if x is None else round(float(x), n)
