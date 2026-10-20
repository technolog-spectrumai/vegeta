"""Package boundaries, checked from the source text (no imports): Vegeta's engineering code never reaches for
robotics — MuJoCo, Chiron, the onboard mission stack, the races, Piccolo."""
import ast
import fnmatch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ROBOTICS_IMPORTS = ("mujoco", "vegeta.chiron", "vegeta.mission", "chiron", "onboard", "races", "piccolo")
HATCHED = ("races", "piccolo", "onboard")              # graduated to Piccolo: forbidden everywhere, moving lists or not

ENGINEERING = ["vegeta-cli/src/vegeta", "vegeta-core/src", "vegeta-ai/src", "vegeta-fidia/src", "components",
               "assemblies", "designs", "side_labs", "scenarios", "examples"]
# Robotics still in Vegeta until it moves to Piccolo/Chiron (docs/migration); emptied by the migration.
MOVING = ["vegeta-cli/src/vegeta/chiron/*", "vegeta-cli/src/vegeta/mission/*", "designs/*_robot.py",
          "designs/*_controller.py", "designs/*_scenario.py", "designs/*_birds*.py", "designs/gait.py",
          "designs/stability_plots.py", "designs/exp/mini_races.py", "designs/tests/*",
          "scenarios/*", "designs/nisus_robot*.py"]
KNOWN_COUPLING = {"vegeta-cli/src/vegeta/mock/__init__.py", "designs/onager_sweeper_cfd.py"}


def imports(path):
    for node in ast.walk(ast.parse(path.read_text(), str(path))):
        if isinstance(node, ast.Import):
            yield from (a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.module


def robotic(name):
    return any(name == r or name.startswith(r + ".") for r in ROBOTICS_IMPORTS)


def test_engineering_code_imports_no_robotics():
    bad = []
    for root in ENGINEERING:
        for p in sorted((ROOT / root).rglob("*.py")):
            rel = p.relative_to(ROOT).as_posix()
            if "__pycache__" in rel or rel in KNOWN_COUPLING or any(fnmatch.fnmatch(rel, m) for m in MOVING):
                continue
            bad += [f"{rel}: {n}" for n in imports(p) if robotic(n)]
    assert bad == []


def test_vegeta_never_imports_piccolo():
    bad = [f"{p.relative_to(ROOT)}" for p in ROOT.rglob("*.py")
           if ".venv" not in p.parts and "vendor" not in p.parts and any(n.split(".")[0] == "piccolo" for n in imports(p))]
    assert bad == []


def test_nothing_imports_what_hatched():
    bad = []
    for p in sorted(ROOT.rglob("*.py")):
        rel = p.relative_to(ROOT).as_posix()
        if ".venv" in p.parts or "vendor" in p.parts or "__pycache__" in rel:
            continue
        bad += [f"{rel}: {n}" for n in imports(p) if n.split(".")[0] in HATCHED]
    assert bad == []
