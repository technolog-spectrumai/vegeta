"""What Vegeta is, from its syntax tree: line statistics, connections, path funnels, and what each part serves.

    python tools/dump_ast.py && python tools/ast_report.py [out/vegeta_ast.xml.gz] [--out docs/anatomy]

Reads only the XML ``tools/dump_ast.py`` writes and writes ``stats.json`` (every number below), ``funnels.svg`` (the
modules the entry points' import paths converge on) and ``areas.dot`` (the area graph for Graphviz).
``tools/graph_metrics.py`` draws the anatomy (``anatomy.svg``). The words in ``docs/anatomy/README.md`` are written from ``stats.json``.

Terms
  area        a part of Vegeta: ``vegeta.<pkg>`` for the packages, the top-level directory otherwise (``designs``…)
  entry       where a run starts: a CLI command, a notebook, a script with ``__main__``, a test file
  family      who starts it: cli, designs, notebooks, races, scenarios, side_labs, benchmark, agents, tests, …
  reach       the modules an entry imports, transitively
  funnel      a module many entries' import paths go through (entry betweenness: shortest paths from every entry)
  orphan      library code no entry reaches, not even a test
  unreferenced  a top-level function or class whose name appears nowhere else in the tree (from the full AST)
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict, deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STDLIB = set(sys.stdlib_module_names) | {"__future__"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BAR = "#2a78d6"


# ------------------------------------------------------------------------------------------------------------- read
def load(path: Path) -> dict:
    """The index layer, plus every name the full syntax trees reference (for unreferenced definitions)."""
    mods, entries, refs = {}, [], Counter()
    opener = gzip.open if path.suffix == ".gz" else open
    cur, depth_ast = None, 0
    with opener(path, "rb") as f:
        for ev, el in ET.iterparse(f, events=("start", "end")):
            tag = el.tag
            if ev == "start":
                if tag == "module":
                    a = el.attrib
                    cur = {"path": a["path"], "area": a["area"], "dotted": a["dotted"], "kind": a["kind"],
                           "main": a["main"] == "1", "sha1": a.get("sha1", ""), **{k: int(a[k]) for k in ("lines", "code", "comment",
                                                                                 "docstring", "blank")},
                           "imports": [], "defs": [], "error": None}
                    mods[cur["path"]] = cur
                elif tag == "ast":
                    depth_ast += 1
                continue
            if tag == "import" and cur is not None and not depth_ast:
                cur["imports"].append(dict(el.attrib))
            elif tag == "def" and cur is not None and not depth_ast:
                d = dict(el.attrib)
                d["lines"] = int(d["lines"])
                cur["defs"].append(d)
            elif tag == "error" and cur is not None:
                cur["error"] = el.text
            elif tag == "entry":
                entries.append(dict(el.attrib))
            elif depth_ast:
                if tag == "Name":
                    refs[(el.get("id"), cur["path"])] += 1
                elif tag == "Attribute":
                    refs[(el.get("attr"), cur["path"])] += 1
                elif tag == "alias":
                    refs[(el.get("name", "").split(".")[-1], cur["path"])] += 1
                elif tag == "Constant" and el.get("value", "").isidentifier():
                    refs[(el.get("value"), cur["path"])] += 1          # getattr(mod, "name"), registries
                if tag == "ast":
                    depth_ast -= 1
                el.clear()
            if tag == "module":
                cur = None
                el.clear()
    return {"modules": mods, "entries": entries, "refs": refs}


# ------------------------------------------------------------------------------------------------------------ graph
def graph(mods):
    g = defaultdict(set)
    for p, m in mods.items():
        for im in m["imports"]:
            t = im.get("target")
            if t and t != p and t in mods:
                g[p].add(t)
    return g


def bfs(g, start):
    seen, q = {start}, deque([start])
    while q:
        for v in g.get(q.popleft(), ()):
            if v not in seen:
                seen.add(v)
                q.append(v)
    return seen


def tarjan(g, nodes):
    index, low, on, stack, out, i = {}, {}, set(), [], [], [0]
    sys.setrecursionlimit(10000)

    def strong(v):
        index[v] = low[v] = i[0]
        i[0] += 1
        stack.append(v)
        on.add(v)
        for w in g.get(v, ()):
            if w not in index:
                strong(w)
                low[v] = min(low[v], low[w])
            elif w in on:
                low[v] = min(low[v], index[w])
        if low[v] == index[v]:
            comp = []
            while True:
                w = stack.pop()
                on.discard(w)
                comp.append(w)
                if w == v:
                    break
            if len(comp) > 1:
                out.append(sorted(comp))

    for v in nodes:
        if v not in index:
            strong(v)
    return out


def entry_betweenness(g, sources):
    """Brandes, with the shortest paths starting at the entries only: how many entry→module paths cross each module."""
    bc = Counter()
    for s in sources:
        stack, pred, sigma, dist = [], defaultdict(list), Counter({s: 1}), {s: 0}
        q = deque([s])
        while q:
            v = q.popleft()
            stack.append(v)
            for w in g.get(v, ()):
                if w not in dist:
                    dist[w] = dist[v] + 1
                    q.append(w)
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    pred[w].append(v)
        delta = Counter()
        while stack:
            w = stack.pop()
            for v in pred[w]:
                delta[v] += sigma[v] / sigma[w] * (1 + delta[w])
            if w != s:
                bc[w] += delta[w]
    return bc


# ---------------------------------------------------------------------------------------------------------- analyse
def analyse(data) -> dict:
    mods, entries, refs = data["modules"], data["entries"], data["refs"]
    g = graph(mods)
    areas = sorted({m["area"] for m in mods.values()})
    area_of = {p: m["area"] for p, m in mods.items()}

    # lines
    by_area = {a: Counter() for a in areas}
    by_kind = Counter()
    for m in mods.values():
        c = by_area[m["area"]]
        c["files"] += 1
        for k in ("lines", "code", "comment", "docstring", "blank"):
            c[k] += m[k]
        c[f"code_{m['kind']}"] += m["code"]
        by_kind[m["kind"]] += m["code"]
    biggest_modules = sorted(mods.values(), key=lambda m: -m["code"])[:25]
    all_defs = [(d, m) for m in mods.values() for d in m["defs"] if d["kind"] != "class"]
    biggest_defs = sorted(all_defs, key=lambda x: -x[0]["lines"])[:25]

    # connections
    fan_in, fan_out = Counter(), Counter()
    area_edges = Counter()
    for p, ts in g.items():
        fan_out[p] = len(ts)
        for t in ts:
            fan_in[t] += 1
            if area_of[p] != area_of[t]:
                area_edges[(area_of[p], area_of[t])] += 1
    cycles = tarjan(g, sorted(mods))
    ext = defaultdict(Counter)
    for m in mods.values():
        for im in m["imports"]:
            e = im.get("external")
            if e and e not in STDLIB and e != "vegeta":
                ext[e][m["area"]] += 1

    # reach and funnels
    reach_by_family = defaultdict(set)
    reached_by = defaultdict(set)                                       # module -> families
    entries_reaching = Counter()                                        # module -> non-test entries
    features = defaultdict(set)                                         # area -> non-test entry names
    for e in entries:
        r = bfs(g, e["module"])
        fam = e["family"]
        reach_by_family[fam] |= r
        for v in r:
            reached_by[v].add(fam)
            if fam != "tests":
                entries_reaching[v] += 1
                features[area_of[v]].add(f"{fam}: {e['name']}")
    sources = sorted({e["module"] for e in entries if e["family"] != "tests"})
    bc = entry_betweenness(g, sources)
    funnels = [p for p, _ in bc.most_common(40) if mods[p]["kind"] not in ("test",)][:25]

    libs = [p for p, m in mods.items() if m["kind"] in ("lib",)]
    orphans = sorted(p for p in libs if not reached_by.get(p))
    test_only = sorted(p for p in libs if reached_by.get(p) == {"tests"})

    # unreferenced top-level definitions (names nowhere else, from the full AST)
    name_files = defaultdict(set)
    for (name, path), n in refs.items():
        name_files[name].add(path)
    unref, test_ref = [], []
    for p, m in mods.items():
        if m["kind"] not in ("lib", "script"):
            continue
        for d in m["defs"]:
            if "." in d["qual"] or d["name"].startswith(("__", "pytest_")) or d["name"] in ("main",):
                continue
            where = name_files.get(d["name"], set())
            others = where - {p}
            own_uses = sum(n for (nm, pp), n in ((k, refs[k]) for k in [(d["name"], p)]) if pp == p)
            if not others and own_uses == 0:
                unref.append((p, d))
            elif others and all(mods[o]["kind"] == "test" for o in others) and own_uses == 0:
                test_ref.append((p, d))

    # what serves what
    serves = {}
    for a in areas:
        fams = Counter()
        for p, m in mods.items():
            if m["area"] == a:
                for f in reached_by.get(p, ()):
                    fams[f] += m["code"]
        serves[a] = {"families": dict(fams.most_common()), "features": len(features[a]),
                     "examples": sorted(features[a])[:12]}
    fam_code = {f: sum(mods[p]["code"] for p in r if mods[p]["kind"] != "test") for f, r in reach_by_family.items()}

    return {
        "totals": {"modules": len(mods), "lines": sum(m["lines"] for m in mods.values()),
                   "code": sum(m["code"] for m in mods.values()), "by_kind": dict(by_kind),
                   "entries": Counter(e["family"] for e in entries), "imports_internal": sum(len(v) for v in g.values()),
                   "unparsable": [p for p, m in mods.items() if m["error"]]},
        "areas": {a: dict(by_area[a]) for a in areas},
        "biggest_modules": [{"path": m["path"], "code": m["code"]} for m in biggest_modules],
        "biggest_functions": [{"path": m["path"], "qual": d["qual"], "lines": d["lines"]} for d, m in biggest_defs],
        "fan_in": [{"path": p, "n": n} for p, n in fan_in.most_common(25)],
        "fan_out": [{"path": p, "n": n} for p, n in fan_out.most_common(15)],
        "area_edges": [{"from": a, "to": b, "n": n} for (a, b), n in area_edges.most_common()],
        "cycles": cycles,
        "external": {e: dict(c) for e, c in sorted(ext.items(), key=lambda x: -sum(x[1].values()))},
        "family_reach_code": dict(sorted(fam_code.items(), key=lambda x: -x[1])),
        "funnels": [{"path": p, "betweenness": round(bc[p], 1), "entries": entries_reaching[p],
                     "families": sorted(reached_by[p])} for p in funnels],
        "orphans": [{"path": p, "code": mods[p]["code"]} for p in orphans],
        "test_only": [{"path": p, "code": mods[p]["code"]} for p in test_only],
        "unreferenced": [{"path": p, "def": d["qual"], "lines": d["lines"]} for p, d in
                         sorted(unref, key=lambda x: -x[1]["lines"])],
        "test_referenced_only": [{"path": p, "def": d["qual"], "lines": d["lines"]} for p, d in
                                 sorted(test_ref, key=lambda x: -x[1]["lines"])],
        "serves": serves,
        "undeclared_imports": _undeclared(mods, ext),
        "duplicates": _duplicates(mods),
    }


DIST_TO_IMPORT = {"opencv-python": "cv2", "opencv-python-headless": "cv2", "pillow": "PIL", "usd-core": "pxr",
                  "cadquery-ocp": "OCP", "pyyaml": "yaml", "ipython": "IPython", "matplotlib": "mpl_toolkits",
                  "pyqt6": "PyQt6", "jupyterlab": "IPython", "ipykernel": "IPython", "cadquery": "OCP"}


def declared_dependencies(root: Path = ROOT) -> set[str]:
    """Import names of everything a pyproject, a requirements file or install_local.sh installs."""
    import re
    import tomllib
    names = set()
    for pp in root.rglob("pyproject.toml"):
        if ".venv" in pp.parts:
            continue
        proj = tomllib.loads(pp.read_text()).get("project", {})
        names.add(proj.get("name", ""))
        for dep in proj.get("dependencies", []) + [d for v in proj.get("optional-dependencies", {}).values()
                                                    for d in v]:
            names.add(re.split(r"[<>=!\[; ~]", dep, maxsplit=1)[0])
    for f in list(root.glob("*.sh")) + list(root.rglob("requirements*.txt")):
        if ".venv" in f.parts:
            continue
        text = f.read_text(errors="replace").replace("\\\n", " ")                # join continued lines
        for m in re.finditer(r"pip install([^\n]*)", text) if f.suffix == ".sh" else [None]:
            chunk = m.group(1) if m else text
            for tok in re.findall(r"[\"']?([A-Za-z][\w.-]*)(?:[<>=!~\[][^\s\"']*)?[\"']?", chunk):
                names.add(tok)
    out = set()
    for n in names:
        k = n.lower()
        out.add(DIST_TO_IMPORT.get(k, n.replace("-", "_")))
        out.add(k.replace("-", "_"))
    return out


def _undeclared(mods, ext) -> list:
    """Third-party imports no pyproject, requirements file or install script declares (stale, or missing a dep)."""
    have = declared_dependencies()
    return [{"name": e, "areas": dict(c)} for e, c in ext.items() if e not in have and e.lower() not in have]


def _duplicates(mods) -> list:
    """Files with the same source (sha1), library code only (``__init__``/``__main__`` stubs and tests aside)."""
    groups = defaultdict(list)
    for p, m in mods.items():
        if m["sha1"] and m["code"] >= 10 and m["kind"] not in ("test",):
            groups[m["sha1"]].append(p)
    return sorted(({"files": sorted(v), "code": mods[v[0]]["code"]} for v in groups.values() if len(v) > 1),
                  key=lambda d: -d["code"])


# --------------------------------------------------------------------------------------------------------- drawings
def funnels_svg(st: dict, out: Path):
    """The funnel modules as rows; each row's bar is split by the families whose entries run through it."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = st["funnels"][:20]
    fig, ax = plt.subplots(figsize=(10, 0.36 * len(rows) + 1.6), dpi=100)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for i, r in enumerate(rows):
        y = len(rows) - 1 - i
        ax.barh(y, r["betweenness"], height=0.55, color=BAR, edgecolor=SURFACE, linewidth=2)
        ax.text(r["betweenness"], y, f"  {r['entries']} entries · " + ", ".join(f for f in r["families"]
                                                                               if f != "tests"),
                va="center", fontsize=7, color=INK2)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["path"] for r in reversed(rows)], fontsize=7.5, color=INK)
    ax.set_xlabel("entry betweenness: shortest import paths from every entry point that pass through the module",
                  fontsize=8, color=INK2)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=7.5, length=0)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xlim(0, max(r["betweenness"] for r in rows) * 1.6)
    ax.set_title("The funnels: modules the entry points' import paths converge on", loc="left", fontsize=11,
                 color=INK)
    fig.savefig(out, format=out.suffix[1:], bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def area_dot(st: dict, out: Path):
    lines = ["digraph vegeta {", "  rankdir=LR; node [shape=box, style=rounded];"]
    for a, v in st["areas"].items():
        lines.append(f'  "{a}" [label="{a}\\n{v["code"]} code lines"];')
    for e in st["area_edges"]:
        lines.append(f'  "{e["from"]}" -> "{e["to"]}" [penwidth={1 + math.log(e["n"]):.2f}, label="{e["n"]}"];')
    out.write_text("\n".join(lines + ["}"]) + "\n")


# ------------------------------------------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("xml", nargs="?", default=str(ROOT / "out" / "vegeta_ast.xml.gz"))
    ap.add_argument("--out", default=str(ROOT / "docs" / "anatomy"))
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    st = analyse(load(Path(a.xml)))
    (out / "stats.json").write_text(json.dumps(st, indent=1, default=list) + "\n")
    funnels_svg(st, out / "funnels.svg")
    area_dot(st, out / "areas.dot")
    t = st["totals"]
    print(f"{t['modules']} modules, {t['code']} code lines; {len(st['orphans'])} orphan modules, "
          f"{len(st['unreferenced'])} unreferenced definitions, {len(st['cycles'])} import cycles -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
