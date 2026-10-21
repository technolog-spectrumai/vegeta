"""Vegeta's syntax tree (``tools/dump_ast.py``) → module dependency and function call graphs → architectural metrics,
and three questions: what does it serve, what is its purpose, how is that achieved.

    python tools/dump_ast.py && python tools/graph_metrics.py [out/vegeta_ast.xml.gz]

Needs networkx (``pip install networkx``). Writes ``docs/anatomy/graph_metrics.json`` and, in ``out/``,
``module_dependencies.graphml`` and ``function_calls.graphml`` (they open in Gephi or Cytoscape). Prints the report.

The graphs
  module graph   A → B: A imports B (``dynamic`` edges: importlib with a name from a table)
  call graph     f → g: f calls g, resolved conservatively from the full AST: local functions and classes, names
                 imported from Vegeta's modules (``from x import f``, ``import x as m; m.f()``), ``self.m()`` inside a
                 class. A module's top-level code is the node ``path::<module>`` (notebooks are mostly that).
                 Calls into third-party libraries are not edges; they are the *effects* below.

The standard metrics (as for any code base): size, density, components, fan-in/fan-out (afferent/efferent coupling),
instability I = out/(in+out), import cycles, PageRank, betweenness, change blast radius (transitive importers),
isolated modules, topological order (of the area graph); for calls: most called, most calling, call betweenness,
recursion groups.

The metrics invented for the three questions
  SERVE     serve set S(x): the entry points whose call closure reaches x (a function, a module, an area).
            breadth = |S| / non-test entries; owner = the family most of S belongs to;
            purpose entropy = normalised Shannon entropy of S over families (0: serves one kind of user,
            1: everyone's infrastructure); test-only = S holds tests only (the code serves its tests).
  PURPOSE   a function's products: what its third-party calls make (PRODUCTS: figures, tables, data files, CAD
            geometry, meshes, solver runs, physics simulation, optimisation, exported models, video, LLM answers,
            notebook display). An entry's product profile = the products its call closure reaches. Vegeta's purpose
            = products ranked by the entries (and families) that reach them. Declared purpose = the docstring of the
            area's most imported module; set against the observed one.
  HOW       for each product: the mechanism funnel — the share of entry→product shortest call paths through each
            module and area (subset betweenness from one virtual node per entry to every function making the
            product); depth = mean call distance from an entry to its nearest producer; mechanism = the libraries
            the producers call; chain = the areas with a real share ordered by their mean distance from the entries.
"""
from __future__ import annotations

import argparse
import gzip
import itertools
import json
import math
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

try:
    import networkx as nx
except ImportError:                                                     # pragma: no cover
    sys.exit("graph_metrics needs networkx: pip install networkx")

ROOT = Path(__file__).resolve().parents[1]
FUNCS = ("FunctionDef", "AsyncFunctionDef")
BUILTINS = set(dir(__builtins__)) if isinstance(__builtins__, dict) is False else set(__builtins__)

# product: (third-party prefixes, attribute names on unknown receivers)
PRODUCTS = {
    "figures": (("matplotlib", "pyvista", "mpl_toolkits", "plotly"), {"savefig", "imshow", "add_subplot", "subplots"}),
    "tables": (("pandas", "tabulate"), {"to_markdown", "to_html", "pivot_table", "groupby"}),
    "data files": (("json.dump", "pickle.dump", "numpy.save", "numpy.savez", "numpy.savetxt", "csv.writer",
                    "shutil.copy", "gzip.open"), {"to_csv", "to_json", "to_parquet", "write_text", "write_bytes"}),
    "CAD geometry": (("cadquery", "OCP"), {"exportStep", "exportStl", "extrude", "revolve", "loft", "fillet",
                                          "chamfer", "Workplane"}),
    "meshes": (("gmsh", "trimesh", "stl"), set()),
    "solver runs": (("subprocess.run", "subprocess.Popen", "subprocess.check_call", "subprocess.check_output",
                     "subprocess.call", "os.system"), set()),
    "physics simulation": (("mujoco", "jsbsim", "scipy.integrate"), {"mj_step", "solve_ivp", "odeint"}),
    "optimisation": (("cma", "scipy.optimize"), {"minimize", "least_squares"}),
    "exported models": (("yax", "pxr"), {"to_mjcf", "to_urdf", "to_sdf"}),
    "video": (("cv2", "imageio", "PIL"), {"VideoWriter"}),
    "LLM answers": (("anthropic",), set()),
    "notebook display": (("IPython",), set()),
}


# the library behind a method name met on an unknown receiver (fig.savefig, df.to_csv, mujoco held in a variable)
ATTR_LIB = {"savefig": "matplotlib", "imshow": "matplotlib", "add_subplot": "matplotlib", "subplots": "matplotlib",
            "to_markdown": "pandas", "to_html": "pandas", "pivot_table": "pandas", "groupby": "pandas",
            "to_csv": "pandas", "to_json": "pandas", "to_parquet": "pandas", "write_text": "pathlib",
            "write_bytes": "pathlib", "exportStep": "cadquery", "exportStl": "cadquery", "extrude": "cadquery",
            "revolve": "cadquery", "loft": "cadquery", "fillet": "cadquery", "chamfer": "cadquery",
            "Workplane": "cadquery", "mj_step": "mujoco", "solve_ivp": "scipy", "odeint": "scipy",
            "minimize": "scipy", "least_squares": "scipy", "to_mjcf": "chiron", "to_urdf": "yax", "to_sdf": "yax",
            "VideoWriter": "cv2"}


# ------------------------------------------------------------------------------------------------------------- read
def dotted(el) -> list[str] | None:
    """A call target as parts: Name(id) or Attribute(attr) chains ending in a Name."""
    if el is None:
        return None
    if el.tag == "Name":
        return [el.get("id")]
    if el.tag == "Attribute":
        base = next((c for c in el if c.get("field") == "value"), None)
        b = dotted(base)
        return b + [el.get("attr")] if b else None
    if el.tag == "Call":                                                # f()() / f().g(): the outer function
        return dotted(next((c for c in el if c.get("field") == "func"), None))
    return None


def calls_in(el, skip_defs: bool):
    """The Call elements under ``el`` (not inside nested defs/classes when ``skip_defs``)."""
    stack = list(el)
    while stack:
        n = stack.pop()
        if skip_defs and n.tag in FUNCS + ("ClassDef",):
            continue
        if n.tag == "Call":
            yield n
        stack.extend(n)


class Module:
    def __init__(self, el):
        a = el.attrib
        self.path, self.area, self.kind = a["path"], a["area"], a["kind"]
        self.code = int(a["code"])
        self.imports = [dict(i.attrib) for i in el.findall("import")]
        self.index = {}
        for i in self.imports:
            self.index.setdefault(i["name"], (i.get("target"), i.get("external")))
        tree = el.find("ast/Module")
        self.doc = ""
        body = [c for c in tree if c.get("field") == "body"] if tree is not None else []
        if body and body[0].tag == "Expr":
            v = next(iter(body[0]), None)
            if v is not None and v.tag == "Constant":
                self.doc = v.get("value", "")
        self.aliases = self._aliases(tree) if tree is not None else {}
        self.defs = {}                                                   # qual -> (lines, calls [raw], class)
        self.raw = {}                                                    # node name -> [raw call]
        if tree is None:
            return
        top = {c.get("name") for c in body if c.tag in FUNCS + ("ClassDef",)}
        self.top = top
        mod_calls = []
        for c in body:
            if c.tag in FUNCS:
                self._func(c, c.get("name"), None)
            elif c.tag == "ClassDef":
                cls = c.get("name")
                self.defs[cls] = int(c.get("end_lineno", 0)) - int(c.get("lineno", 0)) + 1
                self.raw[cls] = [self._resolve(x, None) for x in calls_in(c, True)]
                if any(m.tag in FUNCS and m.get("name") == "__init__" for m in c):
                    self.raw[cls].append(("int", self.path, f"{cls}.__init__"))
                for m in c:
                    if m.tag in FUNCS:
                        self._func(m, f"{cls}.{m.get('name')}", cls)
            else:
                mod_calls.extend(self._resolve(x, None) for x in calls_in(c, True))
        self.raw["<module>"] = mod_calls
        self.defs["<module>"] = 0

    def _func(self, el, qual, cls):
        self.defs[qual] = int(el.get("end_lineno", 0)) - int(el.get("lineno", 0)) + 1
        self.raw[qual] = [self._resolve(x, cls) for x in calls_in(el, False)]

    def _aliases(self, tree):
        out = {}
        for n in tree.iter():
            if n.tag == "ImportFrom":
                mod, level = n.get("module") or "", int(n.get("level", "0"))
                for al in n.findall("alias"):
                    name, local = al.get("name"), al.get("asname") or al.get("name")
                    if name == "*":
                        continue
                    pre = "." * level
                    sub = self.index.get(pre + (mod + "." if mod else "") + name)
                    if sub and sub[0]:
                        out[local] = ("mod", sub[0], None)
                        continue
                    hit = self.index.get(pre + mod)
                    if hit and hit[0]:
                        out[local] = ("obj", hit[0], name)
                    elif not level:
                        out[local] = ("ext", f"{mod}.{name}", None)
            elif n.tag == "Import":
                for al in n.findall("alias"):
                    name = al.get("name")
                    local = al.get("asname") or name.split(".")[0]
                    hit = self.index.get(name)
                    if hit and hit[0]:
                        out[local] = ("mod", hit[0], None) if al.get("asname") or "." not in name else \
                            ("pkg", name, None)
                    else:
                        out[local] = ("ext", name if al.get("asname") else name.split(".")[0], None)
        return out

    def _resolve(self, call, cls):
        parts = dotted(next((c for c in call if c.get("field") == "func"), None))
        if not parts:
            return ("none",)
        b = parts[0]
        if len(parts) == 1:
            if b in getattr(self, "top", ()):
                return ("int", self.path, b)
            al = self.aliases.get(b)
            if al:
                if al[0] == "obj":
                    return ("int", al[1], al[2])
                if al[0] == "ext":
                    return ("ext", al[1])
            return ("builtin", b) if b in BUILTINS else ("attr", b)
        if b in ("self", "cls") and cls and len(parts) == 2:
            return ("int", self.path, f"{cls}.{parts[1]}")
        al = self.aliases.get(b)
        if al:
            if al[0] == "mod":
                return ("int", al[1], parts[1])
            if al[0] == "obj":
                return ("int", al[1], f"{al[2]}.{parts[1]}")
            if al[0] == "ext":
                return ("ext", ".".join([al[1]] + parts[1:]))
            if al[0] == "pkg":                                          # import a.b; a.b.f()
                for i in range(len(parts) - 1, 0, -1):
                    hit = self.index.get(".".join(parts[:i]))
                    if hit and hit[0]:
                        return ("int", hit[0], parts[i])
        if b in getattr(self, "top", ()):                               # Class.method()
            return ("int", self.path, f"{b}.{parts[1]}")
        return ("attr", parts[-1])


def load(path: Path):
    mods, entries = {}, []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        for ev, el in ET.iterparse(f, events=("end",)):
            if el.tag == "module":
                m = Module(el)
                mods[m.path] = m
                el.clear()
            elif el.tag == "entry":
                entries.append(dict(el.attrib))
    return mods, entries


def products_of(raw) -> list[tuple[str, str]]:
    """(product, mechanism library) for a raw call."""
    if raw[0] == "ext":
        name = raw[1]
        out = []
        for prod, (prefixes, attrs) in PRODUCTS.items():
            if name.startswith(prefixes) or name.split(".")[-1] in attrs:
                out.append((prod, name.split(".")[0]))
        return out
    if raw[0] == "attr":
        return [(prod, ATTR_LIB.get(raw[1], "?")) for prod, (_, attrs) in PRODUCTS.items() if raw[1] in attrs]
    if raw[0] == "builtin" and raw[1] == "open":
        return [("data files", "builtins")]
    return []


# ----------------------------------------------------------------------------------------------------------- graphs
def build(mods):
    dep = nx.DiGraph()
    for p, m in mods.items():
        dep.add_node(p, area=m.area, kind=m.kind, code=m.code)
    for p, m in mods.items():
        for i in m.imports:
            t = i.get("target")
            if t and t != p and t in mods:
                if dep.has_edge(p, t):
                    dep[p][t]["dynamic"] = dep[p][t]["dynamic"] and i.get("dynamic") == "1"
                else:
                    dep.add_edge(p, t, dynamic=i.get("dynamic") == "1")
    calls = nx.DiGraph()
    effects = defaultdict(set)                                          # node -> {(product, lib)}
    for p, m in mods.items():
        for q, lines in m.defs.items():
            calls.add_node(f"{p}::{q}", module=p, area=m.area, kind=m.kind, lines=lines)
    methods = defaultdict(lambda: defaultdict(list))                   # module -> method name -> quals
    for p, m in mods.items():
        for q in m.defs:
            if "." in q:
                methods[p][q.split(".", 1)[1]].append(q)
    near = {}                                                           # module -> modules a dispatch may land in
    for p in mods:
        direct = {t for t in dep.successors(p)}
        two = {u for t in direct if t.endswith("__init__.py") for u in dep.successors(t)}
        near[p] = {p} | direct | two
    for p, m in mods.items():
        for q, raws in m.raw.items():
            src = f"{p}::{q}"
            for r in raws:
                if r[0] == "attr":                                      # x.foo(): methods named foo nearby
                    for t in near[p]:
                        for mq in methods[t].get(r[1], ()):
                            dst = f"{t}::{mq}"
                            if dst != src and not calls.has_edge(src, dst):
                                calls.add_edge(src, dst, kind="dispatch")
                if r[0] == "int":
                    tgt_mod = mods.get(r[1])
                    if tgt_mod is None:
                        continue
                    name = r[2]
                    if name not in tgt_mod.defs and "." in name and name.split(".")[0] in tgt_mod.defs:
                        name = name.split(".")[0]                       # Class.unknown_attr -> the class
                    if name in tgt_mod.defs:
                        dst = f"{r[1]}::{name}"
                        if dst != src:
                            calls.add_edge(src, dst, kind="resolved")
                else:
                    effects[src].update(products_of(r))
    return dep, calls, effects


def entry_starts(entries, mods, calls):
    """Entry → its start nodes in the call graph (every function and the top level of the entry's module)."""
    out = {}
    for e in entries:
        m = mods.get(e["module"])
        if m is None:
            continue
        key = f"{e['family']}|{e['kind']}|{e['name']}"
        out[key] = (e, [f"{m.path}::{q}" for q in m.defs if calls.has_node(f"{m.path}::{q}")])
    return out


def entropy(counter: Counter) -> float:
    n = sum(counter.values())
    if n == 0 or len(counter) < 2:
        return 0.0
    h = -sum(c / n * math.log2(c / n) for c in counter.values() if c)
    return h / math.log2(len(counter))


def ranking(scores: dict, limit=10, fmt="{:.3f}"):
    return [(k, v) for k, v in sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:limit]]


# ---------------------------------------------------------------------------------------------------------- analyse
def analyse(mods, entries):
    dep, calls, effects = build(mods)
    out = {}

    # --- standard: module dependencies
    fan_out, fan_in = dict(dep.out_degree()), dict(dep.in_degree())
    inst = {n: fan_out[n] / (fan_in[n] + fan_out[n]) if fan_in[n] + fan_out[n] else 0.0 for n in dep}
    sccs = [sorted(s) for s in nx.strongly_connected_components(dep) if len(s) > 1]
    cycles = [c for c in itertools.islice(nx.simple_cycles(dep, length_bound=6), 50)]
    pr = nx.pagerank(dep)
    bc = nx.betweenness_centrality(dep)
    blast = {n: len(nx.ancestors(dep, n)) for n in dep}
    lib = {n for n in dep if dep.nodes[n]["kind"] in ("lib", "script") and dep.nodes[n]["code"] > 0}
    area_g = nx.DiGraph()
    for a, b in dep.edges():
        x, y = dep.nodes[a]["area"], dep.nodes[b]["area"]
        if x != y:
            w = area_g[x][y]["weight"] + 1 if area_g.has_edge(x, y) else 1
            area_g.add_edge(x, y, weight=w)
    a_out = {a: area_g.out_degree(a) for a in area_g}
    a_in = {a: area_g.in_degree(a) for a in area_g}
    a_inst = {a: a_out[a] / (a_in[a] + a_out[a]) for a in area_g if a_in[a] + a_out[a]}
    cond = nx.condensation(area_g)
    topo = [sorted(cond.nodes[c]["members"]) for c in nx.topological_sort(cond)]
    out["modules"] = {
        "nodes": dep.number_of_nodes(), "edges": dep.number_of_edges(),
        "dynamic_edges": sum(1 for *_, d in dep.edges(data="dynamic") if d),
        "density": nx.density(dep), "weak_components": nx.number_weakly_connected_components(dep),
        "strong_components_gt1": sccs,
        "fan_out": ranking(fan_out), "fan_in": ranking(fan_in),
        "instability_of_most_used": ranking({n: inst[n] for n in sorted(dep, key=lambda n: -fan_in[n])[:30]}, 30),
        "cycles": [c + [c[0]] for c in cycles[:15]],
        "pagerank": ranking(pr), "betweenness": ranking(bc), "blast_radius": ranking(blast),
        "isolated_lib_modules": sorted(n for n in nx.isolates(dep) if n in lib),
        "area_instability": dict(sorted(a_inst.items(), key=lambda x: x[1])),
        "area_topological_order_users_first": topo,
    }

    # --- standard: calls
    k = min(400, calls.number_of_nodes())
    cbc = nx.betweenness_centrality(calls, k=k, seed=1, normalized=False)
    rec = [sorted(s) for s in nx.strongly_connected_components(calls)
           if len(s) > 1 or any(calls.has_edge(n, n) for n in s)]
    out["calls"] = {
        "nodes": calls.number_of_nodes(), "edges": calls.number_of_edges(),
        "dispatch_edges": sum(1 for *_, k in calls.edges(data="kind") if k == "dispatch"),
        "most_called": ranking(dict(calls.in_degree())), "most_calling": ranking(dict(calls.out_degree())),
        "betweenness_sampled": ranking(cbc), "recursion_groups": rec[:20],
        "functions_with_effects": sum(1 for v in effects.values() if v),
    }

    # --- SERVE
    starts = entry_starts(entries, mods, calls)
    reach = {}
    serve = defaultdict(set)                                            # node -> entry keys
    for key, (e, st) in starts.items():
        r = set(st)
        for s in st:
            r |= nx.descendants(calls, s)
        reach[key] = r
        for n in r:
            serve[n].add(key)
    fam = {key: e["family"] for key, (e, _) in starts.items()}
    non_test = [k for k in starts if fam[k] != "tests"]
    area_serve = defaultdict(set)
    area_lines = Counter()
    for n, d in calls.nodes(data=True):
        area_lines[d["area"]] += d["lines"]
        area_serve[d["area"]] |= serve.get(n, set())
    serve_rows = {}
    for a in sorted(area_lines, key=lambda a: -area_lines[a]):
        S = area_serve[a]
        fams = Counter(fam[k] for k in S if fam[k] != "tests")
        serve_rows[a] = {
            "lines_in_defs": area_lines[a], "entries": len(S), "non_test_entries": sum(fams.values()),
            "breadth": round(sum(fams.values()) / max(1, len(non_test)), 3),
            "owner": fams.most_common(1)[0][0] if fams else ("tests" if S else "nothing"),
            "purpose_entropy": round(entropy(fams), 3), "families": dict(fams.most_common()),
            "test_only": bool(S) and not fams,
        }
    dead_lines = sum(d["lines"] for n, d in calls.nodes(data=True) if not serve.get(n) and d["kind"] in ("lib",))
    test_only_lines = sum(d["lines"] for n, d in calls.nodes(data=True)
                          if serve.get(n) and all(fam[k] == "tests" for k in serve[n]) and d["kind"] == "lib")
    lib_lines = sum(d["lines"] for n, d in calls.nodes(data=True) if d["kind"] == "lib")
    out["serve"] = {"areas": serve_rows, "lib_lines_in_defs": lib_lines,
                    "lib_lines_unreached_by_calls": dead_lines, "lib_lines_reached_only_by_tests": test_only_lines}

    # --- PURPOSE
    prod_entries = defaultdict(set)
    profile = {}
    for key in starts:
        p = Counter()
        for n in reach[key]:
            for prod, _ in effects.get(n, ()):
                p[prod] += 1
        profile[key] = p
        for prod in p:
            prod_entries[prod].add(key)
    purpose = {}
    for prod in PRODUCTS:
        S = [k for k in prod_entries[prod] if fam[k] != "tests"]
        producers = [n for n, eff in effects.items() if any(x[0] == prod for x in eff)]
        purpose[prod] = {"entries": len(S), "share_of_entries": round(len(S) / max(1, len(non_test)), 3),
                         "families": dict(Counter(fam[k] for k in S).most_common()),
                         "producer_functions": len(producers),
                         "producer_lines": sum(calls.nodes[n]["lines"] for n in producers if calls.has_node(n)),
                         "producer_areas": dict(Counter(calls.nodes[n]["area"] for n in producers
                                                        if calls.has_node(n)).most_common(6))}
    fam_profile = defaultdict(Counter)
    for key in non_test:
        for prod in profile[key]:
            fam_profile[fam[key]][prod] += 1
    declared = {}
    for a in serve_rows:
        cands = [m for m in mods.values() if m.area == a and m.doc and m.kind in ("lib", "script")]
        if cands:
            best = max(cands, key=lambda m: (dep.in_degree(m.path), m.path.endswith("__init__.py"), m.code))
            declared[a] = {"module": best.path, "doc": _first_sentence(best.doc)}
    out["purpose"] = {"products": dict(sorted(purpose.items(), key=lambda x: -x[1]["entries"])),
                      "family_profiles": {f: dict(c.most_common()) for f, c in fam_profile.items()},
                      "declared": declared,
                      "entries_per_family": dict(Counter(fam[k] for k in starts).most_common())}

    # --- HOW
    G = calls.copy()
    entry_nodes = []
    for key in non_test:
        v = f"entry:{key}"
        G.add_node(v, area=f"[{fam[key]}]", kind="entry", lines=0, module="")
        G.add_edges_from((v, s) for s in starts[key][1])
        entry_nodes.append(v)
    how = {}
    for prod in PRODUCTS:
        targets = [n for n, eff in effects.items() if any(x[0] == prod for x in eff) and G.has_node(n)]
        if not targets:
            continue
        sb = nx.betweenness_centrality_subset(G, entry_nodes, targets, normalized=False)
        tot = sum(v for n, v in sb.items() if not n.startswith("entry:")) or 1.0
        by_mod, by_area = Counter(), Counter()
        for n, v in sb.items():
            if v and not n.startswith("entry:"):
                by_mod[G.nodes[n]["module"]] += v / tot
                by_area[G.nodes[n]["area"]] += v / tot
        tset = set(targets)
        depths, dist_area = [], defaultdict(list)
        for v in entry_nodes:
            lengths = nx.single_source_shortest_path_length(G, v)
            hit = [d for n, d in lengths.items() if n in tset]
            if hit:
                depths.append(min(hit) - 1)
                for n, d in lengths.items():
                    if n in tset or sb.get(n):
                        dist_area[G.nodes[n]["area"]].append(d - 1)
        mech = Counter()
        for n in targets:
            w = len([k for k in serve.get(n, ()) if fam[k] != "tests"])
            for p, lib_name in effects[n]:
                if p == prod and w:
                    mech[lib_name] += w
        chain_areas = [a for a, s in by_area.items() if s >= 0.05]
        tgt_areas = Counter(G.nodes[n]["area"] for n in targets
                            if any(fam[k] != "tests" for k in serve.get(n, ())))
        for a, c in tgt_areas.most_common(2):
            if a not in chain_areas:
                chain_areas.append(a)
        chain_areas.sort(key=lambda a: sum(dist_area[a]) / len(dist_area[a]) if dist_area[a] else 99)
        fams = purpose[prod]["families"]
        how[prod] = {"entries": len(depths), "mean_depth": round(sum(depths) / len(depths), 2) if depths else None,
                     "funnel_modules": [(m, round(s, 3)) for m, s in by_mod.most_common(8)],
                     "funnel_areas": [(a, round(s, 3)) for a, s in by_area.most_common(8)],
                     "mechanism": [(m, c) for m, c in mech.most_common(5)],
                     "producer_areas": dict(tgt_areas.most_common(5)),
                     "chain": " → ".join([f"[{'/'.join(list(fams)[:3])}]"] + chain_areas +
                                         [f"({'/'.join(m for m, _ in mech.most_common(3))})"])}
    out["how"] = how
    # what the anatomy drawing needs: who enters where, area imports, which area makes what for how many entries
    enter = Counter()
    for key in non_test:
        e = starts[key][0]
        enter[(fam[key], mods[e["module"]].area)] += 1
    makes = Counter()
    for prod in PRODUCTS:
        for key in non_test:
            areas_hit = {calls.nodes[n]["area"] for n in reach[key] if any(x[0] == prod for x in effects.get(n, ()))}
            for a in areas_hit:
                makes[(a, prod)] += 1
    out["drawing"] = {"enter": [[f, a, n] for (f, a), n in enter.items()],
                      "imports": [[x, y, d["weight"]] for x, y, d in area_g.edges(data=True)],
                      "makes": [[a, p, n] for (a, p), n in makes.items()]}
    return out, dep, calls


# ---------------------------------------------------------------------------------------------------------- drawing
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]       # fixed order
INK, INK2, MUTED, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#fcfcfb"


def draw_anatomy(st, path: Path):
    """Who starts Vegeta's code → the areas (layered by imports, users left) → what comes out.
    Box height ~ sqrt(lines in functions); area colour = the family that reaches it most; edge width ~ sqrt(count)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, PathPatch
    from matplotlib.path import Path as MPath

    serve = st["serve"]["areas"]
    dr = st["drawing"]
    fams_n = st["purpose"]["entries_per_family"]
    fams = [f for f in fams_n if f != "tests"]
    colour = {f: SERIES[i] if i < len(SERIES) else MUTED for i, f in enumerate(fams)}
    areas = [a for a, r in serve.items() if r["lines_in_defs"] > 0 and r["non_test_entries"] > 0 and a != "tools"]
    g = nx.DiGraph()
    g.add_nodes_from(areas)
    imports = [(x, y, w) for x, y, w in dr["imports"] if x in areas and y in areas and w >= 3]
    g.add_weighted_edges_from(imports)
    cond = nx.condensation(g)
    layer_c = {}
    for c in nx.topological_sort(cond):
        layer_c[c] = max((layer_c[p] + 1 for p in cond.predecessors(c)), default=0)
    layer = {a: layer_c[cond.graph["mapping"][a]] + 1 for a in areas}
    products = [p for p, r in st["purpose"]["products"].items() if r["entries"] > 0]
    nlay = max(layer.values()) + 2
    cols = defaultdict(list)
    for f in fams:
        cols[0].append(("fam", f))
    for a in areas:
        cols[layer[a]].append(("area", a))
    for p in products:
        cols[nlay].append(("prod", p))
    size = {}
    maxl = max(serve[a]["lines_in_defs"] for a in areas)
    for f in fams:
        size[("fam", f)] = 0.35 + 0.9 * math.sqrt(fams_n[f] / max(fams_n.values()))
    for a in areas:
        size[("area", a)] = 0.3 + 1.1 * math.sqrt(serve[a]["lines_in_defs"] / maxl)
    maxp = max(st["purpose"]["products"][p]["entries"] for p in products)
    for p in products:
        size[("prod", p)] = 0.3 + 0.9 * math.sqrt(st["purpose"]["products"][p]["entries"] / maxp)
    edges = [(("fam", f), ("area", a), n) for f, a, n in dr["enter"] if f in fams and a in areas]
    edges += [(("area", x), ("area", y), w) for x, y, w in imports]
    mk = [(("area", a), ("prod", p), n) for a, p, n in dr["makes"] if a in areas and p in products]
    top = max((n for *_, n in mk), default=1)
    edges += [e for e in mk if e[2] >= max(3, 0.08 * top)]
    # order inside columns: barycentre sweeps
    order = {c: list(v) for c, v in cols.items()}
    order[nlay].sort(key=lambda n: -st["purpose"]["products"][n[1]]["entries"])
    pos_i = {n: i for c in order for i, n in enumerate(order[c])}
    for _ in range(4):
        for c in sorted(order)[1:-1] + sorted(order)[1:-1][::-1]:
            def bary(n):
                nb = [(pos_i[u], w) for u, v, w in edges if v == n] + [(pos_i[v], w) for u, v, w in edges if u == n]
                return sum(p * w for p, w in nb) / sum(w for _, w in nb) if nb else pos_i[n]
            order[c].sort(key=bary)
            pos_i.update({n: i for i, n in enumerate(order[c])})
    gap, colw = 0.35, 5.0
    xy = {}
    height = max(sum(size[n] for n in v) + gap * (len(v) - 1) for v in order.values())
    for c, v in order.items():
        total = sum(size[n] for n in v) + gap * (len(v) - 1)
        y = (height + total) / 2
        for n in v:
            xy[n] = (c * colw, y - size[n] / 2)
            y -= size[n] + gap
    fig, ax = plt.subplots(figsize=(3.0 * (nlay + 1) + 1, height * 0.62 + 1.6), dpi=100)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    W = 2.3

    def col_of(n):
        if n[0] == "fam":
            return colour[n[1]]
        if n[0] == "area":
            return colour.get(serve[n[1]]["owner"], MUTED)
        return INK2

    maxw = max(w for *_, w in edges)
    for u, v, w in edges:
        (x1, y1), (x2, y2) = xy[u], xy[v]
        back = x2 <= x1
        xa, xb = x1 + W / 2, (x2 - W / 2) if not back else (x2 + W / 2)
        if back:                                                    # same layer (a cycle collapsed): an arc right
            verts = [(xa, y1), (xa + 1.2, y1), (xb + 1.2, y2), (xb, y2)]
        else:
            verts = [(xa, y1), ((xa + xb) / 2, y1), ((xa + xb) / 2, y2), (xb, y2)]
        ax.add_patch(PathPatch(MPath(verts, [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4]),
                               fc="none", ec=col_of(u), lw=0.5 + 6 * math.sqrt(w / maxw), alpha=0.28, zorder=1))
    for n, (x, y) in xy.items():
        h = size[n]
        ax.add_patch(FancyBboxPatch((x - W / 2, y - h / 2), W, h, boxstyle="round,pad=0,rounding_size=0.08",
                                    fc=col_of(n), ec=SURFACE, lw=2, zorder=2))
        if n[0] == "fam":
            sub = f"{fams_n[n[1]]} entry points"
        elif n[0] == "area":
            r = serve[n[1]]
            sub = f"{r['lines_in_defs']:,} lines · {r['non_test_entries']} entries"
        else:
            r = st["purpose"]["products"][n[1]]
            sub = f"{r['entries']} entries ({r['share_of_entries']:.0%})"
        label = n[1].removeprefix("vegeta.") if n[0] == "area" else n[1]
        ax.text(x, y + (0.08 if h > 0.5 else 0.0), label, ha="center", va="center", fontsize=8.5, color="white",
                weight="bold", zorder=3)
        if h > 0.5:
            ax.text(x, y - 0.17, sub, ha="center", va="center", fontsize=6.5, color="white", zorder=3)
        else:
            ax.text(x + W / 2 + 0.08, y, sub, ha="left", va="center", fontsize=6.5, color=INK2, zorder=3)
    for c, title in ((0, "WHO starts it"), (1, "WHAT runs (areas; left imports right)"),
                     (nlay, "WHAT comes out (products)")):
        ax.text(c * colw - W / 2, height + 0.9, title, fontsize=10, color=INK, weight="bold", ha="left")
    ax.set_xlim(-W, nlay * colw + W + 2.2)
    ax.set_ylim(-0.6, height + 1.4)
    ax.axis("off")
    fig.suptitle("Vegeta's anatomy: who starts it, what it runs, what comes out", x=0.01, ha="left", fontsize=13,
                 color=INK)
    ax.text(-W / 2, -0.45, "Area colour = the family whose entry points reach it most. Edge width ~ entry points "
            "(who→area, area→product) or imports (area→area, ≥3). Tool packages are vegeta.<name>. Tests and tools "
            "left out.",
            fontsize=7.5, color=INK2)
    fig.savefig(path, format=path.suffix[1:], bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def _first_sentence(doc: str) -> str:
    s = " ".join(doc.split())
    m = re.match(r"(.+?[.:!?])(\s|$)", s)
    return (m.group(1) if m else s)[:200]


# ------------------------------------------------------------------------------------------------------------ print
def report(st):
    m, c = st["modules"], st["calls"]
    P = print
    P("=== MODULE DEPENDENCY GRAPH (A -> B: A imports B) ===")
    P(f"Modules: {m['nodes']} | Import edges: {m['edges']} ({m['dynamic_edges']} dynamic) | Density: {m['density']:.4f}")
    P(f"Weakly connected components: {m['weak_components']} | Strongly connected (>1 module): "
      f"{len(m['strong_components_gt1'])}")
    for title, key in (("Fan-out / efferent coupling", "fan_out"), ("Fan-in / afferent coupling", "fan_in"),
                       ("PageRank", "pagerank"), ("Betweenness", "betweenness"),
                       ("Change blast radius (transitive importers)", "blast_radius")):
        P(f"\n{title}:")
        for n, v in m[key][:8]:
            P(f"  {n:<62} {v:.3f}" if isinstance(v, float) else f"  {n:<62} {v}")
    P("\nImport cycles:")
    for cy in m["cycles"][:10] or [["None"]]:
        P("  " + " -> ".join(cy))
    P("\nArea instability (0 = depended on, 1 = depends on others):")
    for a, v in m["area_instability"].items():
        P(f"  {a:<22} {v:.2f}")
    P("\nIsolated library modules:", m["isolated_lib_modules"] or "None")
    P("\n=== APPROXIMATE FUNCTION CALL GRAPH (caller -> callee) ===")
    P(f"Functions: {c['nodes']} | Call edges: {c['edges']} ({c['dispatch_edges']} by method name) | "
      f"with third-party effects: {c['functions_with_effects']}")
    for title, key in (("Most called", "most_called"), ("Calling most others", "most_calling"),
                       ("Call betweenness (orchestration hotspots, sampled)", "betweenness_sampled")):
        P(f"\n{title}:")
        for n, v in c[key][:8]:
            P(f"  {n:<80} {v:.1f}" if isinstance(v, float) else f"  {n:<80} {v}")
    P("\nRecursion groups:", c["recursion_groups"][:5] or "None")

    s = st["serve"]
    P("\n=== WHAT DOES IT SERVE ===")
    P(f"{'area':<20} {'lines':>6} {'entries':>7} {'breadth':>7} {'entropy':>7}  owner         families")
    for a, r in s["areas"].items():
        P(f"{a:<20} {r['lines_in_defs']:>6} {r['non_test_entries']:>7} {r['breadth']:>7.2f} {r['purpose_entropy']:>7.2f}  "
          f"{r['owner']:<13} {r['families']}{'  [TEST-ONLY]' if r['test_only'] else ''}")
    P(f"library lines unreached by any entry's calls: {s['lib_lines_unreached_by_calls']} / {s['lib_lines_in_defs']}; "
      f"reached only by tests: {s['lib_lines_reached_only_by_tests']}")

    p = st["purpose"]
    P("\n=== WHAT IS THE PURPOSE (products the entry points reach) ===")
    for prod, r in p["products"].items():
        P(f"  {prod:<20} {r['entries']:>4} entries ({r['share_of_entries']:.0%})  {r['families']}")
    P("\nDeclared purpose (docstring of each area's most imported module):")
    for a, d in p["declared"].items():
        P(f"  {a:<20} {d['doc']}")

    P("\n=== HOW IS IT ACHIEVED (mechanism funnels) ===")
    for prod, h in st["how"].items():
        P(f"\n{prod}: {h['entries']} entries, mean depth {h['mean_depth']}")
        P(f"  chain:     {h['chain']}")
        P("  funnels:   " + ", ".join(f"{a} {v:.0%}" for a, v in h["funnel_areas"][:5]))
        P("  modules:   " + ", ".join(f"{Path(x).name} {v:.0%}" for x, v in h["funnel_modules"][:5]))
        P(f"  mechanism: {h['mechanism']}")


def _rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("xml", nargs="?", default=str(ROOT / "out" / "vegeta_ast.xml.gz"))
    ap.add_argument("--out", default=str(ROOT / "docs" / "anatomy"))
    ap.add_argument("--graphml", default=str(ROOT / "out"))
    a = ap.parse_args(argv)
    mods, entries = load(Path(a.xml))
    st, dep, calls = analyse(mods, entries)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "graph_metrics.json").write_text(json.dumps(st, indent=1, default=list) + "\n")
    Path(a.graphml).mkdir(parents=True, exist_ok=True)
    nx.write_graphml(dep, Path(a.graphml) / "module_dependencies.graphml")
    nx.write_graphml(calls, Path(a.graphml) / "function_calls.graphml")
    draw_anatomy(st, Path(a.out) / "anatomy.svg")
    report(st)
    print(f"\nSaved: {_rel(Path(a.out) / 'graph_metrics.json')}, {_rel(Path(a.out) / 'anatomy.svg')}, "
          f"{_rel(Path(a.graphml))}/module_dependencies.graphml, function_calls.graphml")
    return 0


if __name__ == "__main__":
    sys.exit(main())
