"""Vegeta as one XML file: every module's syntax tree, plus an index layer (lines, imports, definitions, entry points).

    python tools/dump_ast.py                      # -> out/vegeta_ast.xml.gz (the whole tree; ~1 min)
    python tools/dump_ast.py --no-body            # the index layer only (seconds)
    python tools/dump_ast.py --root DIR --out F   # any other tree (the smoke test dumps a fixture)

Reads every ``*.py`` and the code cells of every ``*.ipynb`` (IPython magics and shell lines blanked). Stdlib only.

The file::

    <vegeta root= revision= generated= modules= >
      <module path= area= dotted= kind="lib|test|notebook|script|vendor" main= sha1= lines= code= comment= docstring= blank=>
        <import name= line= target="path/of/internal.py" | external="numpy" dynamic= />
        <def name= qual= line= end= lines= kind="function|class|method" decorators= calls="a b.c ..."/>
        <error>…</error>                                                         a file or cell that does not parse
        <ast> <Module> <FunctionDef field="body" name= lineno= …> … </ast>       one element per ast node
      </module>
      <entrypoints>
        <entry kind="cli|main|notebook|test|shell" name= module= family= />      family: who starts it (family_of)
      </entrypoints>
    </vegeta>

``tools/ast_report.py`` reads only this file and answers: what is there, what connects to what, which paths the
entry points funnel through, and what each part serves.
"""
from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import time
import tokenize
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr

SKIP_DIRS = {".venv", "__pycache__", ".git", ".ipynb_checkpoints", "runs", "_runs", "build", "dist", "node_modules",
             ".pytest_cache", "output", "results", "cache", "out"}
MAGIC = re.compile(r"^\s*[%!]")


# ------------------------------------------------------------------------------------------------------------ files
def walk(root: Path):
    for d, dirs, files in os.walk(root):
        dirs[:] = sorted(x for x in dirs if x not in SKIP_DIRS and not x.endswith(".egg-info"))
        for f in sorted(files):
            if f.endswith((".py", ".ipynb")):
                yield Path(d) / f


def notebook_source(path: Path) -> str:
    """The code cells, one after another, magics blanked (so line numbers count cell lines)."""
    nb = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for c in nb.get("cells", []):
        if c.get("cell_type") != "code":
            continue
        src = c.get("source", "")
        src = "".join(src) if isinstance(src, list) else src
        out.append("\n".join("" if MAGIC.match(ln) else ln for ln in src.splitlines()))
    return "\n\n".join(out) + "\n"


def area_of(rel: Path) -> str:
    """The part of Vegeta a file belongs to: vegeta.<pkg> for the packages, the top-level directory otherwise."""
    p = rel.parts
    if p[0] == "libs" and len(p) > 2:                                         # libs/<dist>/…: the library layer
        if p[1] == "labkit":
            return "vegeta.labkit"
        p = p[1:]
    if p[0] == "labs" and len(p) > 2:                                         # labs/<lab>/…
        return "labs/" + p[1]
    if p[0].startswith("vegeta-"):
        if "src" in p and len(p) > p.index("src") + 2:
            return "vegeta." + p[p.index("src") + 2].removesuffix(".py")
        if len(p) > 2 and p[1] == "tests":                                   # libs/vegeta-cli/tests/<pkg>/…
            pkg = p[2] if len(p) > 3 else p[0].removeprefix("vegeta-")
            return "vegeta." + pkg
        return "vegeta." + p[0].removeprefix("vegeta-")
    if p[0] == "vendor" and len(p) > 1:
        return p[1]
    return p[0] if len(p) > 1 else "root"


def kind_of(rel: Path) -> str:
    n, parts = rel.name, rel.parts
    if parts[0] == "vendor":
        return "vendor"
    if n.endswith(".ipynb"):
        return "notebook"
    if n.startswith("test_") or n.endswith("_test.py") or n == "conftest.py" or "tests" in parts:
        return "test"
    return "lib"


# ------------------------------------------------------------------------------------------------------ module index
def package_roots(root: Path) -> list[Path]:
    """Directories on sys.path for absolute imports: every */src (the distributions), plus the repository root."""
    roots = [p for p in root.rglob("src") if p.is_dir() and not (set(p.relative_to(root).parts) & SKIP_DIRS)]
    labs = [d for d in sorted(root.glob("labs/*/*")) if d.is_dir() and d.name in ("designs", "common", "exp", "sims")]
    return sorted(roots) + labs + [root]                                     # a lab's folders are on its sys.path


def dotted_index(root: Path, files: list[Path]) -> dict[str, Path]:
    idx = {}
    for r in package_roots(root):
        for f in files:
            if f.suffix != ".py":
                continue
            try:
                rel = f.relative_to(r)
            except ValueError:
                continue
            if r == root and rel.parts[0] in ("vendor",) or set(rel.parts[:-1]) & {"src"}:
                continue
            parts = list(rel.with_suffix("").parts)
            if parts[-1] == "__init__":
                parts = parts[:-1]
            if parts and all(x.isidentifier() for x in parts):
                idx.setdefault(".".join(parts), f)
    return idx


class Resolver:
    """Import name → internal file, by the rules Vegeta's code relies on: packages under */src and the repository root,
    then flat modules next to the importer, then a unique flat module anywhere (the designs, labs and notebooks put
    their folders on sys.path)."""

    def __init__(self, root: Path, files: list[Path]):
        self.root = root
        self.dotted = dotted_index(root, files)
        self.stems: dict[str, list[Path]] = {}
        for f in files:
            if f.suffix == ".py" and f.stem != "__init__":
                self.stems.setdefault(f.stem, []).append(f)
        for f in files:                                                      # flat packages: dir/__init__.py
            if f.name == "__init__.py":
                self.stems.setdefault(f.parent.name, []).append(f)

    def resolve(self, name: str, importer: Path, level: int = 0) -> tuple[Path | None, str]:
        if level:
            base = importer.parent
            for _ in range(level - 1):
                base = base.parent
            cand = base.joinpath(*name.split(".")) if name else base
            for c in (cand.with_suffix(".py"), cand / "__init__.py"):
                if c.exists():
                    return c, "relative"
            return None, ""
        parts = name.split(".")
        for i in range(len(parts), 0, -1):
            hit = self.dotted.get(".".join(parts[:i]))
            if hit is not None:
                return hit, "package"
        head = parts[0]
        for d in (importer.parent, importer.parent.parent):                  # next to the importer (or one up)
            for i in range(len(parts), 0, -1):                               # gui.app -> gui/app.py, then gui/
                c = d.joinpath(*parts[:i])
                for f in (c.with_suffix(".py"), c / "__init__.py"):
                    if f.exists() and f != importer:
                        return f, "local"
        hits = self.stems.get(head, [])
        if len(hits) == 1:
            return hits[0], "flat"
        if hits:                                                             # ambiguous: the closest one
            imp = importer.relative_to(self.root).parts
            best = max(hits, key=lambda h: len(os.path.commonprefix([imp, h.relative_to(self.root).parts])))
            return best, "guess"
        return None, ""


# -------------------------------------------------------------------------------------------------------- the index
def line_counts(src: str, tree: ast.AST) -> dict:
    lines = src.splitlines()
    doc = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                doc.update(range(first.lineno, first.end_lineno + 1))
    comment = set()
    code_lines = set()
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type == tokenize.COMMENT:
                comment.add(tok.start[0])
            elif tok.type not in (tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT,
                                  tokenize.ENDMARKER, tokenize.ENCODING):
                code_lines.update(range(tok.start[0], tok.end[0] + 1))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass
    code_lines -= doc
    blank = sum(1 for ln in lines if not ln.strip())
    return {"lines": len(lines), "code": len(code_lines), "comment": len(comment - code_lines - doc),
            "docstring": len(doc), "blank": blank}


def has_main(tree: ast.Module) -> bool:
    for n in tree.body:
        if isinstance(n, ast.If) and isinstance(n.test, ast.Compare) and isinstance(n.test.left, ast.Name) \
                and n.test.left.id == "__name__":
            return True
    return False


def dotted_name(node) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = dotted_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    if isinstance(node, ast.Call):
        return dotted_name(node.func)
    return None


def imports(tree: ast.AST):
    """(name, level, line, dynamic) for import statements, ``from X import y`` (as X.y first), importlib calls."""
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                yield a.name, 0, n.lineno, False, None
        elif isinstance(n, ast.ImportFrom):
            mod = n.module or ""
            yield mod, n.level, n.lineno, False, [a.name for a in n.names]
        elif isinstance(n, ast.Call) and dotted_name(n.func) in ("importlib.import_module", "import_module", "__import__") \
                and n.args:
            if isinstance(n.args[0], ast.Constant) and isinstance(n.args[0].value, str):
                yield n.args[0].value, 0, n.lineno, True, None
            else:                                                            # a name from a table: any module-like
                yield "*", 0, n.lineno, True, None                           # string in the file may be it


def definitions(tree: ast.Module):
    def visit(body, prefix, in_class):
        for n in body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                qual = f"{prefix}{n.name}"
                kind = "class" if isinstance(n, ast.ClassDef) else ("method" if in_class else "function")
                calls = sorted({c for c in (dotted_name(x.func) for x in ast.walk(n) if isinstance(x, ast.Call)) if c})
                decos = [dotted_name(d) or "?" for d in n.decorator_list]
                yield {"name": n.name, "qual": qual, "line": n.lineno, "end": n.end_lineno,
                       "lines": n.end_lineno - n.lineno + 1, "kind": kind, "decorators": " ".join(decos),
                       "calls": " ".join(calls)}
                yield from visit(n.body, qual + ".", isinstance(n, ast.ClassDef))
    yield from visit(tree.body, "", False)


# ------------------------------------------------------------------------------------------------------ the AST body
def ast_xml(node, out, field=None, depth=0):
    attrs = [f" field={quoteattr(field)}"] if field else []
    children = []
    for name, value in ast.iter_fields(node):
        if isinstance(value, ast.AST):
            children.append((name, value))
        elif isinstance(value, list):
            scalars = [str(v) for v in value if v is not None and not isinstance(v, ast.AST)]
            children.extend((name, v) for v in value if isinstance(v, ast.AST))
            if scalars:                                                      # global a, b -> names="a b"
                attrs.append(f" {name}={quoteattr(' '.join(scalars)[:200])}")
        elif value is not None:
            s = value if isinstance(value, str) else repr(value)
            attrs.append(f" {name}={quoteattr(s[:200])}")
    for a in ("lineno", "end_lineno"):
        if hasattr(node, a):
            attrs.append(f' {a}="{getattr(node, a)}"')
    tag = type(node).__name__
    if children:
        out.write(f"<{tag}{''.join(attrs)}>")
        for name, ch in children:
            ast_xml(ch, out, name, depth + 1)
        out.write(f"</{tag}>")
    else:
        out.write(f"<{tag}{''.join(attrs)}/>")


# ---------------------------------------------------------------------------------------------------- entry points
def project_scripts(root: Path):
    import tomllib
    for pp in sorted(root.glob("*/pyproject.toml")):
        if pp.parent.name == "vendor":
            continue
        try:
            scripts = tomllib.loads(pp.read_text()).get("project", {}).get("scripts", {})
        except (tomllib.TOMLDecodeError, OSError):
            continue
        for name, target in scripts.items():
            yield name, target.split(":")[0]


SHELL_PY = re.compile(r"(?:python3?|\$PY|\"\$PY\"|\$py|\"\$py\")\s+(?:-[a-zA-Z]+\s+)*(-m\s+[\w.]+|[\w./${}-]+\.py)")


def shell_entries(root: Path):
    for sh in sorted(list(root.glob("*.sh")) + list(root.glob("scripts/*.sh"))):
        for m in SHELL_PY.finditer(sh.read_text(errors="replace")):
            tgt = m.group(1)
            yield sh.relative_to(root).as_posix(), tgt


def family_of(rel: Path, kind: str) -> str:
    """Who starts code here: people through notebooks and design scripts, the CLI, agents, labs, races, tests."""
    a = area_of(rel)
    if kind == "test":
        return "tests"
    if rel.parts[0] == "labs" and len(rel.parts) > 3:                       # in a lab: the stage it runs from
        return {"studies": "notebooks", "exp": "exp", "sims": "sims", "campaigns": "campaigns"}.get(rel.parts[2], "designs")
    if kind == "notebook":
        return "notebooks"
    return {"designs": "designs", "races": "races", "scenarios": "scenarios", "side_labs": "side_labs",
            "benchmark": "benchmark", "vegeta.fidia": "agents", "vegeta.ai": "agents", "examples": "examples",
            "scripts": "scripts", "notebooks": "notebooks", "tools": "tools"}.get(
        a, "cli" if a.startswith("vegeta.") else "vendor" if rel.parts[0] == "vendor" else "scripts")


# -------------------------------------------------------------------------------------------------------------- main
def git_revision(root: Path) -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True, text=True,
                              timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def dump(root: Path, out_path: Path, body: bool = True) -> dict:
    root = root.resolve()
    files = list(walk(root))
    res = Resolver(root, files)
    dotted_of = {v: k for k, v in res.dotted.items()}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if out_path.suffix == ".gz" else open
    stats = {"modules": 0, "errors": 0, "code": 0}
    entries = []
    with opener(out_path, "wt", encoding="utf-8") as out:
        out.write(f'<?xml version="1.0" encoding="utf-8"?>\n<vegeta root={quoteattr(str(root))} '
                  f'revision={quoteattr(git_revision(root))} generated={quoteattr(time.strftime("%Y-%m-%dT%H:%M:%S"))} '
                  f'modules="{len(files)}">\n')
        for f in files:
            mod_xml = io.StringIO()
            rel = f.relative_to(root)
            kind = kind_of(rel)
            err = None
            try:
                src = notebook_source(f) if f.suffix == ".ipynb" else f.read_text(encoding="utf-8", errors="replace")
            except (json.JSONDecodeError, OSError) as e:
                src, err = "", f"read: {e}"
            try:
                tree = ast.parse(src, filename=str(rel))
            except SyntaxError as e:
                tree, err = ast.Module(body=[], type_ignores=[]), f"{type(e).__name__}: {e}"
            lc = line_counts(src, tree)
            main = has_main(tree) or rel.name == "__main__.py"                 # python -m starts __main__.py
            sha = hashlib.sha1(src.encode()).hexdigest()[:12]
            if kind == "lib" and main:
                kind = "script"
            dotted = dotted_of.get(f, "")
            mod_xml.write(f"<module path={quoteattr(rel.as_posix())} area={quoteattr(area_of(rel))} "
                          f"dotted={quoteattr(dotted)} kind={quoteattr(kind)} main=\"{int(main)}\" sha1=\"{sha}\""
                          + "".join(f' {k}="{v}"' for k, v in lc.items()) + ">\n")
            strings = {c.value for c in ast.walk(tree) if isinstance(c, ast.Constant) and isinstance(c.value, str)
                       and c.value.isidentifier()}
            for name, level, line, dynamic, names in imports(tree):
                targets = []
                if name == "*":                                              # importlib with a variable
                    for sname in sorted(strings):
                        hit, how = res.resolve(sname, f)
                        if hit is not None and how == "local":               # a table of sibling modules
                            mod_xml.write(f"  <import name={quoteattr(sname)} line=\"{line}\" "
                                          f"target={quoteattr(hit.relative_to(root).as_posix())} how=\"string\" dynamic=\"1\"/>\n")
                    continue
                if names and not level:                                      # from pkg import submodule
                    for n in names:
                        hit = res.dotted.get(f"{name}.{n}") if name else None
                        if hit is not None:
                            targets.append((f"{name}.{n}", hit, "package"))
                if level:
                    for n in (names or []):
                        hit, how = res.resolve(f"{name}.{n}" if name else n, f, level)
                        if hit is not None:
                            targets.append(((("." * level) + (name + "." if name else "") + n), hit, how))
                if not targets:
                    hit, how = res.resolve(name, f, level)
                    targets.append((("." * level) + name, hit, how))
                for tname, hit, how in targets:
                    d = ' dynamic="1"' if dynamic else ""
                    if hit is not None:
                        mod_xml.write(f"  <import name={quoteattr(tname)} line=\"{line}\" "
                                      f"target={quoteattr(hit.relative_to(root).as_posix())} how=\"{how}\"{d}/>\n")
                    else:
                        mod_xml.write(f"  <import name={quoteattr(tname)} line=\"{line}\" "
                                      f"external={quoteattr(tname.lstrip('.').split('.')[0])}{d}/>\n")
            for c in ast.walk(tree):                                         # "../designs/velutina.py": loaded by path
                if isinstance(c, ast.Constant) and isinstance(c.value, str) and c.value.endswith(".py") and "\n" not in c.value:
                    hit = (f.parent / c.value).resolve()
                    if hit.is_file() and hit != f.resolve() and root in hit.parents:
                        mod_xml.write(f"  <import name={quoteattr(c.value)} line=\"{c.lineno}\" "
                                      f"target={quoteattr(hit.relative_to(root).as_posix())} how=\"path\" dynamic=\"1\"/>\n")
            for dd in definitions(tree):
                mod_xml.write("  <def" + "".join(f" {k}={quoteattr(str(v))}" for k, v in dd.items()) + "/>\n")
            if err:
                stats["errors"] += 1
                mod_xml.write(f"  <error>{escape(err)}</error>\n")
            if body:
                mod_xml.write("  <ast>")
                ast_xml(tree, mod_xml)
                mod_xml.write("</ast>\n")
            mod_xml.write("</module>\n")
            if kind == "notebook":
                entries.append(("notebook", rel.as_posix(), rel.as_posix(), family_of(rel, kind)))
            elif main:
                entries.append(("main", rel.as_posix(), rel.as_posix(), family_of(rel, kind)))
            elif (kind == "test" and rel.name.startswith("test_")) or rel.name in ("conftest.py", "pytest_plugin.py"):
                entries.append(("test", rel.as_posix(), rel.as_posix(), "tests"))
            stats["modules"] += 1
            stats["code"] += lc["code"]
            out.write(mod_xml.getvalue())
        for name, target in project_scripts(root):
            hit = res.dotted.get(target)
            if hit is not None:
                entries.append(("cli", name, hit.relative_to(root).as_posix(), "cli"))
        for sh, tgt in shell_entries(root):
            if tgt.startswith("-m"):
                hit = res.dotted.get(tgt.split()[1])
            else:
                c = (root / tgt)
                hit = c if c.exists() else None
            if hit is not None:
                entries.append(("shell", f"{sh}: {tgt}", hit.relative_to(root).as_posix(), "shell"))
        out.write("<entrypoints>\n")
        for kind, name, module, fam in entries:
            out.write(f"  <entry kind={quoteattr(kind)} name={quoteattr(name)} module={quoteattr(module)} "
                      f"family={quoteattr(fam)}/>\n")
        out.write("</entrypoints>\n</vegeta>\n")
    stats["entries"] = len(entries)
    return stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--out", default=None, help="default: <root>/out/vegeta_ast.xml.gz (.gz compresses)")
    ap.add_argument("--no-body", action="store_true", help="the index layer only, no syntax trees")
    a = ap.parse_args(argv)
    root = Path(a.root)
    out = Path(a.out) if a.out else root / "out" / "vegeta_ast.xml.gz"
    t = time.time()
    s = dump(root, out, body=not a.no_body)
    print(f"{out}: {s['modules']} modules, {s['code']} code lines, {s['entries']} entry points, "
          f"{s['errors']} unparsable, {out.stat().st_size / 1e6:.1f} MB, {time.time() - t:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
