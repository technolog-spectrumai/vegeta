import ast
import json
import multiprocessing as mp
import os
import subprocess
import sys
from pathlib import Path

import pytest

import porunga
from porunga import Corrupt

HERE = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------------------------------- keys
def test_keys_ignore_dictionary_order_and_follow_every_value():
    a = porunga.key({"x": 1, "y": {"b": [1, 2.5], "a": "s"}})
    b = porunga.key({"y": {"a": "s", "b": (1, 2.5)}, "x": 1})
    assert a == b and len(a) == 64
    assert porunga.key({"x": 1, "y": {"a": "s", "b": [1, 2.5000001]}}) != a
    assert porunga.key({"x": 1}) != porunga.key({"x": 1.0}) != porunga.key({"x": "1"})
    assert porunga.key({"s": {3, 1, 2}}) == porunga.key({"s": [1, 2, 3]})
    assert porunga.key({"p": Path("a/b")}) == porunga.key({"p": "a/b"})


def test_keys_are_the_same_in_another_process():
    params = {"z": [1, {"q": None, "a": True}], "a": "é"}
    code = f"import sys; sys.path.insert(0, {str(HERE / 'src')!r}); import porunga; print(porunga.key({params!r}))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                         env={**os.environ, "PYTHONHASHSEED": "123"}).stdout.strip()
    assert out == porunga.key(params)


def test_unsupported_parameters_are_refused():
    with pytest.raises(TypeError):
        porunga.key({"f": object()})
    with pytest.raises(TypeError):
        porunga.key({1: "non-string key"})
    with pytest.raises(ValueError):
        porunga.key({"x": float("nan")})


# ------------------------------------------------------------------------------------------- store basics
def test_miss_then_save_then_hit_without_recomputing(tmp_path):
    calls = []

    def compute():
        calls.append(1)
        return {"n": 42}

    k = porunga.key({"what": "answer"})
    for _ in range(3):
        if porunga.check(k, root=tmp_path):
            e = porunga.load(k, root=tmp_path)
        else:
            e = porunga.save(k, compute(), metadata={"by": "test"}, root=tmp_path)
        assert e.data == {"n": 42}
    assert len(calls) == 1
    assert e.metadata == {"by": "test"}


def test_load_of_a_missing_key_is_a_key_error(tmp_path):
    k = porunga.key({"none": 0})
    assert not porunga.check(k, root=tmp_path)
    with pytest.raises(KeyError):
        porunga.load(k, root=tmp_path)


def test_files_and_opaque_metadata_round_trip(tmp_path):
    src = tmp_path / "src"
    (src / "sub").mkdir(parents=True)
    (src / "a.csv").write_text("x,y\n1,2\n")
    (src / "sub" / "b.bin").write_bytes(b"\0\1\2")
    meta = {"anything": [1, {"nested": "kept as given"}]}
    k = porunga.key({"files": 1})
    porunga.save(k, {"ok": True}, metadata=meta, files=src, root=tmp_path / "c")
    e = porunga.load(k, root=tmp_path / "c")
    assert sorted(e.files) == ["a.csv", "sub/b.bin"] and e.files["sub/b.bin"].read_bytes() == b"\0\1\2"
    assert e.metadata == meta
    k2 = porunga.key({"files": 2})
    e2 = porunga.save(k2, None, files={"renamed.csv": src / "a.csv"}, root=tmp_path / "c")
    assert e2.files["renamed.csv"].read_text() == "x,y\n1,2\n" and e2.data is None


def test_root_from_configure_and_environment(tmp_path, monkeypatch):
    k = porunga.key({"r": 1})
    monkeypatch.setenv("PORUNGA_DIR", str(tmp_path / "env"))
    porunga.save(k, 1)
    assert (tmp_path / "env" / k / "manifest.json").is_file()
    porunga.configure(tmp_path / "conf")
    try:
        assert not porunga.check(k)
        porunga.save(k, 2)
        assert porunga.load(k).data == 2
    finally:
        porunga.configure(None)
    monkeypatch.delenv("PORUNGA_DIR")
    with pytest.raises(ValueError):
        porunga.check(k)


# ------------------------------------------------------------------------------------------- integrity
def _entry(tmp_path, n=1):
    src = tmp_path / f"f{n}.txt"
    src.write_text("hello world\n" * 100)
    k = porunga.key({"n": n})
    porunga.save(k, {"n": n}, files={"f.txt": src}, root=tmp_path / "c")
    return k, tmp_path / "c" / k


def test_missing_or_truncated_files_are_not_hits(tmp_path):
    k, d = _entry(tmp_path, 1)
    (d / "files" / "f.txt").unlink()
    assert not porunga.check(k, root=tmp_path / "c")
    with pytest.raises(Corrupt):
        porunga.load(k, root=tmp_path / "c")
    k, d = _entry(tmp_path, 2)
    p = d / "files" / "f.txt"
    p.write_bytes(p.read_bytes()[:10])
    assert not porunga.check(k, root=tmp_path / "c")


def test_same_size_damage_is_found_by_verify(tmp_path):
    k, d = _entry(tmp_path)
    p = d / "files" / "f.txt"
    raw = bytearray(p.read_bytes()); raw[0] ^= 0xFF; p.write_bytes(bytes(raw))
    assert porunga.check(k, root=tmp_path / "c")
    assert not porunga.check(k, root=tmp_path / "c", verify=True)


def test_broken_manifest_is_not_a_hit_and_is_replaced(tmp_path):
    k, d = _entry(tmp_path)
    (d / "manifest.json").write_text("{not json")
    assert not porunga.check(k, root=tmp_path / "c")
    porunga.save(k, {"n": "again"}, root=tmp_path / "c")
    assert porunga.load(k, root=tmp_path / "c").data == {"n": "again"}


def test_an_intact_entry_is_kept_when_another_writer_comes_second(tmp_path):
    k = porunga.key({"first": "wins"})
    porunga.save(k, "first", root=tmp_path)
    assert porunga.save(k, "second", root=tmp_path).data == "first"


def test_a_failed_replacement_leaves_the_existing_entry(tmp_path):
    k, d = _entry(tmp_path)
    (d / "files" / "f.txt").unlink()                               # damaged: a replacement is due
    with pytest.raises(FileNotFoundError):
        porunga.save(k, {"n": 9}, files={"f.txt": tmp_path / "does-not-exist"}, root=tmp_path / "c")
    assert (d / "manifest.json").is_file()                         # untouched, still there (and still not a hit)
    assert not list((tmp_path / "c" / ".tmp").iterdir())
    k2, d2 = _entry(tmp_path, 7)
    with pytest.raises(TypeError):
        porunga.save(k2, {"bad": object()}, root=tmp_path / "c")
    assert porunga.load(k2, root=tmp_path / "c").data == {"n": 7}


def _interrupted(root, k):
    import porunga.store as st
    real = st._write

    def dying(path, text):
        if path.name == "manifest.json":
            os._exit(3)                                            # killed between the data and the manifest
        return real(path, text)
    st._write = dying
    porunga.save(k, {"never": "visible"}, root=root)


def test_an_interrupted_write_is_never_visible(tmp_path):
    k = porunga.key({"interrupted": 1})
    p = mp.get_context("fork").Process(target=_interrupted, args=(tmp_path, k))
    p.start(); p.join()
    assert p.exitcode == 3
    assert not porunga.check(k, root=tmp_path) and not (tmp_path / k).exists()
    assert list((tmp_path / ".tmp").iterdir())                     # the leftover staging directory …
    assert porunga.clean_partial(tmp_path) == 1                    # … removed only on request
    porunga.save(k, {"now": "visible"}, root=tmp_path)
    assert porunga.load(k, root=tmp_path).data == {"now": "visible"}


def _writer(root, k, i, q):
    src = Path(root) / f"src{i}.bin"
    src.write_bytes(os.urandom(1 << 16))
    e = porunga.save(k, {"writer": i}, files={"blob.bin": src}, root=root)
    q.put(e.data["writer"])


def test_concurrent_writers_publish_exactly_one_intact_entry(tmp_path):
    k = porunga.key({"race": True})
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    ps = [ctx.Process(target=_writer, args=(tmp_path, k, i, q)) for i in range(8)]
    for p in ps:
        p.start()
    for p in ps:
        p.join()
    seen = {q.get() for _ in ps}
    assert len(seen) == 1                                          # every writer got the one published entry
    assert porunga.check(k, root=tmp_path, verify=True)
    assert porunga.load(k, root=tmp_path).data == {"writer": seen.pop()}
    assert [p.name for p in tmp_path.iterdir() if not p.name.startswith((".", "src"))] == [k]


def test_bad_keys_and_file_names_are_refused(tmp_path):
    with pytest.raises(ValueError):
        porunga.check("not-a-key", root=tmp_path)
    with pytest.raises(ValueError):
        porunga.save(porunga.key({}), 1, files={"../escape": __file__}, root=tmp_path)


# --------------------------------------------------------------------------------------------- standalone
def test_standard_library_only_and_knows_no_host():
    allowed = set(sys.stdlib_module_names) | {"porunga"}
    for p in (HERE / "src" / "porunga").glob("*.py"):
        for node in ast.walk(ast.parse(p.read_text())):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                    [node.module] if isinstance(node, ast.ImportFrom) and node.level == 0 else []
            assert all(n.split(".")[0] in allowed for n in names), (p.name, names)


def test_the_example_counts_once_then_reuses(tmp_path):
    txt = tmp_path / "t.txt"
    txt.write_text("the cat and the dog and the bird\n")
    run = lambda *a: subprocess.run([sys.executable, str(HERE / "example.py"), str(txt), "--cache", str(tmp_path / "c"), *a],
                                    capture_output=True, text=True, check=True).stdout
    assert run().startswith("miss") and run().startswith("hit") and run("--upper").startswith("miss")
