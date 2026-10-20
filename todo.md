# Ozaru — TODO

A simple common layer: Porunga (cache), Mimesis (mocks), Yanus (requirement checks). Stdlib only, no host knowledge.
Legend: `[ ]` open · `[x]` done · `[-]` deferred

## Stage 1 — The three utilities
- [x] 1.1 Porunga: `key` (canonical JSON → SHA-256, JSON text or value), `check`, `load`, `save` (atomic), `clean_partial`
- [x] 1.2 Porunga tests: deterministic keys, reuse, misses, missing/truncated files, interrupted and concurrent writes, failed replacement
- [x] 1.3 Porunga `docs/identity.md`: canonical form, reference implementation, test vectors (tested)
- [x] 1.4 Porunga example unrelated to science (`porunga/example.py`, word counts)
- [x] 1.5 Mimesis moved in from the hosts (identical copies)
- [x] 1.6 Yanus = requirement checks only (`Tool`, `Package`, `Env`, `AnyOf`, `check_needs`, `table`); the old framework stays out
- [x] 1.7 Rule tests: stdlib only, no host imports, the libraries never import each other
- [x] 1.8 Logo committed as a plain file (no LFS)
- [x] 1.9 README, `docs/boundaries.md`

## Stage 2 — Hosts carry it (vendor/ozaru by git subtree)
- [x] 2.1 Vegeta vendors Ozaru 33702e4 (signed subtree commits), `scripts/update_ozaru.sh`
- [x] 2.2 Vegeta installs porunga/mimesis/yanus offline from vendor/ozaru in `install_local.sh` (checked with networking off)
- [x] 2.3 Alabastron vendors the same revision (subtree 33702e4, signed, same script) and installs it offline
- [x] 2.4 Offline check of both hosts: clean clone, fresh venv, `unshare -n` (no network, proxy unset): the vendored Ozaru installs with `--no-index`, its 32 tests pass, `update_ozaru.sh --check` matches c4711e8, the hosts import their key modules and mocks and pass their separation tests; no GitHub fetch in install or activate (one comment only)

## Stage 2b — Yax (design interchange, Vegeta -> Piccolo / Nomios / Ketos)
- [x] 2b.1 `yax/`: spec v1 (`docs/spec.md`), package write/load/check, stdlib STL + exact mesh mass properties, units and frames, `yax check|info`
- [x] 2b.2 Test vector `examples/cube.yax` (exact), corruption, atomic replace, frames; `scripts/test.sh --smoke`
- [x] 2b.3 Vegeta writes Yax (`vegeta.dedalus.export_yax`), Piccolo reads it
- [x] 2b.4 Yax moved to Ogma (https://github.com/technolog-spectrumai/ogma) with its history: it belongs with the part physics; Ozaru stays three stdlib utilities

## Stage 3 — Later
- [ ] 3.1 Porunga: an `entries()` listing for maintenance scripts, if a host needs one
- [-] 3.2 Eviction — not planned; cleaning stays explicit
