<img src="logo.png" alt="Ozaru" width="200">

# Ozaru

A simple common layer for **Vegeta** and **Alabastron**: three small libraries, standard library only, each usable on
its own, none of them aware of either host.

| Library | Import | What it does |
|---|---|---|
| [Porunga](porunga/README.md) | `porunga` | a generic local cache: `key(parameters)`, `check(key)`, `load(key)`, `save(key, data, metadata)` |
| [Mimesis](mimesis/README.md) | `mimesis` | fake computations: mock modes, labels, banners, isolation |
| [Yanus](yanus/README.md) | `yanus` | requirement checks: is a tool, a package, an environment variable there? |

Everything that knows about runs, books, chapters, notebooks or a host — side labs, run directories,
which parameters identify a result — lives in the host. When in doubt, it stays out of
Ozaru (`docs/boundaries.md`).

## How the hosts carry it

Vegeta and Alabastron keep Ozaru as committed code in `vendor/ozaru` (a `git subtree`), install it from there with
`pip install --no-deps --no-build-isolation --no-index -e vendor/ozaru/{porunga,mimesis,yanus}` inside their
`./install_local.sh`, and never fetch it during a build or a run. Shared changes are made here first, then pulled
into each host with its `scripts/update_ozaru.sh <ref>` (the only command that contacts this repository; it records
the revision in `vendor/OZARU_REVISION`).

## Tests

```bash
scripts/test.sh          # all three libraries and the layer's own rules, from the source tree, offline
```
