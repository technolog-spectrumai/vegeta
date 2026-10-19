# Mimesis in Vegeta — how mock results are labelled

`vendor/ozaru/mimesis/` (part of Ozaru, shared with Alabastron by git subtree) is a small package with no dependencies outside the standard library (`mimesis-mock`, import `mimesis`;
its own README and tests). It was copied from Alabastron, which uses it for its FEniCSx stubs.
It sets the convention for imitating expensive APIs: a stub result says it is one (`"mock": true`, the mimic's name,
a note), and a loud banner says when stubs are in force.

## In Vegeta

| piece | what it does here |
|---|---|
| `vegeta.mock` (the notebooks' `DEBUG` / `VEGETA_DEBUG` switch) | still patches Talos, Aeromant and Mellonia with fixed numbers. When it installs, it shows a `mimesis.banner`. `mock.label(d)` / `mock.is_mock(d)` label a dict made under the mocks. |
| `workflows/runkit.py` | every new run's `manifest.json` carries `"mock": true` or `false` (`runkit.mock_active()`: `VEGETA_DEBUG` on, or the mocks installed). A run resumed under the mocks becomes mock. |
| Yanus (saved-run reader) | refuses any run that `mimesis.is_mock(manifest)` marks as mock. Mock numbers are never read as engineering results. |

Vegeta has no Mimesis mimics registered yet (`mimesis.available()` is empty). `vegeta.mock` still patches the tools
directly. Moving its stubs to `@mimesis.mimic` with recorded schemas (`record_schema` / `check`) is a possible next
step, not done.

Install: `./install_local.sh` installs it offline from `vendor/ozaru/mimesis`. `activate.sh` puts its `src` on the path. Tests:
`scripts/test_all.sh` (or `vendor/ozaru/scripts/test.sh`). Change it in Ozaru, then `scripts/update_ozaru.sh <ref>`.
