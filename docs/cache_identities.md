# Cache identities

Side labs publish each completed run's `summary.json` into the Porunga cache (`$PORUNGA_DIR`, default
`cache/porunga/`) under a key; notebooks derive the same key and load it. The key is `porunga.key(identity)`:
SHA-256 of the identity's canonical JSON (`vendor/ozaru/porunga/docs/identity.md`). Both sides build the identity
with `labs/aero/designs/kestrel_keys.py`. A changed design, campaign, setting or mock flag is a different key, so a stale or
mock result is never read as the current one.

| Identity | Fields |
|---|---|
| `vegeta.kestrel.airframe-cfd` v1 | `variant`, `campaign` (sweep, stall, …), `quality` (screening/fine), `design.params_sha256`; `design.prop_lab` for the rotor-disk campaigns (blown, cruise, transitions, all); `mock` |
| `vegeta.kestrel.structure-fea` v1 | `variant`, `campaign` (frame, joint, all), `settings.element_size_mm`, `settings.n_modes`, `design.params_sha256`; `mock` |

Payload: `data` = the run's summary (unchanged shape); metadata = `run_dir`, `run_id`, `mock`. Logs, cases and native
solver files stay in the run directory under `side_labs/runs/`.
