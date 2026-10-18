# Not integrated

This folder is a verbatim copy of `mimesis/` from Alabastron (https://github.com/technolog-spectrumai/alabastron,
commit 7265e3c, 2026-10-18), grafted here for later use. **Nothing in Vegeta installs, imports or tests it**:
`install_local.sh`, the package `pyproject.toml` files, `vegeta.mock`, `scripts/test_all.sh` and the workflows are
unchanged. Do not wire it in without a plan: Vegeta's own mocks (`vegeta.mock`, Talos/Aeromant/Mellonia) work
differently.

To try it on its own: `python -m venv /tmp/m && /tmp/m/bin/pip install -e "mimesis[test]" && /tmp/m/bin/pytest mimesis/tests`.
