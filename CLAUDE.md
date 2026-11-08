# CLAUDE.md

Continual Backprop (CBP) for gym-continuousDoubleAuction, as the `cbp` extension under gym-CDA's
`gym_cda.extensions` entry point. Start at [README.md](README.md); the docs index is
[docs/README.md](docs/README.md), and [docs/development.md](docs/development.md) has the layout,
the install and the test inventory.

## Rules

1. **Settings live in `cda_cbp/config.py` `DEFAULTS`** and the `docs/settings.md` table, together;
   `tests/test_docs.py` checks the two agree.
2. **`third_party/loss-of-plasticity` is the authors' code**, installed unchanged as `lop`
   (`tests/test_lop.py` checks the bytes). Never edit or reformat it.
3. **Run `python -m pytest -q`** (about 20 s), `ruff format --check .` and
   `python -m pyflakes cda_cbp tests` before committing. gym-CDA must be installed first; see
   docs/development.md.
4. **Test first.** A fix lands with a regression test that fails on the previous source.
5. **Atomic commits**, one fix per commit: code, then tests, then docs.
6. **Reproduce before you claim** (skill `cbp-verify-finding`).
7. **Reviews file, they do not fix.** Each confirmed finding is a GitHub issue with a severity
   (`S1-blocking` to `S4-minor`), a kind (`bug`, `enhancement`, `documentation`, `test-gap`,
   `refactor`), an issue type (Bug, Task or Feature), and `agent` or `human`. An issue that needs
   both is split into sub-issues, each labelled.
8. **Docs follow the code** (skill `cbp-doc-sync`); record a change under `## Unreleased` in
   `CHANGELOG.md`, and never rewrite a released section.

## Facts worth not re-deriving

- `cbp.py` is the algorithm and imports no RLlib; `learner.py` is the RLlib wiring
  (`CBPLearnerMixin`: hooks, `apply_gradients`, metrics, `get_state`/`set_state`).
- A missing `optimizer` or `count_rule` means `torch_adam` / `nature` (an old checkpoint's
  behaviour); a new run's defaults are `adam_gnt` / `arxiv`.
- The default `mlp` network has four replaceable layers per module, two per network.
