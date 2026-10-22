# Changelog

## Unreleased

- **The loss-of-plasticity code is a submodule.** `third_party/loss-of-plasticity` pins the
  authors' repository; the wheel installs its `lop`, `lop.utils` and `lop.algos` unchanged, and
  ships its licence. Clone with `--recurse-submodules` (docs/development.md).

## 0.1.0

- **Extracted from gym-continuousDoubleAuction.** Continual Backprop (`train/model/cbp.py`,
  `cbp_learner.py`, their tests and `docs/25_continual_backprop.md`) moved here with its git
  history, as the `cbp` extension under gym-CDA's `gym_cda.extensions` entry point.
- **Settings** are gym-CDA's `cbp_*` keys without the prefix, with the same defaults, under
  `extension_settings["cbp"]`; listing `"cbp"` in `extensions_enabled` replaces `cbp_enabled`.
- **Old checkpoints**: the mixin reads the legacy `continual_backprop` group, so a run written
  with gym-CDA's CBP resumes with CBP on.
- **Equivalence**: `tests/test_equivalence.py` shows identical weights, optimiser state, CBP state
  and metrics against gym-CDA's built-in CBP from the same state on the same episodes.
- Tuned Adam (`adam_betas`, `adam_weight_decay`) stayed in gym-CDA.
