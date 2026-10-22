# Changelog

## Unreleased

- **The loss-of-plasticity code is a submodule.** `third_party/loss-of-plasticity` pins the
  authors' repository; the wheel installs its `lop`, `lop.utils` and `lop.algos` unchanged, and
  ships its licence. Clone with `--recurse-submodules` (docs/development.md).
- **`optimizer` setting, `adam_gnt` by default.** When units are replaced, CBP builds the authors'
  `AdamGnT` (imported from `lop`) with gym-CDA's Adam settings, and a replacement resets the
  replaced weights' timestep as well as both moments, as arXiv Algorithm 2 specifies.
  `torch_adam` keeps the old behaviour. A settings block without the key (a checkpoint written
  before it) means `torch_adam`, so a resumed run trains as it did.
- **Tuned Adam keeps AdamGnT.** A run that sets gym-CDA's `adam_betas` or `adam_weight_decay`
  still trains with `AdamGnT`, with those values; it needs the gym-CDA that composes tuned Adam
  inside the extensions (gym-CDA #335). `test_adam_gnt.py` checks the optimiser's type both ways.
- **`count_rule` setting, `arxiv` by default.** `arxiv` follows arXiv Algorithms 1 and 2: the
  counter grows by `n_l · ρ` (the layer's width) while any unit is eligible, and a step replaces
  every whole unit it holds. `lop` is the authors' GnT with `accumulate=True`, which grows it by
  `|eligible| · ρ` and so replaces more slowly while units mature; `arxiv` meant this until #2.
  `nature` keeps the old cap of one per layer per step, and is what a settings block without the
  key means.
- **Heads other extensions add are reset too.** A replacement zeroes the new unit's outgoing
  weights in every head gym-CDA reports for the trunk, `TrunkOutput.extra_heads` included, so
  cda-discorl's `aux_pi` and `disco_heads` no longer read a replaced unit with the old unit's
  weights (#3). `tests/test_extra_heads.py`.
- **Needs gym-CDA 2.1** (`>=2.1,<3`), for `extra_trunk_heads` and lazy extension loading (#4).
- **Reference test against the authors' code.** `tests/test_lop_reference.py` runs this package's
  generate-and-test beside `lop.algos.gnt.GnT` for 60 steps and finds the same replacements, ages,
  utility ranking, network and Adam state. It also pins that lop's apparent bias compensation adds
  nothing (the unit's running mean is zeroed first).
- **Equivalence test removed.** `tests/test_equivalence.py` compared against gym-CDA's built-in
  CBP, which no longer exists, so it only ever skipped.
- **`effective_rank` is gym-CDA's.** The `cbp_batch_effective_rank` metric calls
  `train.probe.rank.effective_rank` (NaN for fewer than two rows) instead of a torch copy. Needs a
  gym-CDA with that function's `too_few_rows` argument and `optim_learner.adam_settings`.
- **Init bound from `lop`.** A replaced feed-forward unit's incoming weights use the authors'
  `get_layer_bound` rather than a copy of it.

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
