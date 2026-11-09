# Changelog

## Unreleased

- **Optimiser ownership and allocation (#22, #24).** `AdamGnT` is subclassed as `CBPAdamGnT`,
  which keeps the authors' update but builds its beta scalars once per param group instead of
  once per parameter per step. `select_and_replace` also caches the previous eligible count
  for `arxiv`, so the common step advances its counter without reading the count back from
  the device; the read happens only when a replacement becomes possible. A replacing run lets
  the base Learner register first, requires exactly one default `torch.optim.Adam`, replaces
  it with `CBPAdamGnT`, and raises if a base mixin registered a second optimizer or param
  group.
- **Agent fixes (2026-11-09).** #10 (LSTM padding), #11 (GAE pass), #12 (settings types),
  #14 (checkpoint RNG and counter), #19/#20 (CI pins and lint), #25 (version metadata),
  #27 (rank sample), #28 (saturation), #29 (dead check), #30 (CI hardening) and #31
  (several learners). The human-decision and GPU-measurement issues are untouched.
- **LSTM padding is excluded from CBP statistics (#10).** The hook applies RLlib's
  `loss_mask`, for the statistics and the rank sample, so a stateful encoder's zero-padded
  timesteps are not counted as activations.
- **The GAE value pass no longer enters the statistics (#11).** Collection is bracketed by
  `_uncompiled_update`, so the connector's `compute_values` forward is ignored rather than
  merged into the critic's first utility update.
- **Settings are type- and range-checked (#12).** `"metrics_only": "false"` and
  `"replacement_rate": "1e-4"` are refused instead of silently running the wrong
  experiment; `metrics_every_n_updates >= 1` and `dead_unit_threshold >= 0`.
  `compensate_mean` with `scope: "recurrent"` warns.
- **A resume keeps the replacement RNG and the update counter (#14).** Both ride in the
  checkpoint's CBP block, so the units drawn after a resume match an uninterrupted run and
  the metrics cadence does not restart.
- **CI pins both upstream repos and checks lint/types (#19, #20).** `GYM_CDA_REF` and
  cda-jepa are pinned by commit, a weekly scheduled run tracks `master`/`main`, and a lint
  job runs `ruff format --check .` and pyflakes while the test job runs mypy. A `dev` extra
  installs the tools.
- **The version comes from the installed metadata (#25).** `pyproject.toml`'s version is
  the one source; the test fails on a stale editable install.
- **The rank sample merges every call in a step (#27).** A layer that runs more than once
  (MoE experts, the JEPA trunk) no longer has the effective-rank metric measured on its
  last call alone.
- **Saturation is measured only for bounded activations (#28).** Tanh and sigmoid keep the
  metric; GELU/ReLU layers omit it, so a healthy strongly active GELU unit is not counted
  as saturated.
- **The dead "CBP twice" check is gone (#29).** No supported gym-CDA has `cbp_enabled` or
  `cbp_metrics_only`, and the branch could not run.
- **CI is hardened (#30).** The workflow has `contents: read`, actions are pinned by SHA,
  and Dependabot keeps the pins current.
- **Several learners can replace (#31).** Each layer's statistics are all-reduced across
  the learner group before the utility update, so every learner ranks and replaces the same
  units; the config no longer refuses `num_learners > 1`.
- **Review and tidy-up (2026-11-08).** Five new findings filed (#27-#31); the open review issues
  labelled `agent` or `human`, with #13, #15, #16 and #18 split into sub-issues (#19-#26).
- **Stale names fixed (#17).** The no-replaceable-layer warning names
  `extension_settings["cbp"]["scope"]` instead of gym-CDA's old `cbp_scope`; docstrings and
  `docs/continual_backprop.md` point at this repository's docs or name gym-CDA's;
  `docs/settings.md` says `replacement_rate` and `maturity_threshold` count iterations under
  `fire_on: "iteration"`.
- **Tests: 203 to 172, about twice as fast.** Duplicates, constant and source-text checks, and a
  test that skipped on every supported gym-CDA were removed; the tests that build an `Algorithm`
  share one Ray session, and the checkpoint tests read the same firing run as the update tests
  instead of training an identical one. docs/development.md lists the files.
- **Agent skills and CLAUDE.md.** `cbp-verify-finding` (with `hook_rows.py`, which prints what the
  hooks counted beside `loss_mask`) and `cbp-doc-sync` (with `check_docs.py`).

- **Docs brought up to date.** `docs/continual_backprop.md` names the settings as they are now
  (`metrics_only`, `scope`, ... rather than `cbp_*`), describes how gym-CDA composes the mixin
  through its extension point, and lists this repository's files; its banner is one paragraph.
  `docs/settings.md` drops the rule about gym-CDA's own CBP (no supported gym-CDA has it) and the
  notes that repeated the rules, and names the settings in the rest. `docs/metrics.md` links the
  cadence caveat here rather than to a gym-CDA section that no longer exists. The README draws how
  `lop`, this package and gym-CDA fit.

- **The `jepa` tests use whichever package provides `jepa`.** gym-CDA's own `jepa` (2.2 and
  earlier) or cda-jepa's, which the test extra now installs. The JEPA Learner is looked up
  through gym-CDA's registry instead of imported from `gym_continuousDoubleAuction.train.model.jepa_learner`,
  a config selecting cda-jepa's `jepa` enables its extension, and without either the tests skip
  (`tests/conftest.py`, `requires_jepa`). The research note links point at cda-jepa's `docs/jepa.md`.

## 0.2.0

- **Version 0.2.0.** gym-CDA records this beside every checkpoint that enables `cbp`. Checkpoints
  written on `main` between #5 and this bump say 0.1.0, though they ran these defaults; their
  recorded settings (`optimizer`, `count_rule`) tell them apart from 0.1.0 proper.
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
