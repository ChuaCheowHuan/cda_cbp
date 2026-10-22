# Moving from gym-CDA's built-in CBP

Continual Backprop was part of gym-CDA (`train/model/cbp.py`, `cbp_learner.py`, the
`continual_backprop` group of `train_config.json`) until it moved into this package. This page is
what changes for a run, a config and a checkpoint.

## A config

| Before (gym-CDA) | Now |
|---|---|
| `"cbp_enabled": true` | `"extensions_enabled": ["cbp"]` |
| `"cbp_metrics_only": true` | `"extensions_enabled": ["cbp"]` and `"extension_settings": {"cbp": {"metrics_only": true}}` |
| `"cbp_<key>": value` | `"extension_settings": {"cbp": {"<key>": value}}` |
| `"adam_betas"`, `"adam_weight_decay"` | unchanged: tuned Adam stayed in gym-CDA |

Every old key keeps its default ([settings.md](settings.md)). Two keys are new, `optimizer` and
`count_rule`; a run that does not set them gets the arXiv paper's behaviour, and a checkpoint that
does not record them keeps the old one (settings.md, Rules). A gym-CDA version without built-in CBP
rejects the old `cbp_*` keys with a message naming this package.

An equivalence test showed the two were the same mechanism while both existed: from the same
state, on the same episodes, the built-in CBP and this package produced identical weights,
optimiser state, CBP state and `cbp_*` metrics, with units replaced, for `mlp`, `jepa`, `lstm`
under `recurrent` scope, `metrics_only`, and `fire_on: "iteration"` with `compensate_mean`. It
was removed once gym-CDA no longer had a built-in CBP to compare with (CHANGELOG, Unreleased);
`tests/test_lop_reference.py` now compares against the authors' own implementation instead.

## A checkpoint

A checkpoint stores its Learner class, and that class names the mixins by module path. Nothing
about an old checkpoint has to be converted:

- **Written without CBP**: resumes as before, with or without this package installed.
- **Written with CBP on** (`cbp_enabled` or `cbp_metrics_only`): install this package. gym-CDA's
  `cbp_learner.CBPLearnerMixin` then resolves to this package's mixin, which reads the old
  `continual_backprop` settings stored in the checkpoint, so CBP resumes on, with the same settings
  and its restored per-unit state (the state key, `continual_backprop`, is unchanged). Without the
  package the resume is refused with a message naming it.
- **Written with tuned Adam only**: resumes as before; `TunedAdamMixin` is still in gym-CDA.

A resume cannot switch CBP on or off or change a setting: that is a fresh run, as it was before.
So the config that resumes a CBP checkpoint lists `"cbp"` in `extensions_enabled` and repeats its
settings, without the `cbp_` prefix, under `extension_settings.cbp` (gym-CDA reads the old group as
those settings). For example, a checkpoint trained with `cbp_enabled: true` and
`cbp_utility: "contribution"` resumes with:

```json
"extensions_enabled": ["cbp"],
"extension_settings": {"cbp": {"utility": "contribution"}}
```

Checked on a real checkpoint written before the move (16 replacements, `utility` "contribution"):
with the package and that config it resumed with the same per-unit ages and replacement count
and kept replacing; without the package, or with a config that leaves out `"cbp"`, gym-CDA
refused the resume and said why.

## Metrics

The metric names (`cbp_*`) and what they mean are unchanged ([metrics.md](metrics.md)).
