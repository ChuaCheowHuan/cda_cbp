# cda_cbp

Continual Backprop (CBP) for [gym-continuousDoubleAuction](https://github.com/ChuaCheowHuan/gym-continuousDoubleAuction),
as an installable training extension.

Continual Backprop is backprop plus the continual, selective reinitialisation of low-utility hidden
units. It keeps a network able to learn when the problem keeps changing, which league self-play
does by construction: the opponents a policy trains against change under it. This package plugs
it into gym-CDA's PPO Learner through the `gym_cda.extensions` entry point. It was part of gym-CDA
until it moved here; [docs/migration.md](docs/migration.md) covers what changed.

- Dohare, Hernandez-Garcia, Lan, Rahman, Mahmood & Sutton, *Loss of plasticity in deep continual
  learning*, Nature 632, 2024.
- Dohare, Sutton & Mahmood, *Continual Backprop: Stochastic Gradient Descent with Persistent
  Randomness*, arXiv:2108.06325.

## Install

gym-CDA is not on PyPI. Install it first, then this package:

```bash
pip install "gym_continuousDoubleAuction @ git+https://github.com/ChuaCheowHuan/gym-continuousDoubleAuction"
pip install "git+https://github.com/ChuaCheowHuan/cda_cbp"
```

Installing changes no run.

## Enable

In the train config:

```json
"extensions_enabled": ["cbp"],
"extension_settings": {"cbp": {"replacement_rate": 1e-4}}
```

Every setting has a default, so `{}` is enough. To measure the correlates of plasticity loss
without replacing anything, set `"metrics_only": true`. [docs/settings.md](docs/settings.md) lists
every setting.

| | |
|---|---|
| Encoders | all of gym-CDA's; `lstm` needs `"scope": "recurrent"` (otherwise it has nothing to replace, and says so) |
| Learners | `num_learners` 0 or 1 when replacing; any number with `metrics_only` |
| Composition | wraps every other extension's Learner mixin (`learner_order` 100); keeps the MoE and JEPA loss terms |
| Resumes | the settings are fixed for a run; checkpoints written with gym-CDA's built-in CBP resume with it on |
| Tuned Adam | not here: `adam_betas` and `adam_weight_decay` are gym-CDA settings, independent of CBP |

## Read the metrics before believing it works

At this project's default budget, CBP is not expected to fire much, and a CBP that never fires
looks exactly like one that is working. Watch `cbp_replacements` and `cbp_mature_unit_frac` first
([docs/metrics.md](docs/metrics.md)). At ~20,000 optimiser steps no loss of plasticity was
detectable here; the measurement that matters is at the 10^6 to 10^8 steps the project needs
([docs/continual_backprop.md](docs/continual_backprop.md) §3.3).

## Documentation

[docs/README.md](docs/README.md) indexes the rest: the method and its fit, settings, metrics,
migration and development.

## Compatibility

Needs a gym-CDA with `learner_order` in its extension point and `train/model/layout.py`. CI tests
against the ref in [ci/cda_cbp.yml](ci/cda_cbp.yml).

## License

MIT, the same as gym-CDA, from which this code came.
