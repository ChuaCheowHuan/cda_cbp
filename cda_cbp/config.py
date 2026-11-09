"""The `cbp` settings block: every key, its default, and the rules between them.

gym-CDA fills in these defaults and rejects any key not listed here before
`validate` sees the block. A run sets the block in its train config:

    "extensions_enabled": ["cbp"],
    "extension_settings": {"cbp": {"replacement_rate": 1e-4}}

Enabling the extension turns Continual Backprop on; `metrics_only` measures the
plasticity correlates without replacing anything. The keys are gym-CDA's old
`cbp_*` keys without the prefix, with the same defaults, and docs/settings.md
explains each one.
"""

from __future__ import annotations

from typing import Any, Dict

from cda_cbp.cbp import CBPConfig

#: The extension's name: its entry point, and what `extensions_enabled` lists.
NAME = "cbp"

DEFAULTS: Dict[str, Any] = {
    # Measure the correlates of plasticity loss, replace nothing.
    "metrics_only": False,
    # Units replaced per unit per step (an accumulator, not a probability); a step is an
    # optimiser step, or an iteration under fire_on "iteration". count_rule picks the units.
    "replacement_rate": 1e-4,
    # Steps a unit must age before it can be replaced (optimiser steps, or iterations under
    # fire_on "iteration").
    "maturity_threshold": 100,
    # Decay of the running utility average, in [0, 1).
    "utility_decay": 0.99,
    # "contribution", "mean_corrected" or "overall".
    "utility": "overall",
    # "feedforward", or "recurrent" to add an LSTM core's top layer.
    "scope": "feedforward",
    # "adam_step" (after every optimiser step) or "iteration".
    "fire_on": "adam_step",
    # Clear Adam's per-weight state (moments, and the timestep under adam_gnt) for a replaced unit.
    "reset_optimizer_state": True,
    # "adam_gnt" (the authors' AdamGnT: per-weight timestep, arXiv Algorithm 2) or "torch_adam".
    "optimizer": "adam_gnt",
    # "arxiv" (counter += n_l * rho, replace every whole unit it holds), "lop" (counter +=
    # |eligible| * rho, as the authors' GnT) or "nature" (|eligible| * rho, at most one per step).
    "count_rule": "arxiv",
    # Move a replaced unit's mean contribution into the next layer's bias.
    "compensate_mean": False,
    # Mean |activation| below which a unit counts as dead, for the metric only.
    "dead_unit_threshold": 0.01,
    # How often (in updates) the effective-rank metric is computed.
    "metrics_every_n_updates": 10,
}


def to_cbp_config(settings: Dict[str, Any]) -> CBPConfig:
    """The algorithm's config for an enabled extension with these settings."""
    return CBPConfig(enabled=True, **settings)


def validate(settings: Dict[str, Any], train_config) -> None:
    """Raise on a bad value.

    Values are checked by building the algorithm's config. Replacing with
    several learners is supported: each learner's statistics are all-reduced
    before the replacement choice, so the replicas rank and replace the same
    units (gym-CDA R-12, #182).
    """
    to_cbp_config(settings)
