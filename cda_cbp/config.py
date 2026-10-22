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
    # Units replaced per mature unit per optimiser step (an accumulator, not a probability).
    "replacement_rate": 1e-4,
    # Optimiser steps a unit must age before it can be replaced.
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
    """Raise on a bad value, or a combination gym-CDA cannot run.

    Values are checked by building the algorithm's config. A run that also
    sets gym-CDA's own `cbp_enabled` or `cbp_metrics_only` (while gym-CDA still
    has them) would compose CBP twice, so it is refused. Replacing units with
    several learners is refused too: each learner picks the unit to replace
    from its own minibatch, outside the gradient, so the replicas would drift
    apart (gym-CDA docs/15 R-12, #182). `metrics_only` replaces nothing and is
    allowed.
    """
    to_cbp_config(settings)
    if getattr(train_config, "cbp_enabled", False) or getattr(
        train_config, "cbp_metrics_only", False
    ):
        raise ValueError(
            "extensions_enabled lists 'cbp' while cbp_enabled or cbp_metrics_only is also "
            "set; that would apply Continual Backprop twice. Use the extension only, with "
            "its settings under extension_settings['cbp']."
        )
    learners = getattr(train_config, "num_learners", 0) or 0
    if learners > 1 and not settings["metrics_only"]:
        raise ValueError(
            f"extension 'cbp' with num_learners={learners}: continual backprop picks units "
            "to replace from each learner's own minibatch, so the replicas would drift "
            "apart. Use num_learners 0 or 1, or metrics_only, until the choice is "
            "synchronised."
        )
