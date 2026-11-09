"""The optimiser-ownership rule chosen in #21 (implemented for #22).

CBP calls the base's `configure_optimizers_for_module` and then takes ownership
of exactly the one default `torch.optim.Adam` it registered. A base that
registers a second optimizer is refused rather than silently bypassed.
"""

from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from ray.rllib.core.learner.learner import DEFAULT_OPTIMIZER

from cda_cbp.learner import CBPLearnerMixin
from cda_cbp.optim import CBPAdamGnT


def _learner_config():
    return {
        "extensions": {
            "extensions_enabled": ["cbp"],
            "extension_settings": {"cbp": {"optimizer": "adam_gnt"}},
        }
    }


class _BaseLearner:
    """The parts of RLlib's Learner that the mixin's optimiser method uses."""

    def __init__(self, module, second_optimizer=False):
        self.module = {"m": module}
        self.config = SimpleNamespace(learner_config_dict=_learner_config())
        self.second_optimizer = second_optimizer
        self._module_optimizers = defaultdict(list)
        self._named_optimizers = {}
        self._optimizer_parameters = {}
        self._optimizer_lr_schedules = {}

    def get_parameters(self, module):
        return list(module.parameters())

    def register_optimizer(
        self,
        *,
        module_id,
        optimizer,
        params,
        lr_or_lr_schedule=None,
        optimizer_name=DEFAULT_OPTIMIZER,
    ):
        full_name = f"{module_id}_{optimizer_name}"
        self._module_optimizers[module_id].append(full_name)
        self._named_optimizers[full_name] = optimizer
        self._optimizer_parameters[optimizer] = list(params)

    def get_optimizers_for_module(self, module_id):
        return [
            (name[len(module_id) + 1 :], self._named_optimizers[name])
            for name in self._module_optimizers[module_id]
        ]

    def configure_optimizers_for_module(self, module_id, config=None):
        params = self.get_parameters(self.module[module_id])
        self.register_optimizer(
            module_id=module_id,
            optimizer=torch.optim.Adam(params, lr=0.1),
            params=params,
            lr_or_lr_schedule=0.1,
        )
        if self.second_optimizer:
            self.register_optimizer(
                module_id=module_id,
                optimizer=torch.optim.Adam(params, lr=0.1),
                params=params,
                lr_or_lr_schedule=0.1,
            )


class _Learner(CBPLearnerMixin, _BaseLearner):
    pass


def _module():
    return nn.Sequential(nn.Linear(4, 3), nn.Tanh(), nn.Linear(3, 2))


def test_the_default_optimizer_is_replaced_with_cbp_adam_gnt():
    learner = _Learner(_module())
    learner.configure_optimizers_for_module("m", config=SimpleNamespace(lr=0.05))

    registered = learner.get_optimizers_for_module("m")
    assert len(registered) == 1
    name, optimizer = registered[0]
    assert name == DEFAULT_OPTIMIZER
    assert type(optimizer) is CBPAdamGnT
    assert set(learner._optimizer_parameters) == {optimizer}


def test_a_base_with_a_second_optimizer_is_refused():
    learner = _Learner(_module(), second_optimizer=True)

    with pytest.raises(ValueError, match="registered optimizers"):
        learner.configure_optimizers_for_module("m", config=SimpleNamespace(lr=0.05))
