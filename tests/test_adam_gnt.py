"""CBP with Adam as arXiv 2108.06325v3 Algorithm 2 specifies it.

Algorithm 2 keeps a timestep per weight, not per tensor, and a replacement
sets the replaced weights' timesteps to zero along with both moment estimates,
so Adam's bias correction restarts for exactly those weights. `torch.optim.Adam`
keeps one scalar step per tensor and cannot do that; the authors' `AdamGnT`
(imported from `lop`, never copied) keeps a step tensor the shape of the
parameter. With `optimizer: "adam_gnt"`, the default for a new run, CBP builds
`AdamGnT` and the reset covers its step.

What these pin:

  * The step, like the moments, is zeroed for exactly the replaced slices, on
    a feed-forward layer and on an LSTM core.
  * torch Adam's scalar step is left alone, so `"torch_adam"` (and every
    checkpoint written before this setting existed) behaves as before.
  * A settings block without `optimizer` means torch Adam: a checkpoint is
    rebuilt from its own recorded settings, which lack the key.
  * Inside a real `Algorithm`, the Learner builds `AdamGnT` with gym-CDA's Adam
    settings when units are replaced, and torch Adam otherwise.
"""

import pytest
import ray
import torch
import torch.nn as nn
from conftest import TEST_CFG, cbp
from lop.utils.AdamGnT import AdamGnT

from gym_continuousDoubleAuction.train.train import TrainConfig, build_config

from cda_cbp import config as cbp_settings
from cda_cbp.cbp import (
    CBPConfig,
    CBPLayerState,
    RecurrentLayer,
    ReplaceableLayer,
    select_and_replace,
)
from cda_cbp.learner import _as_config


def _block():
    torch.manual_seed(0)
    seq = nn.Sequential(nn.Linear(6, 5), nn.Tanh(), nn.Linear(5, 3))
    return seq, ReplaceableLayer(
        name="block", incoming=seq[0], outgoing=[seq[2]], activation=seq[1]
    )


def _stepped(optimizer, module, x, steps=3):
    for _ in range(steps):
        optimizer.zero_grad()
        module(x).sum().backward()
        optimizer.step()


def _replace(layer, index, num_units, optimizer, scope="feedforward"):
    state = CBPLayerState.zeros(num_units)
    state.age[:] = 10
    utility = torch.full((num_units,), 5.0)
    utility[index] = 0.0
    config = CBPConfig(enabled=True, replacement_rate=1.0, maturity_threshold=0, scope=scope)
    replaced = select_and_replace(
        state,
        layer,
        utility,
        config,
        generator=torch.Generator().manual_seed(0),
        optimizer=optimizer,
    )
    assert replaced == [index]


class TestTheStepIsResetWithTheMoments:
    def test_feedforward_exactly_the_replaced_slices(self):
        seq, layer = _block()
        optimizer = AdamGnT(seq.parameters(), lr=1e-2)
        _stepped(optimizer, seq, torch.randn(4, 6))

        _replace(layer, 1, 5, optimizer)

        w_in = optimizer.state[seq[0].weight]
        b_in = optimizer.state[seq[0].bias]
        w_out = optimizer.state[seq[2].weight]
        others = [0, 2, 3, 4]
        for key in ("step", "exp_avg", "exp_avg_sq"):
            assert torch.all(w_in[key][1] == 0), key
            assert torch.all(b_in[key][1] == 0), key
            assert torch.all(w_out[key][:, 1] == 0), key
        assert torch.all(w_in["step"][others] == 3)
        assert torch.all(b_in["step"][others] == 3)
        assert torch.all(w_out["step"][:, others] == 3)

    def test_the_reset_weights_restart_adams_bias_correction(self):
        """After the reset, the replaced row's step counts from zero again."""
        seq, layer = _block()
        optimizer = AdamGnT(seq.parameters(), lr=1e-2)
        x = torch.randn(4, 6)
        _stepped(optimizer, seq, x)
        _replace(layer, 1, 5, optimizer)
        _stepped(optimizer, seq, x, steps=1)
        step = optimizer.state[seq[0].weight]["step"]
        assert torch.all(step[1] == 1) and torch.all(step[0] == 4)

    def test_lstm_gate_rows_recurrent_and_head_columns(self):
        torch.manual_seed(0)
        hidden = 4
        lstm = nn.LSTM(5, hidden, num_layers=2, batch_first=True)
        head = nn.Linear(hidden, 3)
        layer = RecurrentLayer("lstm", lstm, [head], layer_index=1)
        params = list(lstm.parameters()) + list(head.parameters())
        optimizer = AdamGnT(params, lr=1e-2)

        def forward(x):
            return head(lstm(x)[0])

        _stepped(optimizer, forward, torch.randn(3, 6, 5))
        index = 2
        _replace(layer, index, hidden, optimizer, scope="recurrent")

        rows = [g * hidden + index for g in range(4)]
        other_rows = [r for r in range(4 * hidden) if r not in rows]
        other_columns = [j for j in range(hidden) if j != index]
        for name in ("weight_ih_l1", "weight_hh_l1", "bias_ih_l1", "bias_hh_l1"):
            step = optimizer.state[getattr(lstm, name)]["step"]
            assert torch.all(step[rows] == 0), name
            kept = step[other_rows]
            if name == "weight_hh_l1":
                # The unit's own recurrent output is an input too: its column is reset.
                kept = kept[:, other_columns]
            assert torch.all(kept == 3), name
        assert torch.all(optimizer.state[lstm.weight_hh_l1]["step"][:, index] == 0)
        head_step = optimizer.state[head.weight]["step"]
        assert torch.all(head_step[:, index] == 0)
        assert torch.all(head_step[:, other_columns] == 3)

    def test_torch_adams_scalar_step_is_left_alone(self):
        seq, layer = _block()
        optimizer = torch.optim.Adam(seq.parameters(), lr=1e-2)
        _stepped(optimizer, seq, torch.randn(4, 6))

        _replace(layer, 1, 5, optimizer)

        assert float(optimizer.state[seq[0].weight]["step"]) == 3.0
        assert torch.all(optimizer.state[seq[0].weight]["exp_avg"][1] == 0)


class TestTheSetting:
    def test_a_new_run_defaults_to_adam_gnt(self):
        assert cbp_settings.DEFAULTS["optimizer"] == "adam_gnt"

    def test_a_block_without_it_means_torch_adam(self):
        """A checkpoint is rebuilt from its own settings, which lack the key."""
        recorded = {k: v for k, v in cbp_settings.DEFAULTS.items() if k != "optimizer"}
        lcd = {
            "extensions": {"extensions_enabled": ["cbp"], "extension_settings": {"cbp": recorded}}
        }
        assert _as_config(lcd).optimizer == "torch_adam"

    def test_the_legacy_group_means_torch_adam(self):
        assert _as_config({"continual_backprop": {"cbp_enabled": True}}).optimizer == "torch_adam"

    def test_an_unknown_value_is_refused(self):
        with pytest.raises(ValueError, match="optimizer"):
            CBPConfig(enabled=True, optimizer="sgd")


def _optimizers(train_kwargs):
    algo = build_config(TrainConfig(**TEST_CFG, **train_kwargs))[0].build_algo()
    try:
        return dict(algo.learner_group._learner._named_optimizers)
    finally:
        algo.stop()


class TestTheLearnerBuildsTheOptimiser:
    @classmethod
    def setup_class(cls):
        ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False, num_cpus=2)

    @classmethod
    def teardown_class(cls):
        ray.shutdown()

    def test_replacing_builds_adam_gnt(self):
        optimizers = _optimizers(cbp())
        assert optimizers and all(type(o) is AdamGnT for o in optimizers.values())

    def test_with_gym_cdas_adam_settings(self):
        optimizers = _optimizers(dict(**cbp(), adam_betas=[0.99, 0.99], adam_weight_decay=1e-4))
        for optimizer in optimizers.values():
            group = optimizer.param_groups[0]
            assert tuple(group["betas"]) == (0.99, 0.99)
            assert group["weight_decay"] == 1e-4

    def test_torch_adam_when_asked(self):
        optimizers = _optimizers(cbp(optimizer="torch_adam"))
        assert all(type(o) is torch.optim.Adam for o in optimizers.values())

    def test_metrics_only_keeps_torch_adam(self):
        """Measuring must not change the trajectory, the optimiser included."""
        optimizers = _optimizers(cbp(metrics_only=True))
        assert all(type(o) is torch.optim.Adam for o in optimizers.values())
