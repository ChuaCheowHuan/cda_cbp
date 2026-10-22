"""How many units a step replaces, and where their new weights come from.

arXiv 2108.06325v3 Algorithms 1 and 2 replace "n_l * rho of eligible features"
each step and say nothing about the fraction. `count_rule: "arxiv"`, the
default for a new run, carries the fraction over and replaces every whole unit
it adds up to. `"nature"` is the Nature paper's Algorithm 1 (`if c > 1`, at most
one unit per layer per step), which every run used before this setting existed,
so a settings block without the key means it.

The new incoming weights are drawn from the layer's starting distribution, with
the bound computed by the authors' `get_layer_bound` (imported from `lop`).
"""

import math

import pytest
import torch
import torch.nn as nn
from lop.algos.cbp_linear import get_layer_bound

from cda_cbp import config as cbp_settings
from cda_cbp.cbp import CBPConfig, CBPLayerState, ReplaceableLayer, select_and_replace
from cda_cbp.learner import _as_config


def _setup(units=5):
    torch.manual_seed(0)
    seq = nn.Sequential(nn.Linear(6, units), nn.Tanh(), nn.Linear(units, 3))
    layer = ReplaceableLayer(name="block", incoming=seq[0], outgoing=[seq[2]], activation=seq[1])
    state = CBPLayerState.zeros(units)
    state.age = torch.full((units,), 10, dtype=torch.long)
    return layer, state


def _arxiv(**kw):
    return CBPConfig(enabled=True, maturity_threshold=0, count_rule="arxiv", **kw)


class TestTheArxivRule:
    def test_every_whole_unit_is_replaced_and_the_fraction_kept(self):
        """5 units at 0.5 adds 2.5: two units now, 0.5 carried over."""
        layer, state = _setup()
        utility = torch.tensor([3.0, 0.0, 4.0, 1.0, 2.0])
        replaced = select_and_replace(state, layer, utility, _arxiv(replacement_rate=0.5))
        assert sorted(replaced) == [1, 3]  # the two lowest utilities
        assert state.accumulator == pytest.approx(0.5)
        assert state.replacements == 2

    def test_a_small_rate_accumulates_and_fires_at_one(self):
        """`n_l * rho` of 0.5 per step: nothing, then one unit when it reaches 1."""
        layer, state = _setup()
        utility = torch.arange(5, dtype=torch.float)
        config = _arxiv(replacement_rate=0.1)
        assert select_and_replace(state, layer, utility, config) == []
        assert select_and_replace(state, layer, utility, config) == [0]
        assert state.accumulator == pytest.approx(0.0)

    def test_never_more_than_the_eligible_units(self):
        layer, state = _setup()
        state.age = torch.tensor([0, 10, 10, 0, 0])
        config = CBPConfig(
            enabled=True, maturity_threshold=5, count_rule="arxiv", replacement_rate=5.0
        )
        replaced = select_and_replace(state, layer, torch.zeros(5), config)
        assert sorted(replaced) == [1, 2]

    def test_the_nature_rule_still_caps_at_one(self):
        layer, state = _setup()
        config = CBPConfig(
            enabled=True, maturity_threshold=0, count_rule="nature", replacement_rate=0.5
        )
        assert len(select_and_replace(state, layer, torch.arange(5.0), config)) == 1
        assert state.accumulator == pytest.approx(1.5)


class TestTheSetting:
    def test_a_new_run_defaults_to_arxiv(self):
        assert cbp_settings.DEFAULTS["count_rule"] == "arxiv"

    def test_a_block_without_it_means_nature(self):
        recorded = {k: v for k, v in cbp_settings.DEFAULTS.items() if k != "count_rule"}
        lcd = {
            "extensions": {"extensions_enabled": ["cbp"], "extension_settings": {"cbp": recorded}}
        }
        assert _as_config(lcd).count_rule == "nature"

    def test_an_unknown_value_is_refused(self):
        with pytest.raises(ValueError, match="count_rule"):
            CBPConfig(enabled=True, count_rule="while")


class TestTheInitBoundIsTheAuthors:
    def test_it_is_the_layers_default_init_bound(self):
        """RLlib's MLP keeps torch's default Linear init, U(+/- 1/sqrt(fan_in)),
        which is lop's `init="default"`."""
        layer = nn.Linear(37, 4)
        assert get_layer_bound(layer, "default", 1.0) == pytest.approx(1 / math.sqrt(37))

    def test_a_replacement_draws_inside_it(self):
        layer, state = _setup()
        select_and_replace(state, layer, torch.arange(5.0), _arxiv(replacement_rate=0.2))
        bound = get_layer_bound(layer.incoming, "default", 1.0)
        assert torch.all(layer.incoming.weight[0].abs() <= bound)

    def test_and_cbp_imports_it_rather_than_restating_it(self):
        import inspect

        from cda_cbp import cbp

        assert cbp.get_layer_bound is get_layer_bound
        assert "math.sqrt(layer.incoming.in_features)" not in inspect.getsource(cbp)
