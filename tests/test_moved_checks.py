"""Checks that lived in gym-CDA's own suite until CBP moved here.

Each one pins CBP against something in gym-CDA, so it has to run where both
are importable: here, with gym-CDA installed as a dependency.
"""

import inspect
import math

import numpy as np
import torch

from cda_cbp import cbp
from gym_continuousDoubleAuction.train.probe import rank as rank_module


class TestEffectiveRankIsTheProbes:
    """One definition, gym-CDA's (gym-CDA #251, docs/24), called rather than copied.

    The metric used to keep a torch copy of `probe.rank.effective_rank`, and
    only a test kept the two the same measurement. Now it calls the probe's.
    """

    def test_they_agree(self):
        rng = np.random.default_rng(7)
        for matrix in (
            rng.standard_normal((256, 32)),
            rng.standard_normal((256, 4)) @ rng.standard_normal((4, 32)),
            rng.standard_normal((64, 64)),
            np.zeros((32, 8)),
        ):
            assert rank_module.effective_rank(matrix) == cbp.effective_rank(torch.tensor(matrix))

    def test_no_copy_of_the_decomposition_is_kept(self):
        source = inspect.getsource(cbp.effective_rank)
        assert "svd" not in source and "cumsum" not in source

    def test_the_one_deliberate_divergence_is_the_degenerate_row_count(self):
        """Fewer than two rows: 0 in the probe's report column, NaN in a metric
        series, where 0.0 would read as total collapse."""
        for matrix in (np.empty((0, 8)), np.ones((1, 8))):
            assert rank_module.effective_rank(matrix) == 0
            assert math.isnan(cbp.effective_rank(torch.tensor(matrix)))

    def test_a_tensor_that_requires_grad(self):
        activations = torch.randn(16, 4, requires_grad=True)
        assert cbp.effective_rank(activations) == rank_module.effective_rank(
            activations.detach().numpy()
        )


class TestAReplacedUnitsBiasIsZeroed:
    """gym-CDA #251: the comment said the bias was redrawn; the code zeroes it."""

    def test_the_comment_says_zeroed(self):
        import inspect

        source = inspect.getsource(cbp._reinitialise)
        assert "for weight and bias" not in source
        assert "bias is NOT drawn" in source

    def test_and_the_code_does_zero_it(self):
        incoming, outgoing = torch.nn.Linear(4, 3), torch.nn.Linear(3, 2)
        with torch.no_grad():
            incoming.bias.fill_(7.0)
        layer = cbp.ReplaceableLayer(
            name="x", incoming=incoming, outgoing=[outgoing], activation=torch.nn.Tanh()
        )
        state = cbp.CBPLayerState.zeros(3)
        cbp._reinitialise(state, layer, 1, None, None, cbp.CBPConfig())
        assert float(incoming.bias[1]) == 0.0 and float(incoming.bias[0]) == 7.0
