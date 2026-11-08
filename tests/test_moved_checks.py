"""Checks that lived in gym-CDA's own suite until CBP moved here.

Each one pins CBP against something in gym-CDA, so it has to run where both
are importable: here, with gym-CDA installed as a dependency.
"""

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
