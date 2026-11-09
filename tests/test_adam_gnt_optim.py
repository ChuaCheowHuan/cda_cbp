"""`CBPAdamGnT` is the authors' AdamGnT with its scalar allocations cached.

The optimiser maths must not move: it is compared step by step with the
original `lop.utils.AdamGnT`. The only intended change is that the two beta
scalars are built once per param group rather than once per parameter.
"""

from unittest.mock import patch

import torch
import torch.nn as nn
from lop.utils.AdamGnT import AdamGnT

from cda_cbp.optim import CBPAdamGnT


def _net():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(6, 8), nn.Tanh(), nn.Linear(8, 3))


def _step(optimizer, net, x, y):
    optimizer.zero_grad()
    ((net(x) - y) ** 2).mean().backward()
    optimizer.step()


def test_it_matches_adam_gnt_step_by_step():
    ours, theirs = _net(), _net()
    theirs.load_state_dict(ours.state_dict())
    our_opt = CBPAdamGnT(ours.parameters(), lr=1e-2, betas=(0.9, 0.99))
    their_opt = AdamGnT(theirs.parameters(), lr=1e-2, betas=(0.9, 0.99))
    data = torch.Generator().manual_seed(2)

    for _ in range(10):
        x = torch.randn(32, 6, generator=data)
        y = torch.randn(32, 3, generator=data)
        _step(our_opt, ours, x, y)
        _step(their_opt, theirs, x, y)

        for ours_p, theirs_p in zip(ours.parameters(), theirs.parameters()):
            torch.testing.assert_close(ours_p, theirs_p, rtol=1e-6, atol=1e-7)
            for key in ("step", "exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(
                    our_opt.state[ours_p][key], their_opt.state[theirs_p][key]
                )


def test_the_beta_scalars_are_built_once_per_group():
    net = _net()
    optimizer = CBPAdamGnT(net.parameters(), lr=1e-2)
    x, y = torch.randn(8, 6), torch.randn(8, 3)

    with patch("cda_cbp.optim.torch.tensor", wraps=torch.tensor) as tensor:
        for _ in range(3):
            _step(optimizer, net, x, y)

    # Two scalars for the one group, on the first step; none on later steps.
    assert tensor.call_count == 2
    assert len(optimizer._cbp_beta_tensors) == 1
    assert len(optimizer.param_groups) == 1
