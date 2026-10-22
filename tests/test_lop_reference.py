"""This package's generate-and-test against the authors' own, step by step.

`lop.algos.gnt.GnT` is the reference implementation from the paper's authors
(imported from the `lop` submodule, never copied). Both run on the same
network, the same minibatches and the same `AdamGnT` from the same starting
state. After every step the two must agree on:

  * which units were replaced, in every layer;
  * the ages, and the utility ranking (lop averages the outgoing weights where
    the paper and this package sum them, so the utilities differ by the
    constant factor `n_out` per layer and the ranking is the same);
  * the network after replacement: zeroed outgoing columns, zeroed incoming
    bias, the mean contribution moved into the next layer's bias;
  * Adam's state, timestep included, so the replaced slices were reset alike.

The new incoming weights are random draws from different generators, so they
are copied from lop's network into this package's after each replacement;
everything else is compared, not copied.

What the reference cannot check, and why it is set up this way:

  * `utility="contribution"`. lop's mean-corrected utilities subtract a running
    mean that already includes the current step; arXiv eq. 3 uses the previous
    step's, as this package does, so those rankings can differ.
  * `count_rule="arxiv"` with lop's `accumulate=True`. lop's default
    (`accumulate=False`) draws a coin for the fraction instead, which matches
    neither paper.
  * `compensate_mean=False`. lop's `gen_new_features` looks as if it moves the
    replaced unit's mean contribution into the next layer's bias, but
    `test_features` has already zeroed that unit's running mean, so it adds
    nothing: in effect lop does not compensate, like the papers and like this
    package's default. `test_lops_bias_compensation_adds_nothing` pins that.
"""

import copy


import torch
import torch.nn as nn
from lop.algos.gnt import GnT
from lop.utils.AdamGnT import AdamGnT

from cda_cbp.cbp import (
    CBPConfig,
    CBPLayerState,
    _sequential_layers,
    activation_stats,
    select_and_replace,
    update_utility,
)

DECAY = 0.9
RATE = 0.05
MATURITY = 5
STEPS = 60
WIDTH = 16


def _net():
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Linear(6, WIDTH), nn.Tanh(), nn.Linear(WIDTH, WIDTH), nn.Tanh(), nn.Linear(WIDTH, 3)
    )


def _forward(net, x):
    """The output and each hidden layer's activations, as GnT takes them."""
    features = []
    h = x
    for module in net:
        h = module(h)
        if isinstance(module, nn.Tanh):
            features.append(h)
    return h, features


def _step(net, optimizer, x, y):
    optimizer.zero_grad()
    out, features = _forward(net, x)
    ((out - y) ** 2).mean().backward()
    optimizer.step()
    return [f.detach() for f in features]


def _run():
    """Yield, per step, what each implementation replaced and their state."""
    ours, theirs = _net(), _net()
    theirs.load_state_dict(ours.state_dict())
    our_opt = AdamGnT(ours.parameters(), lr=1e-2)
    their_opt = AdamGnT(theirs.parameters(), lr=1e-2)

    gnt = GnT(
        net=theirs,
        hidden_activation="tanh",
        opt=their_opt,
        decay_rate=DECAY,
        replacement_rate=RATE,
        init="default",
        maturity_threshold=MATURITY,
        util_type="contribution",
        accumulate=True,
    )
    config = CBPConfig(
        enabled=True,
        utility="contribution",
        utility_decay=DECAY,
        replacement_rate=RATE,
        maturity_threshold=MATURITY,
        count_rule="arxiv",
        optimizer="adam_gnt",
        compensate_mean=False,
    )
    layers = _sequential_layers(ours, "net")
    states = [CBPLayerState.zeros(WIDTH) for _ in layers]
    generator = torch.Generator().manual_seed(1)

    data = torch.Generator().manual_seed(2)
    for _ in range(STEPS):
        x = torch.randn(32, 6, generator=data)
        y = torch.randn(32, 3, generator=data)

        our_features = _step(ours, our_opt, x, y)
        their_features = _step(theirs, their_opt, x, y)

        ours_replaced, utilities = [], []
        for layer, state, h in zip(layers, states, our_features):
            stats = activation_stats(h, state, config)
            corrected = update_utility(state, layer, stats, config)
            utilities.append(corrected.clone())
            ours_replaced.append(
                sorted(select_and_replace(state, layer, corrected, config, generator, our_opt))
            )

        to_replace, counts = gnt.test_features(features=their_features)
        theirs_replaced = [sorted(int(i) for i in idx) for idx in to_replace]
        their_utilities = [u.clone() for u in gnt.bias_corrected_util]
        gnt.gen_new_features(to_replace, counts)
        gnt.update_optim_params(to_replace, counts)

        # The fresh incoming rows are random draws; take lop's so the runs stay comparable.
        with torch.no_grad():
            for layer_index, indices in enumerate(theirs_replaced):
                if indices:
                    ours[2 * layer_index].weight[indices] = theirs[2 * layer_index].weight[indices]

        # Copies, not the live objects: every step is compared as it stood.
        yield dict(
            ours=copy.deepcopy(ours.state_dict()),
            theirs=copy.deepcopy(theirs.state_dict()),
            our_opt=_optimizer_state(our_opt, ours),
            their_opt=_optimizer_state(their_opt, theirs),
            ours_replaced=ours_replaced,
            theirs_replaced=theirs_replaced,
            utilities=utilities,
            their_utilities=their_utilities,
            ages=[s.age.clone() for s in states],
            their_ages=[a.clone() for a in gnt.ages],
        )


def _optimizer_state(optimizer, net):
    return [
        {key: value.clone() for key, value in optimizer.state[p].items()} for p in net.parameters()
    ]


STEPS_RUN = list(_run())


def test_units_are_replaced_at_all():
    """Without replacements the comparison below would be vacuous."""
    total = sum(len(i) for step in STEPS_RUN for i in step["theirs_replaced"])
    assert total >= 5


def test_the_same_units_are_replaced_each_step():
    for t, step in enumerate(STEPS_RUN):
        assert step["ours_replaced"] == step["theirs_replaced"], t


def test_the_ages_agree():
    for t, step in enumerate(STEPS_RUN):
        for ours, theirs in zip(step["ages"], step["their_ages"]):
            assert torch.equal(ours.float(), theirs), t


def test_the_utilities_agree_up_to_lops_mean_over_outgoing_weights():
    for t, step in enumerate(STEPS_RUN):
        for n_out, ours, theirs in zip((WIDTH, 3), step["utilities"], step["their_utilities"]):
            assert torch.allclose(ours, theirs * n_out, rtol=1e-4, atol=1e-7), t


def test_the_networks_agree_after_replacement():
    for t, step in enumerate(STEPS_RUN):
        for name, ours in step["ours"].items():
            assert torch.allclose(ours, step["theirs"][name], atol=1e-6), (t, name)


def test_adams_state_agrees_timestep_included():
    for t, step in enumerate(STEPS_RUN):
        for a, b in zip(step["our_opt"], step["their_opt"]):
            assert torch.equal(a["step"], b["step"]), t
            for key in ("exp_avg", "exp_avg_sq"):
                assert torch.allclose(a[key], b[key], atol=1e-6), (t, key)


def test_lops_bias_compensation_adds_nothing():
    """GnT zeroes a picked unit's running mean before it would move it into the bias."""
    net = _net()
    opt = AdamGnT(net.parameters(), lr=1e-2)
    gnt = GnT(
        net=net,
        hidden_activation="tanh",
        opt=opt,
        decay_rate=DECAY,
        replacement_rate=0.5,
        init="default",
        maturity_threshold=0,
        util_type="contribution",
        accumulate=True,
    )
    x = torch.randn(32, 6, generator=torch.Generator().manual_seed(3))
    for _ in range(3):
        _, features = _forward(net, x)
        to_replace, counts = gnt.test_features(features=[f.detach() for f in features])
    assert counts[0] > 0
    assert torch.all(gnt.mean_feature_act[0][to_replace[0]] == 0)
    # Replace in the first layer only, so the second layer's bias is only a consumer.
    to_replace[1], counts[1] = torch.empty(0, dtype=torch.long), 0
    bias = net[2].bias.detach().clone()
    gnt.gen_new_features(to_replace, counts)
    assert torch.equal(net[2].bias.detach(), bias)
