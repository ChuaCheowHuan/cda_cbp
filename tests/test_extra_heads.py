"""Heads another extension adds to a trunk are consumers too (#3).

cda-discorl's `aux_pi` and `disco_heads` read the actor embedding, the output
of the trunk `pi` reads. A replacement that zeroed the new unit's outgoing
weights in `pi` but not in them would leave those heads reading the new unit
with the old unit's weights. gym-CDA reports such heads, declared by the module
class in `extra_trunk_heads`, as `TrunkOutput.extra_heads`.
"""

import pytest
import torch
import torch.nn as nn

from gym_continuousDoubleAuction.train.model.encoders import MLP_ENCODER_TYPE
from gym_continuousDoubleAuction.train.model.model_handler import build_trainable_module_spec
from gym_continuousDoubleAuction.train.model.moe_learner import CDAPPOTorchRLModule

from cda_cbp.cbp import CBPConfig, CBPLayerState, find_replaceable_layers, select_and_replace

DECLARED = {
    "encoder": ("aux_pi", "disco_heads", "aux_mlp"),
    "actor_encoder": ("aux_pi", "disco_heads", "aux_mlp"),
}


class _ExtraHeads:
    extra_trunk_heads = DECLARED

    def setup(self):
        super().setup()
        width = next(m for m in self.pi.modules() if isinstance(m, nn.Linear)).in_features
        self.aux_pi = nn.Linear(width, 7)
        # Parallel heads, as cda-discorl's disco_heads: every Linear reads the trunk.
        self.disco_heads = nn.ModuleDict(
            {"category": nn.ModuleDict({"y": nn.Linear(width, 3), "q": nn.Linear(width, 5)})}
        )
        # A two-layer head: only its first Linear reads the trunk, though the
        # second has the same input width.
        self.aux_mlp = nn.Sequential(nn.Linear(width, width), nn.Tanh(), nn.Linear(width, 2))


def _build(spaces, encoder_type=MLP_ENCODER_TYPE, **kwargs):
    spec = build_trainable_module_spec(*spaces, encoder_type=encoder_type, **kwargs)
    spec.module_class = type("Extra", (_ExtraHeads, spec.module_class or CDAPPOTorchRLModule), {})
    return spec.build()


def _consumers(module, layer):
    names = {id(m): n for n, m in module.named_modules()}
    return sorted(names[id(out)] for out in layer.outgoing)


EXTRA = ["aux_mlp.0", "aux_pi", "disco_heads.category.q", "disco_heads.category.y"]


@pytest.mark.parametrize("share", [False, True], ids=["separate", "shared"])
def test_the_trunks_last_layer_feeds_every_declared_head(spaces, share):
    module = _build(spaces, vf_share_layers=share)
    actor = "encoder.encoder.net.mlp.2" if share else "encoder.actor_encoder.net.mlp.2"
    by_name = {layer.name: layer for layer in find_replaceable_layers(module)}
    heads = ["pi.net.mlp.0", "vf.net.mlp.0"] if share else ["pi.net.mlp.0"]
    assert _consumers(module, by_name[actor]) == sorted(heads + EXTRA)
    if not share:
        critic = by_name["encoder.critic_encoder.net.mlp.2"]
        assert _consumers(module, critic) == ["vf.net.mlp.0"]


def test_a_replacement_zeroes_the_unit_in_every_declared_head(spaces):
    module = _build(spaces)
    layer = next(
        l for l in find_replaceable_layers(module) if l.name == "encoder.actor_encoder.net.mlp.2"
    )
    width = layer.incoming.out_features
    state = CBPLayerState.zeros(width)
    state.age = torch.full((width,), 1000, dtype=torch.long)
    utility = torch.arange(width, dtype=torch.float)
    config = CBPConfig(
        enabled=True, replacement_rate=1.5 / width, maturity_threshold=0, count_rule="arxiv"
    )
    (replaced,) = select_and_replace(state, layer, utility, config)
    for head in (module.aux_pi, module.disco_heads["category"]["y"], module.aux_mlp[0]):
        assert torch.count_nonzero(head.weight[:, replaced]) == 0


def test_an_lstm_trunk_feeds_its_declared_heads(spaces):
    module = _build(spaces, encoder_type="lstm")
    (actor,) = [
        l
        for l in find_replaceable_layers(module, scope="recurrent")
        if l.name == "encoder.actor_encoder.lstm"
    ]
    assert _consumers(module, actor) == sorted(["pi.net.mlp.0"] + EXTRA)
