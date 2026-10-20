"""The gate for removing CBP from gym-CDA: this package must train identically.

Runs only against a gym-CDA that still has its own Continual Backprop
(`cbp_enabled` in `TrainConfig`), so that both paths exist in one process.

A whole `algo.train()` is not reproducible run to run here (two identical
built-in runs already differ), so the comparison is made where CBP acts: the
Learner update. Two algorithms are built, one with gym-CDA's built-in CBP and
one with this extension. The second learner is given the first's exact state
(weights, optimiser, CBP's per-unit state), and then both are updated on the
same sampled episodes, with every RNG reseeded before each update, for several
rounds. The replacement rate is forced high so units are actually replaced.
Weights, optimiser state, CBP state and every logged `cbp_*` metric must match
exactly. A control with replacement switched off must differ, which shows the
comparison can fail.

Against a gym-CDA without built-in CBP (after the removal) the module is
skipped: there is nothing left to compare with.
"""

import copy
import random

import numpy as np
import pytest
import ray
import torch
from conftest import TEST_CFG, cbp

from gym_continuousDoubleAuction.train.config import TrainConfig
from gym_continuousDoubleAuction.train.train import build_config

from cda_cbp.learner import CBPLearnerMixin

pytestmark = pytest.mark.skipif(
    "cbp_enabled" not in TrainConfig.__dataclass_fields__,
    reason="this gym-CDA has no built-in Continual Backprop to compare against",
)

#: Forced high so that units are replaced within two iterations.
FIRING = dict(replacement_rate=0.05, maturity_threshold=1, metrics_every_n_updates=1)

#: (case id, extension settings, encoder_type). The built-in config is derived
#: from the same settings by adding the `cbp_` prefix.
CASES = [
    ("mlp", FIRING, "mlp"),
    ("jepa", FIRING, "jepa"),
    ("lstm-recurrent", dict(FIRING, scope="recurrent"), "lstm"),
    ("metrics-only", dict(FIRING, metrics_only=True), "mlp"),
    ("iteration-compensated", dict(FIRING, fire_on="iteration", compensate_mean=True), "mlp"),
]

ROUNDS = 3
MODULES = ("policy_0", "policy_1")


def _builtin(settings):
    """gym-CDA's own keys for the same settings."""
    keys = {f"cbp_{k}": v for k, v in settings.items()}
    keys.setdefault("cbp_enabled", True)
    return keys


def _build(overrides):
    ppo_config, _cb = build_config(TrainConfig(**TEST_CFG, **overrides))
    return ppo_config.build_algo()


def _reseed(algo):
    """Reset every RNG an update draws from, the same way for both algorithms.

    The global ones, and each `jepa` encoder's own mask generator: those are
    seeded per encoder from a process-wide counter, on purpose, so two learners
    in one process draw different masks unless they are reset here.
    """
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    learner = algo.learner_group._learner
    for m in MODULES:
        for i, (_name, sub) in enumerate(learner.module[m].named_modules()):
            if hasattr(sub, "_mask_rng"):
                sub._mask_generator = torch.Generator().manual_seed(1000 + i)


def _update(algo, episodes):
    _reseed(algo)
    results = algo.learner_group.update(
        episodes=copy.deepcopy(episodes),
        timesteps={},
        num_epochs=algo.config.num_epochs,
        minibatch_size=algo.config.minibatch_size,
        shuffle_batch_per_epoch=algo.config.shuffle_batch_per_epoch,
    )
    result = results[0] if isinstance(results, list) else results
    return {
        m: {k: float(v.peek()) for k, v in result[m].items() if k.startswith("cbp_")}
        for m in MODULES
    }


def _snapshot(learner):
    weights = {
        m: {k: v.detach().clone() for k, v in learner.module[m].state_dict().items()}
        for m in MODULES
    }
    optim = {
        m: [
            {k: v.clone() for k, v in slots.items() if isinstance(v, torch.Tensor)}
            for slots in learner.get_optimizer(module_id=m).state.values()
        ]
        for m in MODULES
    }
    state = {
        f"{m}/{name}": layer.get_state()
        for m, layers in learner._cbp_state.items()
        for name, layer in layers.items()
    }
    replacements = sum(
        layer.replacements for layers in learner._cbp_state.values() for layer in layers.values()
    )
    return weights, optim, state, replacements


def _run_pair(builtin_overrides, extension_overrides):
    """Update both learners side by side; return their snapshots and metrics."""
    reference, candidate = _build(builtin_overrides), _build(extension_overrides)
    try:
        ref, cand = reference.learner_group._learner, candidate.learner_group._learner
        cand.set_state(ref.get_state())
        ref_metrics, cand_metrics = [], []
        for _ in range(ROUNDS):
            episodes = reference.env_runner_group.local_env_runner.sample(
                num_episodes=TEST_CFG["num_episodes_per_iter"]
            )
            ref_metrics.append(_update(reference, episodes))
            cand_metrics.append(_update(candidate, episodes))
        return (_snapshot(ref), ref_metrics), (_snapshot(cand), cand_metrics), cand
    finally:
        reference.stop()
        candidate.stop()


def _assert_same(a, b, path="root"):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b), path
    elif isinstance(a, dict):
        assert set(a) == set(b), path
        for key in a:
            _assert_same(a[key], b[key], f"{path}.{key}")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            _assert_same(x, y, f"{path}[{i}]")
    elif isinstance(a, float) and a != a:
        assert b != b, path  # both NaN
    else:
        assert a == b, path


@pytest.fixture(scope="module", autouse=True)
def _ray():
    ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False, num_cpus=2)
    yield
    ray.shutdown()


@pytest.mark.parametrize(
    "settings, encoder_type", [c[1:] for c in CASES], ids=[c[0] for c in CASES]
)
def test_the_extension_trains_exactly_like_the_builtin(settings, encoder_type):
    (ref, ref_metrics), (cand, cand_metrics), learner = _run_pair(
        dict(_builtin(settings), encoder_type=encoder_type),
        dict(cbp(**settings), encoder_type=encoder_type),
    )
    assert isinstance(learner, CBPLearnerMixin)
    for name, a, b in zip(("weights", "optimiser", "cbp_state", "replacements"), ref, cand):
        _assert_same(a, b, name)
    _assert_same(ref_metrics, cand_metrics, "metrics")
    assert ref_metrics[-1]["policy_0"], "no cbp_ metric was logged"
    if not settings.get("metrics_only"):
        assert ref[3] > 0, "nothing was replaced, so the comparison proves little"


def test_the_comparison_can_fail():
    """A control: the extension with replacement off must not match the built-in with it on."""
    (ref, _), (cand, _), _learner = _run_pair(
        dict(_builtin(FIRING)), cbp(**dict(FIRING, metrics_only=True))
    )
    assert ref[3] > 0 and cand[3] == 0
    with pytest.raises(AssertionError):
        _assert_same(ref[0], cand[0], "weights")
