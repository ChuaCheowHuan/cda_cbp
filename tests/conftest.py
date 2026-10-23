"""Shared helpers: the env's spaces, a built trainable module, and CBP enabled as an extension."""

import pytest

from gym_continuousDoubleAuction.envs.continuousDoubleAuction_env import (
    continuousDoubleAuctionEnv,
)
from gym_continuousDoubleAuction.train import extensions
from gym_continuousDoubleAuction.train.model import encoders
from gym_continuousDoubleAuction.train.model.encoders import (
    ENCODER_REGISTRY,
    MLP_ENCODER_TYPE,
    known_encoder_type,
)
from gym_continuousDoubleAuction.train.model.model_handler import build_trainable_module_spec
from gym_continuousDoubleAuction.train.model.moe_learner import CDAPPOTorchRLModule

#: A training config small enough for a test iteration.
TEST_CFG = dict(
    num_agents=4,
    num_trained_agents=2,
    max_step=64,
    num_episodes_per_iter=2,
    num_epochs=1,
    minibatch_size=64,
    num_env_runners=0,
    num_learners=0,
    num_gpus_per_learner=0,
    episode_data_dir=None,
    log_level="ERROR",
    seed=7,
)


def cbp(**settings):
    """TrainConfig keyword arguments that enable this extension with `settings`."""
    return dict(extensions_enabled=["cbp"], extension_settings={"cbp": settings})


def jepa_available() -> bool:
    """Whether a `jepa` encoder can be built: gym-CDA's own (2.2 and earlier) or cda-jepa's.

    Looking the name up is what loads cda-jepa's encoder, so this also registers it.
    """
    try:
        known_encoder_type("jepa")
    except ValueError:
        return False
    return True


#: For the tests that use the `jepa` encoder, which left gym-CDA for cda-jepa.
requires_jepa = pytest.mark.skipif(
    not jepa_available(), reason="no jepa encoder: install cda-jepa (the test extra does)"
)


def encoder_types():
    """Every encoder to test: the registered ones, plus `jepa` when a package provides it."""
    names = set(ENCODER_REGISTRY)
    if jepa_available():
        names.add("jepa")
    return [MLP_ENCODER_TYPE] + sorted(names)


def with_jepa(**config):
    """TrainConfig keyword arguments selecting `jepa`, with the extension providing it enabled.

    gym-CDA refuses `jepa` from cda-jepa unless `"jepa"` is in `extensions_enabled`;
    its own `jepa` (2.2 and earlier) belongs to no extension.
    """
    extension_of = getattr(encoders, "encoder_extension", lambda name: None)
    extension = extension_of("jepa")
    if extension:
        config["extensions_enabled"] = [*config.get("extensions_enabled", []), extension]
    return dict(config, encoder_type="jepa")


def jepa_learner():
    """The `jepa` encoder's Learner class, from whichever package registered it."""
    from gym_continuousDoubleAuction.train.model.moe_learner import CDAPPOTorchLearner

    return encoders.learner_class_for("jepa", CDAPPOTorchLearner)


@pytest.fixture(autouse=True)
def _fresh_discovery():
    """Each test sees the installed extensions as they are now."""
    extensions.discover.cache_clear()
    yield
    extensions.discover.cache_clear()


@pytest.fixture(scope="module")
def spaces():
    env = continuousDoubleAuctionEnv({})
    agent_id = env.agents[0]
    return env.get_observation_space(agent_id), env.get_action_space(agent_id)


def build_module(spaces, encoder_type=MLP_ENCODER_TYPE, **kwargs):
    obs_space, act_space = spaces
    spec = build_trainable_module_spec(obs_space, act_space, encoder_type=encoder_type, **kwargs)
    if spec.module_class is None:
        spec.module_class = CDAPPOTorchRLModule
    return spec.build()
