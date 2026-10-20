"""Shared helpers: the env's spaces, a built trainable module, and CBP enabled as an extension."""

import pytest

from gym_continuousDoubleAuction.envs.continuousDoubleAuction_env import (
    continuousDoubleAuctionEnv,
)
from gym_continuousDoubleAuction.train import extensions
from gym_continuousDoubleAuction.train.model.encoders import MLP_ENCODER_TYPE
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
