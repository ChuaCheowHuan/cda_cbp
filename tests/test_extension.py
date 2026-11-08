"""The package as gym-CDA sees it: discovered, defaulted, validated, composed."""

import pytest
from conftest import TEST_CFG, cbp

from gym_continuousDoubleAuction.train import extensions
from gym_continuousDoubleAuction.train.config import TrainConfig

import cda_cbp
from cda_cbp.cbp import CBPConfig
from cda_cbp.config import DEFAULTS, NAME, to_cbp_config
from cda_cbp.learner import LEGACY_CONFIG_KEY, _as_config


def test_gym_cda_discovers_the_installed_package():
    assert extensions.discover()[NAME] is cda_cbp.EXTENSION


#: Keys whose `CBPConfig` value is the legacy one on purpose: a checkpoint is
#: rebuilt from its own recorded settings, and one written before the key
#: existed must keep behaving as it did. A new run gets the block's default.
LEGACY_WHEN_MISSING = {"optimizer": "torch_adam", "count_rule": "nature"}


def test_the_defaults_are_the_algorithms_defaults():
    """The block's defaults must not drift from `CBPConfig`'s, or "default" would mean two things."""
    settings = {**DEFAULTS, **LEGACY_WHEN_MISSING}
    assert to_cbp_config(settings) == CBPConfig(enabled=True)


def test_enabled_with_no_settings_fills_every_default():
    cfg = TrainConfig(**TEST_CFG, **cbp())
    assert extensions.resolve(cfg.extensions_enabled, cfg.extension_settings)[NAME] == DEFAULTS


def test_an_unknown_setting_is_refused():
    with pytest.raises(ValueError, match="unknown keys"):
        TrainConfig(**TEST_CFG, **cbp(enabled=True))


def test_a_bad_value_is_refused_when_the_config_is_built():
    """gym-CDA calls `validate` while building the TrainConfig; the values are in test_cbp.py."""
    with pytest.raises(ValueError, match="Unknown utility"):
        TrainConfig(**TEST_CFG, **cbp(utility="nope"))


class TestSettingsAreReadOffTheLearnerConfig:
    def test_the_extensions_settings_win(self):
        lcd = {
            "extensions": {
                "extensions_enabled": [NAME],
                "extension_settings": {NAME: dict(DEFAULTS, replacement_rate=0.5)},
            },
            LEGACY_CONFIG_KEY: {"cbp_enabled": True, "cbp_replacement_rate": 0.25},
        }
        config = _as_config(lcd)
        assert config.replaces and config.replacement_rate == 0.5

    def test_a_legacy_checkpoints_group_is_read(self):
        """A run written while CBP was part of gym-CDA must resume with CBP on."""
        lcd = {LEGACY_CONFIG_KEY: {"cbp_enabled": True, "cbp_replacement_rate": 0.25}}
        config = _as_config(lcd)
        assert config.replaces and config.replacement_rate == 0.25

    def test_metrics_only_measures_without_replacing(self):
        lcd = {
            "extensions": {
                "extensions_enabled": [NAME],
                "extension_settings": {NAME: dict(DEFAULTS, metrics_only=True)},
            }
        }
        config = _as_config(lcd)
        assert config.active and not config.replaces

    def test_nothing_configured_is_off(self):
        assert not _as_config(None).active
