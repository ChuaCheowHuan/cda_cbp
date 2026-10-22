"""Continual Backprop for gym-continuousDoubleAuction, as an installable extension.

Installing this package registers it with gym-CDA through the
`gym_cda.extensions` entry point; a run uses it only when `"cbp"` is in
`extensions_enabled`. See gym-CDA's docs/30_extensions.md for the contract,
`config.py` for the settings and docs/continual_backprop.md for the method.
"""

from gym_continuousDoubleAuction.train.extensions import Extension

from cda_cbp import config
from cda_cbp.learner import CBPLearnerMixin

__version__ = "0.2.0"

#: Wraps every other extension's Learner mixin, as CBP did inside gym-CDA:
#: it observes and replaces units after the whole optimiser step.
LEARNER_ORDER = 100

EXTENSION = Extension(
    name=config.NAME,
    version=__version__,
    defaults=config.DEFAULTS,
    validate=config.validate,
    learner_mixin=CBPLearnerMixin,
    learner_order=LEARNER_ORDER,
)
