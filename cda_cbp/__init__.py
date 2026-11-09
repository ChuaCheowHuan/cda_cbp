"""Continual Backprop for gym-continuousDoubleAuction, as an installable extension.

Installing this package registers it with gym-CDA through the
`gym_cda.extensions` entry point; a run uses it only when `"cbp"` is in
`extensions_enabled`. See gym-CDA's docs/30_extensions.md for the contract,
`config.py` for the settings and docs/continual_backprop.md for the method.
"""

from importlib.metadata import PackageNotFoundError, version

from gym_continuousDoubleAuction.train.extensions import Extension

from cda_cbp import config
from cda_cbp.learner import CBPLearnerMixin

try:
    #: The installed distribution's version, so pyproject.toml is the one source.
    #: gym-CDA records it beside every checkpoint that enables `cbp`; a hardcoded
    #: copy here would drift from the release it names.
    __version__ = version("cda-cbp")
except PackageNotFoundError:  # a source checkout that was never installed
    __version__ = "0+unknown"

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
