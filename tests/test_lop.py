"""The loss-of-plasticity code this package imports, shipped unchanged.

`lop` is the authors' repository (ChuaCheowHuan/loss-of-plasticity, a fork of
shibhansh/loss-of-plasticity), a git submodule at `third_party/` that the
wheel installs as the top-level package `lop`. Its own `setup.py` packages
nothing (there is no `lop/__init__.py`), so a pip dependency on it would
install no code; see docs/development.md.
"""

import hashlib
import importlib
import pathlib
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SUBMODULE = ROOT / "third_party" / "loss-of-plasticity"

#: The modules this package and its tests import from lop.
USED = {
    "lop.utils.AdamGnT": "AdamGnT",
    "lop.algos.gnt": "GnT",
    "lop.algos.cbp_linear": "get_layer_bound",
}


@pytest.mark.parametrize("module, name", sorted(USED.items()))
def test_it_is_importable(module, name):
    assert hasattr(importlib.import_module(module), name)


@pytest.mark.parametrize("module", sorted(USED))
def test_the_installed_file_is_the_submodules_unchanged(module):
    """No local copy and no local edit: the bytes are the pinned commit's."""
    if not (SUBMODULE / ".git").exists():
        pytest.skip("no submodule checkout (an installed wheel)")
    installed = pathlib.Path(importlib.import_module(module).__file__)
    relative = pathlib.Path(*module.split(".")).with_suffix(".py")
    committed = subprocess.run(
        ["git", "-C", str(SUBMODULE), "show", f"HEAD:{relative.as_posix()}"],
        check=True,
        capture_output=True,
    ).stdout
    assert hashlib.sha256(installed.read_bytes()).digest() == hashlib.sha256(committed).digest()


def test_the_experiment_code_is_not_installed():
    """`lop.envs` needs gym 0.23 and mujoco; nothing here uses it."""
    with pytest.raises(ImportError):
        importlib.import_module("lop.envs")
