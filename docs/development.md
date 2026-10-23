# Development

## Layout

```
cda_cbp/__init__.py   EXTENSION: what gym-CDA discovers through the entry point
cda_cbp/config.py     DEFAULTS and validate (the `cbp` settings block)
cda_cbp/cbp.py        the algorithm; imports no RLlib, so the formulas test without an Algorithm
cda_cbp/learner.py    CBPLearnerMixin: hooks, apply_gradients, metrics, get/set_state
tests/                pytest; `python -m pytest -q`
third_party/loss-of-plasticity  the authors' code, a git submodule, installed unchanged as `lop`
.github/workflows/cda_cbp.yml  CI
```

## Install for development

gym-CDA is not on PyPI. Install it first (CPU torch is enough), then this package:

```bash
pip install torch==2.13.0 --extra-index-url https://download.pytorch.org/whl/cpu
pip install "gym_continuousDoubleAuction @ git+https://github.com/ChuaCheowHuan/gym-continuousDoubleAuction"
git clone --recurse-submodules https://github.com/ChuaCheowHuan/cda_cbp
cd cda_cbp
pip install -e ".[test]"
python -m pytest -q
```

A clone made without `--recurse-submodules` needs `git submodule update --init` before the
install, or the build stops at the missing `third_party/loss-of-plasticity/lop` directory.

## The loss-of-plasticity code (`lop`)

This package imports from the loss-of-plasticity authors' code
([ChuaCheowHuan/loss-of-plasticity](https://github.com/ChuaCheowHuan/loss-of-plasticity), a fork
of shibhansh/loss-of-plasticity) rather than keeping copies of it. That repository's `setup.py`
packages nothing (it has no `lop/__init__.py`, so `find_packages()` finds no package), so a pip
dependency on it would install no code. It is therefore a git submodule pinned to one commit, and
`pyproject.toml` maps the top-level package `lop` onto it: `lop`, `lop.utils` and `lop.algos` are
installed byte for byte, and `lop.envs` and the experiment scripts (gym 0.23, mujoco) are not.
`pip install git+https://github.com/ChuaCheowHuan/cda_cbp` fetches the submodule itself.

To move to a newer commit of it: `git -C third_party/loss-of-plasticity fetch && git -C
third_party/loss-of-plasticity checkout <sha>`, commit the submodule, and let CI run.
`tests/test_lop.py` checks that the installed files are the pinned commit's, unchanged.

## Tests

| File | Covers |
|---|---|
| `test_cbp.py` | The algorithm with no Ray and no `Algorithm`: discovery (matching the layers recorded on gym-CDA at f7c1353), utility, replacement, accumulator, determinism, state, metrics, device placement, DDP unwrapping, mean compensation, the learner-count rule |
| `test_cbp_recurrent.py` | CBP on an LSTM core under `scope: "recurrent"` |
| `test_wiring.py` | A real gym-CDA `Algorithm`: composition over the encoder's Learner and outermost, real updates, metrics-only, checkpoint state, a real save and restore, and the resume guards |
| `test_lop_reference.py` | Step by step against the authors' `lop.algos.gnt.GnT` on the same network, data and `AdamGnT`: the same units replaced, ages, utility ranking, network and Adam state (timestep included) |
| `test_extension.py` | The package as gym-CDA sees it: discovery, defaults, validation, legacy settings |
| `test_lop.py` | The `lop` modules used import, are the pinned submodule's files unchanged, and `lop.envs` is not installed |
| `test_count_rule.py` | The `arxiv` counter grows by `n_l · ρ` and `lop`'s by the eligible count, both replace every whole unit it holds, `nature` at most one, a missing `count_rule` means `nature`, and the init bound is `lop`'s `get_layer_bound` |
| `test_extra_heads.py` | Heads another extension declares in `extra_trunk_heads` are consumers of the trunk's last layer, on MLP and LSTM trunks, and a replacement zeroes the unit in each |
| `test_adam_gnt.py` | `AdamGnT`'s per-weight timestep is reset with the moments for exactly the replaced slices, torch Adam is left alone, a missing `optimizer` means `torch_adam`, and a real `Algorithm` builds the optimiser asked for |
| `test_moved_checks.py` | Checks that lived in gym-CDA's suite until CBP moved: the rank metric calls gym-CDA's `effective_rank` (NaN below two rows) and keeps no copy, and a replaced unit's bias is zeroed |
| `test_docs.py` | The settings table matches `DEFAULTS`, and every relative link resolves |

The `jepa` cases need a `jepa` encoder: gym-CDA's own (2.2 and earlier) or cda-jepa's, which
`.[test]` installs. Without one they skip (`requires_jepa` in `tests/conftest.py`).

## The gym-CDA contract

This package relies on gym-CDA for:

- `train/extensions.py`: `Extension` (with `learner_order`), `settings_for`, the fingerprinting
  of `ext.cbp.<key>` on a resume, and refusing a restore whose extension is missing.
- `train/model/layout.py`: `trunk_outputs(module)`, where each trunk ends and which heads read it,
  `extra_heads` (gym-CDA 2.1) included.
  gym-CDA's `test_model_layout.py` pins it, so a rename there fails gym-CDA's CI rather than this
  package silently.
- `train/model/cbp_learner.py` (after CBP left gym-CDA): a shim that resolves
  `CBPLearnerMixin` to this package's, so checkpoints written while CBP was part of gym-CDA
  still load.

The gym-CDA ref CI tests against is `GYM_CDA_REF` in `.github/workflows/cda_cbp.yml`.
