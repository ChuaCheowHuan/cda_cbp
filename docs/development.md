# Development

## Layout

```
cda_cbp/__init__.py   EXTENSION: what gym-CDA discovers through the entry point
cda_cbp/config.py     DEFAULTS and validate (the `cbp` settings block)
cda_cbp/cbp.py        the algorithm; imports no RLlib, so the formulas test without an Algorithm
cda_cbp/learner.py    CBPLearnerMixin: hooks, apply_gradients, metrics, get/set_state
tests/                pytest; `python -m pytest -q`
.github/workflows/cda_cbp.yml  CI
```

## Install for development

gym-CDA is not on PyPI. Install it first (CPU torch is enough), then this package:

```bash
pip install torch==2.13.0 --extra-index-url https://download.pytorch.org/whl/cpu
pip install "gym_continuousDoubleAuction @ git+https://github.com/ChuaCheowHuan/gym-continuousDoubleAuction"
pip install -e ".[test]"
python -m pytest -q
```

## Tests

| File | Covers |
|---|---|
| `test_cbp.py` | The algorithm with no Ray and no `Algorithm`: discovery (matching the layers recorded on gym-CDA at f7c1353), utility, replacement, accumulator, determinism, state, metrics, device placement, DDP unwrapping, mean compensation, the learner-count rule |
| `test_cbp_recurrent.py` | CBP on an LSTM core under `scope: "recurrent"` |
| `test_wiring.py` | A real gym-CDA `Algorithm`: composition over the encoder's Learner and outermost, real updates, metrics-only, checkpoint state, a real save and restore, and the resume guards |
| `test_equivalence.py` | The gate for removing CBP from gym-CDA (see [migration.md](migration.md)). Skipped against a gym-CDA without built-in CBP |
| `test_extension.py` | The package as gym-CDA sees it: discovery, defaults, validation, legacy settings |
| `test_docs.py` | The settings table matches `DEFAULTS`, and every relative link resolves |

## The gym-CDA contract

This package relies on gym-CDA for:

- `train/extensions.py`: `Extension` (with `learner_order`), `settings_for`, the fingerprinting
  of `ext.cbp.<key>` on a resume, and refusing a restore whose extension is missing.
- `train/model/layout.py`: `trunk_outputs(module)`, where each trunk ends and which heads read it.
  gym-CDA's `test_model_layout.py` pins it, so a rename there fails gym-CDA's CI rather than this
  package silently.
- `train/model/cbp_learner.py` (after CBP left gym-CDA): a shim that resolves
  `CBPLearnerMixin` to this package's, so checkpoints written while CBP was part of gym-CDA
  still load.

The gym-CDA ref CI tests against is `GYM_CDA_REF` in `.github/workflows/cda_cbp.yml`.
