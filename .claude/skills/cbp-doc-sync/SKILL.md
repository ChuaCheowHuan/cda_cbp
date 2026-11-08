---
name: cbp-doc-sync
description: Bring cda_cbp's README, docs/*.md, CHANGELOG and docstrings back in line with the code after a change to tests, settings, metrics, files or the gym-CDA contract, or when asked to "update the docs". Lists every stale reference and every missing or stale test-inventory row, then fixes them.
---

# Keep cda_cbp's docs saying what the code does

## Steps

1. **Scan:** `python .claude/skills/cbp-doc-sync/scripts/check_docs.py`. It reports
   - `stale-ref`: a gym-CDA-era name in current text (`docs/25`, `cbp_learner.py` as this repo's
     file, `cbp_scope` and other `cbp_<key>` settings, `train_config.json` as this repo's config,
     `jepa_learner` as an importable module). `docs/migration.md` and `CHANGELOG.md` describe the
     old names on purpose and are skipped, and so is any line that names gym-CDA.
   - `inventory`: a `tests/test_*.py` with no row in the `docs/development.md` test table, or a
     row for a file that no longer exists.
   - `changelog`: no `## Unreleased` section while the branch has commits after the last version.
   It exits non-zero when it finds anything.
2. **Settings and links** are already tests: `tests/test_docs.py` checks the `docs/settings.md`
   table against `DEFAULTS` and every relative link. Run `python -m pytest tests/test_docs.py -q`.
3. **Fix current-tense text.** Text that says what used to be true ("was 0.1.0", "before #2")
   stays. A gym-CDA reference is kept when it says it is gym-CDA's (`gym-CDA docs/29 R-13`) or is
   a full URL.
4. **Record it** under `## Unreleased` in `CHANGELOG.md`. Never rewrite a released section.

## Guardrails

- A new setting goes in `cda_cbp/config.py` `DEFAULTS` and the `docs/settings.md` table
  together; a new metric in `docs/metrics.md` and `_cbp_log`'s reductions together.
- Docs commit after the code and tests they describe.
