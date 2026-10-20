"""The docs say what the code does: the settings table, and links that resolve."""

import re
from pathlib import Path

import pytest

from cda_cbp.config import DEFAULTS

ROOT = Path(__file__).resolve().parent.parent
DOCS = sorted([ROOT / "README.md", ROOT / "CHANGELOG.md", *(ROOT / "docs").glob("*.md")])


def test_the_settings_table_matches_the_defaults():
    text = (ROOT / "docs" / "settings.md").read_text()
    rows = dict(re.findall(r"^\| `([a-z_]+)` \| `([^`]+)` \|", text, flags=re.M))
    assert rows == {k: __import__("json").dumps(v) for k, v in DEFAULTS.items()}


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: str(p.relative_to(ROOT)))
def test_every_relative_link_resolves(doc):
    text = re.sub(r"```.*?```", "", doc.read_text(), flags=re.S)
    broken = []
    for target in re.findall(r"\]\(([^)\s]+)\)", text):
        if re.match(r"[a-z]+:", target) or target.startswith("#"):
            continue
        path = (doc.parent / target.split("#")[0]).resolve()
        if not path.exists():
            broken.append(target)
    assert broken == []
