"""List stale gym-CDA-era references and test-inventory drift in cda_cbp's docs.

    python .claude/skills/cbp-doc-sync/scripts/check_docs.py [repo_root]

Prints `kind  file:line  message` per finding and exits 1 if there are any.
"""

import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()

#: Names that belonged to gym-CDA's built-in CBP. Current text should not use them as this
#: repository's own (a line that says "gym-CDA" or holds a URL is taken to be about gym-CDA).
STALE = [
    (r"docs/25\b", "gym-CDA's old CBP doc: docs/continual_backprop.md"),
    (r"\bcbp_learner(\.py)?\b", "gym-CDA's old file: cda_cbp/learner.py"),
    (
        r"\bcbp_(scope|utility|fire_on|replacement_rate|maturity_threshold|metrics_only)\b",
        "old gym-CDA key: extension_settings['cbp'][...]",
    ),
    (r"\btrain_config\.json\b", "gym-CDA's config: docs/settings.md"),
    (r"\bjepa_learner\b(?!\(\))", "moved to cda-jepa"),
    (r"\bdocs/1[16]\b(?! §)", "a gym-CDA doc: say so, or link it"),
]
SKIP = {"CHANGELOG.md", "docs/migration.md"}
ABOUT_GYM = re.compile(r"gym-CDA|gym_continuousDoubleAuction|https?://|shim|legacy|Legacy|old")


def current_text_files():
    files = [
        ROOT / "README.md",
        *sorted((ROOT / "docs").glob("*.md")),
        *sorted((ROOT / "cda_cbp").glob("*.py")),
    ]
    return [f for f in files if f.exists() and str(f.relative_to(ROOT)) not in SKIP]


def stale_refs():
    for path in current_text_files():
        in_fence = False
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if line.lstrip().startswith("```"):
                in_fence = not in_fence
            if in_fence or ABOUT_GYM.search(line):
                continue
            for pattern, hint in STALE:
                if re.search(pattern, line):
                    yield (
                        "stale-ref",
                        f"{path.relative_to(ROOT)}:{number}",
                        f"{hint} | {line.strip()}",
                    )


def inventory():
    dev = ROOT / "docs" / "development.md"
    rows = set(re.findall(r"^\| `(test_[a-z0-9_]+\.py)` \|", dev.read_text(), flags=re.M))
    files = {p.name for p in (ROOT / "tests").glob("test_*.py")}
    for name in sorted(files - rows):
        yield "inventory", "docs/development.md", f"no row for tests/{name}"
    for name in sorted(rows - files):
        yield "inventory", "docs/development.md", f"row for missing tests/{name}"


def changelog():
    text = (ROOT / "CHANGELOG.md").read_text()
    if "## Unreleased" in text:
        return
    version = re.search(r'^version = "([^"]+)"', (ROOT / "pyproject.toml").read_text(), re.M)
    log = subprocess.run(
        [
            "git",
            "-C",
            str(ROOT),
            "log",
            "--oneline",
            "-G",
            "^version",
            "-1",
            "--format=%h",
            "--",
            "pyproject.toml",
        ],
        capture_output=True,
        text=True,
    ).stdout.strip()
    ahead = (
        subprocess.run(
            ["git", "-C", str(ROOT), "rev-list", "--count", f"{log}..HEAD"],
            capture_output=True,
            text=True,
        ).stdout.strip()
        if log
        else "0"
    )
    if ahead not in ("", "0"):
        yield (
            "changelog",
            "CHANGELOG.md",
            f"{ahead} commits since version {version and version[1]}, no ## Unreleased",
        )


def main() -> int:
    found = [*stale_refs(), *inventory(), *changelog()]
    for kind, where, message in found:
        print(f"{kind:10} {where}  {message}")
    print(f"TOTAL {len(found)}")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
