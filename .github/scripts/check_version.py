"""Verify the version is declared consistently before anything is published.

The version lives in two places that must be bumped together (see AGENTS.md):
`project.version` in pyproject.toml and `MeteoRaster.VERSION` in
meteoraster/meteoraster.py. On a tag push the tag has to agree with both.

PyPI filenames are immutable - a version can never be re-uploaded - so a
mismatch has to fail the build rather than reach the index.

The module is parsed with a regex instead of imported: the build job has no
runtime dependencies installed.
"""

from __future__ import annotations

import os
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

declared = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]

source = (ROOT / "meteoraster" / "meteoraster.py").read_text(encoding="utf-8")
match = re.search(r"^\s*VERSION\s*=\s*['\"]([^'\"]+)['\"]", source, re.MULTILINE)
if match is None:
    sys.exit("FAIL: could not find MeteoRaster.VERSION in meteoraster/meteoraster.py")
in_code = match.group(1)

problems = []
if declared != in_code:
    problems.append(
        f"pyproject.toml declares {declared!r} but MeteoRaster.VERSION is {in_code!r}"
    )

ref = os.environ.get("GITHUB_REF", "")
if ref.startswith("refs/tags/"):
    tag = ref[len("refs/tags/") :]
    if tag.lstrip("v") != declared:
        problems.append(f"tag {tag!r} does not match version {declared!r}")

if problems:
    sys.exit("FAIL: version mismatch\n" + "\n".join(f"  - {p}" for p in problems))

print(f"OK: version {declared} is consistent.")
