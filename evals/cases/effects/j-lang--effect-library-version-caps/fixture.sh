set -euo pipefail
mkdir -p docs src/mdreport
cat > pyproject.toml <<'TOML'
[project]
name = "mdreport"
version = "0.9.0"
description = "Render Markdown reports from HTTP data sources"
requires-python = ">=3.10"
dependencies = [
    "httpx>=0.25,<1.0",
    "markdown-it-py>=3.0,<4",
    "rich>=13.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
TOML
cat > src/mdreport/__init__.py <<'PY'
"""Render Markdown reports from HTTP data sources."""
PY
cat > docs/compat.md <<'MD'
# Compatibility notes

- httpx: CI tests against 0.25 through 0.28. No breakage is known.
- markdown-it-py: 4.0 changed the tokens `MarkdownIt.parse` returns, and our renderer fails on it
  (issue #41). Stay below 4 until the renderer is ported.
- rich: no issues on any 13.x or 14.x release.
MD
