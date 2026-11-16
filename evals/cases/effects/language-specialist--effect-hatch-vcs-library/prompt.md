---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

Use the kb:language-specialist agent to write the packaging for a new Python library, acme-geo (import name acme_geo, src layout, Python 3.10 and newer). Save ./pyproject.toml and ./notes.md. Requirements: build with hatchling; the version comes from git tags and is never written by hand; downstream type checkers must see the package's type hints once it is installed; the dependencies are pydantic 2 or newer and httpx, and httpx has no known incompatibility with any release, so do not restrict it from future releases; the metadata declares requires-python for 3.10 and newer. In notes.md, say what the CI job that builds the release must do so the tag-derived version is correct, and name the one extra file the package needs for the type hints requirement.
