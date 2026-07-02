---
name: j-doc-sync
description: "Post-ship documentation sync — discover and fix stale docs after shipping changes. Use when docs may be out of date after a release or feature merge. Do NOT use for writing new docs from scratch (use /j-docs instead)."
argument-hint: "<branch, tag range, or 'since last release'>"
---

Invoke the `post-ship-doc-sync` skill via the Skill tool before doing anything else. Sync documentation with recent code changes.

For substantial rewrites, you may delegate to the `documentation-writer` agent via the Task tool (loads `references/documentation/`). Verify its output before committing.

Scope: $ARGUMENTS

If no argument provided, default to `$(git describe --tags --abbrev=0)..HEAD`.
