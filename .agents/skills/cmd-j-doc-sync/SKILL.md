---
name: cmd-j-doc-sync
description: "Post-ship documentation sync — discover and fix stale docs after shipping changes. Use when docs may be out of date after a release or feature merge. Do NOT use for writing new docs from scratch (use /j-docs instead)."
disable-model-invocation: true
---

# Documentation Sync

Invoke the `post-ship-doc-sync` skill before doing anything else. Sync documentation with recent code changes.

For substantial rewrites, you may delegate to the `documentation-writer` agent (loads `.agents/references/documentation/`). Verify its output before committing.

Scope: the user's provided input

If no argument provided, default to `$(git describe --tags --abbrev=0)..HEAD`.
