---
name: j-doc-sync
description: "Post-ship documentation sync — discover and fix stale docs after shipping changes. Use when docs may be out of date after a release or feature merge. Do NOT use for writing new docs from scratch (use /j-docs instead)."
argument-hint: "<branch, tag range, or 'since last release'>"
model: sonnet
effort: medium
---

Invoke the `post-ship-doc-sync` skill via the Skill tool before doing anything else. Sync documentation with recent code changes.

Resolve the scope in this order:
1. If arguments are provided, use them: $ARGUMENTS
2. Else, check for a current PR: `gh pr view --json number,url,baseRefName,headRefName`. If one exists, review it (`gh pr diff`) and treat its full change set as the scope (`origin/<baseRefName>...HEAD`) — update `docs/` and every README.md to reflect all changes made in the PR.
3. Else, default to `$(git describe --tags --abbrev=0)..HEAD`.

For substantial rewrites, you may delegate to the `documentation-writer` agent via the Task tool (loads `references/documentation/`). Verify its output before committing.
