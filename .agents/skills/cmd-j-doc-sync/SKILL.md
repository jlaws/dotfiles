---
name: cmd-j-doc-sync
description: "Use when invoking the j-doc-sync workflow."
disable-model-invocation: true
---

# Documentation Sync

Invoke the `post-ship-doc-sync` skill before doing anything else. Sync documentation with recent code changes.

Resolve the scope in this order:
1. If the user provided input, use it as the scope.
2. Else, check for a current PR: `gh pr view --json number,url,baseRefName,headRefName`. If one exists, review it (`gh pr diff`) and treat its full change set as the scope (`origin/<baseRefName>...HEAD`) — update `docs/` and every README.md to reflect all changes made in the PR.
3. Else, default to `$(git describe --tags --abbrev=0)..HEAD`.

For substantial rewrites, you may delegate to the `documentation-writer` agent (loads `.agents/references/documentation/`). Verify its output before committing.
