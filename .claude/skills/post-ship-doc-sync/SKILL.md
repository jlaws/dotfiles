---
name: post-ship-doc-sync
description: Reusable workflow to discover and fix stale documentation after shipping changes. Use when docs may be out of date after a release, merge, or significant feature work. Do NOT use for writing new documentation from scratch (use documentation-writer agent).
compatibility: claude-code
allowed-tools: Read, Grep, Glob, Bash, Edit, Write
---

# Post-Ship Documentation Sync

## Step 1 — Identify Doc Files

Search for documentation files in the project:
- README*, ARCHITECTURE*, CONTRIBUTING*, CLAUDE.md, CHANGELOG*
- `docs/` directory (all .md files)
- Any *.md files in project root

## Step 2 — Diff Scope

Determine what changed since last release:

```bash
git log --oneline <range>
git diff --name-only <range>
```

If a range is provided, use it. Otherwise default to last tag..HEAD (`git describe --tags --abbrev=0`..HEAD).

## Step 3 — Staleness Detection

| Doc Section Type | Stale When | Detection Method |
|------------------|-----------|-----------------|
| API reference | Endpoints added/removed/changed | Route definitions in diff |
| Setup/install | Dependencies changed | package.json/pyproject.toml/Cargo.toml in diff |
| Architecture | New modules/services | New directories or major file additions |
| Config reference | Env vars/settings changed | Config keys in diff |
| CLI usage | Commands/flags changed | argparse/commander/clap definitions in diff |
| Feature docs | Behavior changed | Business logic files in diff |

For each doc file, cross-reference its content against the changed files to identify stale sections.

## Step 4 — Generate Minimal Edits

- Preserve existing voice and style
- No wholesale rewrites — surgical updates only
- Update facts, not prose style
- Add new sections only for genuinely new features
- Remove references to deleted features

## Step 5 — Present for Approval

- Show diff preview for each doc file before applying
- Wait for explicit approval before committing
- Never auto-commit documentation changes

## Red Flags — STOP

- Rewriting entire documentation files
- Speculative documentation (documenting unshipped features)
- Updating counts/stats without verification
- Changing doc structure/organization (that's a separate task)
