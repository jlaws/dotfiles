---
name: j-doc-sync
description: "Post-ship documentation sync — discover and fix stale docs after shipping changes. Use when docs may be out of date after a release or feature merge. Do NOT use for writing new docs from scratch (use /j-docs instead)."
argument-hint: "<branch, tag range, or 'since last release'>"
---

Sync documentation with recent code changes: $ARGUMENTS

---

## Step 1 — Determine Scope

Parse the argument for a git range. Defaults:
- No argument: last tag..HEAD (use `git describe --tags --abbrev=0`)
- Branch name: main..{branch}
- Tag range: as specified (e.g., `v1.0.0..v1.1.0`)

Run `git log --oneline <range>` and `git diff --name-only <range>` to understand what changed.

## Step 2 — Find Doc Files

Search for: README*, ARCHITECTURE*, CONTRIBUTING*, CLAUDE.md, CHANGELOG*, docs/*.md, root *.md files.

## Step 3 — Staleness Detection

For each doc file, cross-reference against the diff:

| Doc Section Type | Stale When | Detection Method |
|------------------|-----------|-----------------|
| API reference | Endpoints added/removed/changed | Route definitions in diff |
| Setup/install | Dependencies changed | package.json/pyproject.toml/Cargo.toml in diff |
| Architecture | New modules/services | New directories or major file additions |
| Config reference | Env vars/settings changed | Config keys in diff |
| CLI usage | Commands/flags changed | argparse/commander/clap definitions in diff |
| Feature docs | Behavior changed | Business logic files in diff |

## Step 4 — Generate Edits

For each stale section:
- Read the doc file in full
- Identify the specific lines that are outdated
- Generate minimal, surgical edits (preserve voice/style)
- Add sections only for genuinely new features
- Remove references to deleted features

## Step 5 — Present for Approval

Show all proposed changes as diffs. Wait for approval before committing.

### Rules
- No wholesale rewrites — update facts, not prose style
- No speculative documentation (unshipped features)
- No updating counts/stats without verification from code
- Ask before committing any changes
