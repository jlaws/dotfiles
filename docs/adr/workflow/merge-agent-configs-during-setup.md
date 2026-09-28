---
status: accepted
topic: workflow
created: 2026-09-28
updated: 2026-09-28
deciders: ["@jlaws"]
---
# Merge agent machine configs during setup, overwrite skills and commands

## Context

`setup.sh` synchronizes agent configurations for Claude Code (`.claude/`), OpenAI Codex (`.codex/`), Google Antigravity (`.gemini/`), and the shared knowledge base (`.agents/`).

Previously, `macos_setup/dotfiles.py:apply_file` unconditionally overwrote target files via `shutil.copy2`. Re-running `./setup.sh` clobbered machine-specific customizations, such as local environment settings, device permissions, and host-tailored model or tool overrides, stored in `.claude/settings.json` and `.codex/config.toml`.

At the same time, executable knowledge assets (skills under `.agents/skills` and `.gemini/antigravity-cli/skills`, commands under `.claude/commands`, and prompts under `.codex/prompts`) represent repository code that must match upstream definitions. Leaving stale or removed skills in the destination leads to phantom command discovery and broken references.

The setup runner must execute on bare macOS system Python (Python 3.9.6 floor) without third-party dependencies (`tomllib` was added in Python 3.11; `pip` packages are prohibited at runtime).

## Decision Drivers

* **Must preserve machine-local configuration** across `./setup.sh` runs without manual backup or re-entry.
* **Must keep skills, commands, prompts, and agent definitions authoritative** to ensure removed or refactored tools stay in sync.
* **Must run in Python 3.9 standard library** without virtual environments or runtime pip packages.
* **Must preserve guarded uninstall and rollback integrity** via `macos_setup.archive`.

## Decision

We will divide agent synchronization into two lifecycle policies:

1. **Executable Knowledge Assets (Overwrite)**: Skills, slash commands, agent personas, prompts, hooks, and references are copied directly from the repository, replacing destination files. Stale removed commands and legacy directories are cleaned up via `AGENT_REMOVALS` and `AGENT_REMOVAL_GLOBS`.
2. **Machine Configuration Files (Deep Key Merge)**: Target configuration files (`.claude/settings.json`, `.codex/config.toml`, and `.gemini/antigravity-cli/settings.json`) are merged when the target already exists:
   - Repository-managed keys update or insert into the target.
   - User-defined or machine-local keys and sections absent from the repository template are preserved.
   - For `.json`, merging uses Python stdlib `json` with recursive dictionary updates and order-preserving permission list unions (`permissions.allow`, `permissions.deny`). Hooks are replaced with repo hooks.
   - For `.toml`, merging uses a minimal stdlib-only section/key parser and serializer that preserves existing machine tables, keys, and comments without requiring `tomllib` or external packages. Hooks are replaced with repo hooks.
   - If a target configuration file contains invalid syntax, it is archived to `~/.dotfile-archive/` with a warning, and overwritten with the clean repository template.

Before applying the merge, the existing destination file is copied to the run's archive (`<archive>/files/`) and tracked in `manifest.json` with action `"replaced"` and the SHA256 of the merged result. Revert via `./setup.sh --uninstall` restores the user's pre-merge configuration if the file has not been altered post-setup.

## Rationale

Agent harnesses (Claude Code, Codex, Antigravity) expect single configuration files at deterministic paths (`~/.claude/settings.json`, `~/.codex/config.toml`) and do not provide native `.local` include or cascade directives. Merging incoming repository settings into the existing file allows upstream security policies and tool allowances to propagate while preserving user-added environment settings.

Overwriting skills and commands while merging configs aligns with their distinct lifecycles: code and workflows are stateless and repo-driven; configuration holds stateful host bindings.

## Consequences

**Gained**: Users can freely customize local agent settings without fear of `./setup.sh` or `./setup.sh -c` erasing their changes. Upstream permission additions and recommended defaults still apply cleanly. Guarded rollback continues to function.

**Accepted**: `macos_setup` must maintain a stdlib-only TOML merge implementation compatible with Python 3.9. Merge conflict rules must be deterministic (repo keys override target on collisions).

## Ruled Out

| Idea | Why ruled out | When |
|------|---------------|------|
| Git-style three-way text merge | Prone to syntax-breaking conflict markers during automated non-interactive runs (`--force`). | 2026-09-28 |
| Harness `.local` include configuration files | Upstream harnesses (`claude`, `codex`, `antigravity`) do not support an `include` or `.local` file pattern for their primary configuration files. | 2026-09-28 |
| Third-party TOML library via pip | Violates the strict standard-library-only requirement for `macos_setup` on fresh macOS installations (`README.md:79-80`, `CLAUDE.md:20`). | 2026-09-28 |

## Enforcement

- Owner: `macos_setup/dotfiles.py`
- Pinned by: `tests/test_dotfiles.py` and `tests/test_adr.py`

## Reversal Conditions

Upstream agent CLIs introduce native user-level override layers or include directives (e.g., `settings.local.json` or `config.local.toml`), removing the need for `macos_setup` to merge configuration files.

## Related

- [Reference trees share a section set, not a body](reference-tree-section-parity.md)
- [ADRs are living documents, and dead ones are deleted](adrs-are-living-documents.md)
