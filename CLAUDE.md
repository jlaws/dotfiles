# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

Personal macOS dotfiles and development environment configuration. Combines traditional Unix dotfile management with a shared Codex/Gemini knowledge base (`.agents/`) and tool-specific configs for Claude Code, Codex, and Gemini CLI.

## Commands

### Setup & Installation
```bash
./setup.sh                 # Interactive: syncs dotfiles, installs packages, configures macOS
./setup.sh --force         # Non-interactive: skips all confirmation prompts
./setup.sh --list-archives # List timestamped archives of prior runs
./setup.sh --uninstall     # Revert the most recent run (guarded)
./setup.sh --dry-run -m    # Preview a step's changes without writing
```

`setup.sh` is a shim that runs the `macos_setup` Python package (stdlib only, Python 3.12+).

### Linting & Tests
```bash
make check    # ruff check + ty (via .venv)
make fix      # auto-fix + format
make test     # stdlib unittest suite (python -m unittest)
```

### What setup.sh does:
1. **Syncs dotfiles** to `~` (root dotfiles + agent configs), archiving replaced files first
2. **Installs Homebrew packages**: coreutils, findutils, fd, gnu-sed, moreutils, vim, grep, openssh, screen, wget, git, git-lfs, gh, autojump, mermaid-cli, poppler, rustup, mold, uv, node, pyright; initializes the stable Rust toolchain and `rust-analyzer` component through rustup
3. **Configures macOS**: ~200 `defaults` settings for Finder, Dock, Safari, security, etc., snapshotting each domain first

Every run writes a timestamped archive to `~/.dotfile-archive/<timestamp>/` (files, per-domain
plist snapshots, `manifest.json`). `--uninstall` reverts a run; reverts are **guarded** — a file
or setting is only restored if it still matches what setup applied, otherwise left as-is.
Homebrew packages are not uninstalled.

## Repository Structure

```
dotfiles/
├── Root dotfiles (.zshrc, .extra, .gitconfig, .vimrc, .editorconfig, etc.)
├── ghosty_config.txt  # Ghostty terminal configuration reference
├── setup.sh           # Entry-point shim → python3 -m macos_setup
├── macos_setup/       # Python package: install + archive + uninstall/reset
├── tests/             # stdlib unittest suite for macos_setup
├── pyproject.toml     # ruff config, project metadata
├── Makefile           # lint, format, fix, test targets
├── .agents/           # Shared Codex/Gemini KB (agentskills.io spec)
│   ├── skills/        # cmd-j-* entry points + reusable workflows
│   └── references/    # Shared domain knowledge by category
├── .claude/           # Claude Code (self-contained)
│   ├── CLAUDE.md, settings.json
│   ├── agents/, commands/, hooks/, skills/, references/
├── .codex/            # Codex config + native agents/prompts
│   ├── AGENTS.md, config.toml
│   ├── agents/, prompts/, hooks/, rules/
└── .gemini/           # Gemini CLI (skills/refs reused from .agents/)
    ├── GEMINI.md, settings.json
    ├── agents/, commands/, hooks/, policies/
```

## Key Files

| File | Purpose |
|------|---------|
| `.zshrc` | Loads Oh My Zsh, sources `.extra` |
| `.extra` | 60+ aliases, functions, PATH setup (234 lines) |
| `.gitconfig` | Git aliases (`l`, `s`, `d`, `go`, `dm`, `amend`) |
| `.vimrc` | Solarized Dark, relative line numbers, centralized backups |
| `setup.sh` | Shim that execs `python3 -m macos_setup` |
| `macos_setup/` | Install/archive/uninstall package (stdlib only) |
| `ghosty_config.txt` | Ghostty terminal configuration reference |

## Shell Aliases (from .extra)

**Navigation**: `..`, `...`, `dl` (~/Downloads), `dt` (~/Desktop), `p` (~/Workspace)
**Git shortcuts**: `g`, `ga`, `gm`, `gcf`, `gr`, `gs`, `grmb`, `gitclean`
**Cleanup**: `rmdd`, `rma`, `rmp`, `emptytrash`, `update`
**Network**: `ip`, `localip`, `ips`, `flush`
**macOS**: `show`/`hide` (hidden files), `afk` (lock screen)
**Swift**: `fm` (format), `fr` (lint), `fp` (format+lint)

## Editing Guidelines

- Shell configs use `#` comments, keep aliases short and documented
- `.extra` is the primary customization point (not `.zshrc`)
- macOS settings live in the `SETTINGS` registry in `macos_setup/macos_defaults.py` (one dict per setting: scope, domain, key, type, value); this single list drives both apply and revert
- `macos_setup` is stdlib-only (no runtime pip deps) so it runs on a fresh Mac; keep the subprocess boundary behind the `Runner` seam for testability
- Follow TDD for `macos_setup` changes; add/adjust `tests/` and keep `make test` + `make check` green
- Agent skills follow the [agentskills.io](https://agentskills.io/specification) spec (SKILL.md with YAML frontmatter)
- Do not hardcode counts of KB assets (agents, commands, references, skills) — they go stale
