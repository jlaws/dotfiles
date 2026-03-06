# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

Personal macOS dotfiles and development environment configuration. Combines traditional Unix dotfile management with an extensive Claude Code knowledge base (commands, skills).

## Commands

### Setup & Installation
```bash
./setup.py           # Interactive: syncs dotfiles, installs packages, configures macOS
./setup.py --force   # Non-interactive: skips all confirmation prompts
```

### What setup.py does:
1. **Syncs dotfiles** to `~` via rsync (excludes .git, setup.py, README)
2. **Installs Homebrew packages**: coreutils, findutils, gnu-sed, moreutils, vim, grep, openssh, screen, wget, git, git-lfs, gh, autojump, mermaid-cli, uv, node, pyright, rust-analyzer
3. **Configures macOS**: ~200 `defaults write` commands for Finder, Dock, Safari, security, etc.

### Linting
```bash
make check    # ruff check + mypy strict
make fix      # auto-fix + format
```

## Repository Structure

```
dotfiles/
├── Root dotfiles (.zshrc, .extra, .gitconfig, .vimrc, .editorconfig, etc.)
├── ghosty_config.txt  # Ghostty terminal configuration reference
├── setup.py           # Main installation script (Python)
├── pyproject.toml     # Project config, ruff + mypy
├── Makefile           # lint, format, typecheck targets
└── .claude/           # Claude Code knowledge base
    ├── CLAUDE.md      # Global standards (synced to ~/.claude/)
    ├── agents/        # 13 specialist agents
    ├── commands/      # 20 commands across 13 categories
    ├── references/    # 167 domain knowledge files
    └── skills/        # 16 contextual skill workflows
```

## Key Files

| File | Purpose |
|------|---------|
| `.zshrc` | Loads Oh My Zsh, sources `.extra` |
| `.extra` | 60+ aliases, functions, PATH setup (229 lines) |
| `.gitconfig` | Git aliases (`l`, `s`, `d`, `go`, `dm`, `amend`) |
| `.vimrc` | Solarized Dark, relative line numbers, centralized backups |
| `setup.py` | Main setup script (Python, ~965 lines) |
| `pyproject.toml` | ruff + mypy config, project metadata |
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
- macOS `defaults write` commands in setup.py use data-driven `Default` NamedTuples
- Python code linted with `ruff` and type-checked with `mypy --strict`
- Claude knowledge base files are markdown with YAML frontmatter
