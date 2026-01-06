# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

Personal macOS dotfiles and development environment configuration. Combines traditional Unix dotfile management with an extensive Claude Code knowledge base (agents, commands, skills).

## Commands

### Setup & Installation
```bash
./setup.sh           # Interactive: syncs dotfiles, installs packages, configures macOS
./setup.sh --force   # Non-interactive: skips all confirmation prompts
```

### What setup.sh does:
1. **Syncs dotfiles** to `~` via rsync (excludes .git, setup.sh, README, init/)
2. **Installs Homebrew packages**: coreutils, findutils, gnu-sed, vim, grep, git, git-lfs, autojump
3. **Configures macOS**: ~200 `defaults write` commands for Finder, Dock, Safari, security, etc.

## Repository Structure

```
dotfiles/
├── Root dotfiles (.zshrc, .extra, .gitconfig, .vimrc, .editorconfig, etc.)
├── init/              # App configs (Sublime, iTerm colors, Spectacle)
├── setup.sh           # Main installation script
└── .claude/           # Claude Code knowledge base
    ├── CLAUDE.md      # Global coding standards (already exists)
    ├── agents/        # 79 specialized AI agent definitions
    ├── commands/      # 47 reusable command definitions
    └── skills/        # 154 executable skill workflows
```

## Key Files

| File | Purpose |
|------|---------|
| `.zshrc` | Loads Oh My Zsh, sources `.extra` |
| `.extra` | 50+ aliases, functions, PATH setup (223 lines) |
| `.gitconfig` | Git aliases (`l`, `s`, `d`, `go`, `dm`, `amend`) |
| `.vimrc` | Solarized Dark, relative line numbers, centralized backups |
| `setup.sh` | Main orchestration script (~700 lines) |

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
- macOS `defaults write` commands in setup.sh follow pattern: domain, key, type, value
- Claude knowledge base files are markdown with YAML frontmatter
