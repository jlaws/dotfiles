# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

Personal macOS dotfiles and development environment configuration. Combines traditional Unix dotfile management with an extensive Claude Code knowledge base (commands, skills).

## Commands

### Setup & Installation
```bash
./setup.sh           # Interactive: syncs dotfiles, installs packages, configures macOS
./setup.sh --force   # Non-interactive: skips all confirmation prompts
```

### What setup.sh does:
1. **Syncs dotfiles** to `~` via rsync (excludes .git, setup.sh, README)
2. **Installs Homebrew packages**: coreutils, findutils, gnu-sed, moreutils, vim, grep, openssh, screen, wget, git, git-lfs, gh, autojump, mermaid-cli, node, pyright, rust-analyzer
3. **Configures macOS**: ~200 `defaults write` commands for Finder, Dock, Safari, security, etc.

## Repository Structure

```
dotfiles/
├── Root dotfiles (.zshrc, .extra, .gitconfig, .vimrc, .editorconfig, etc.)
├── ghosty_config.txt  # Ghostty terminal configuration reference
├── setup.sh           # Main installation script
└── .claude/           # Claude Code knowledge base
    ├── CLAUDE.md      # Global standards + skills index (synced to ~/.claude/)
    ├── commands/      # 8 invocable commands (all are thin wrappers to skills)
    └── skills/        # 142 contextual skill workflows
```

## Key Files

| File | Purpose |
|------|---------|
| `.zshrc` | Loads Oh My Zsh, sources `.extra` |
| `.extra` | 60+ aliases, functions, PATH setup (229 lines) |
| `.gitconfig` | Git aliases (`l`, `s`, `d`, `go`, `dm`, `amend`) |
| `.vimrc` | Solarized Dark, relative line numbers, centralized backups |
| `setup.sh` | Main orchestration script (~770 lines) |
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
- macOS `defaults write` commands in setup.sh follow pattern: domain, key, type, value
- Claude knowledge base files are markdown with YAML frontmatter
