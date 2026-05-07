# Dotfiles

Dotfiles and macOS setup for a streamlined development environment.

## What's Included

- **Shell Configuration** (`.zshrc`, `.extra`) - Zsh configuration with aliases, functions, and environment setup
- **Git Configuration** (`.gitconfig`, `.gitignore`) - Global git settings and ignore patterns
- **Editor Configuration** (`.editorconfig`, `.vimrc`) - Consistent coding styles across editors
- **macOS Preferences** - Sensible defaults for Finder, Dock, Safari, and more
- **Homebrew Packages** - Essential command-line tools

## Installation

### 1. Install Oh My Zsh

```zsh
sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
```

### 2. Install Homebrew

```zsh
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Follow the post-install instructions to add Homebrew to your PATH.

### 3. Install Ghostty Terminal

Download and install [Ghostty](https://ghostty.org/).

### 4. Install Powerline Fonts (Optional)

For the best terminal experience with special characters:

```zsh
# Clone and install
git clone https://github.com/powerline/fonts.git --depth=1
cd fonts
./install.sh
cd ..
rm -rf fonts
```

Then set your terminal font to a Powerline font (e.g., "Meslo LG M for Powerline").

### 5. Clone This Repository

```zsh
git clone https://github.com/jlaws/dotfiles.git ~/Workspace/dotfiles
cd ~/Workspace/dotfiles
```

### 6. Update Personal Configuration

Before running the setup script, update `.gitconfig` with your own name and email:

```gitconfig
[user]
	name = Your Name
	email = your-email@example.com
```

### 7. Run the Setup Script

```zsh
./setup.sh
```

Or skip confirmation prompts:

```zsh
./setup.sh --force
```

## What the Setup Script Does

The setup script performs three main tasks:

### 1. Sync Dotfiles
Copies all dotfiles to your home directory using `rsync`, preserving any local changes not in the repo.

### 2. Install Packages (via Homebrew)
- **GNU utilities**: `coreutils`, `findutils`, `gnu-sed`, `moreutils`
- **Updated tools**: `vim`, `grep`, `openssh`, `screen`, `wget`
- **Git tools**: `git`, `git-lfs`, `gh`
- **Shell utilities**: `autojump`, `mermaid-cli`
- **Language servers**: `node`, `pyright`, `rust-analyzer`

### 3. Configure macOS
Sets hundreds of macOS preferences including:
- **UI/UX**: Faster animations, expanded save/print dialogs, disabled auto-correct
- **Input**: Tap to click, fast key repeat, natural scrolling disabled
- **Finder**: Show hidden files, path bar, status bar, list view default
- **Dock**: Auto-hide, no recent apps, fast animations
- **Safari**: Developer tools enabled, privacy settings, no auto-fill
- **Security**: Password required immediately after sleep

## Manual Configuration

Some settings can't be automated and require manual setup:

### System Preferences
- **Security & Privacy** → FileVault (enable disk encryption)
- **Security & Privacy** → Firewall (enable)
- **Keyboard** → Modifier Keys (Caps Lock → Escape, if desired)

### Applications
- **Ghostty**: Primary terminal — config reference at `ghosty_config.txt` in repo root
- **Xcode**: Sign in with Apple ID, install additional components

## File Overview

| File | Purpose |
|------|---------|
| `.zshrc` | Zsh configuration, loads Oh My Zsh and sources `.extra` |
| `.extra` | Aliases, functions, PATH, and environment variables |
| `.gitconfig` | Git configuration (aliases, colors, defaults) |
| `.gitignore` | Global gitignore patterns |
| `.editorconfig` | Editor settings (indent style, charset, etc.) |
| `.vimrc` | Vim configuration |
| `.hushlogin` | Suppress "Last login" message in terminal |
| `.gitattributes` | Git file handling attributes |
| `.wgetrc` | Wget configuration |
| `ghosty_config.txt` | Ghostty terminal configuration reference |
| `setup.sh` | Main setup script |

## Customization

### Adding Local Overrides

The `.extra` file is sourced by `.zshrc`. You can create a `~/.extra.local` file for machine-specific settings that won't be committed:

```zsh
# Example ~/.extra.local
export WORK_API_KEY="secret"
alias myproject="cd ~/work/myproject"
```

Then add to your `.zshrc`:
```zsh
[ -f ~/.extra.local ] && source ~/.extra.local
```

### Updating

To pull the latest changes and re-sync:

```zsh
cd ~/Workspace/dotfiles
./setup.sh
```

### Claude

#### Commands & Skills

The `.claude/` directory contains an extensive knowledge base of `/j-*` commands, specialist agents, skills, and references that mirror the shared `.agents/` knowledge base.

### Gemini

The `.gemini/` directory contains a parallel configuration optimized for the Gemini CLI:

- **`GEMINI.md`** — persistent instructions (auto-loaded by Gemini)
- **`commands/*.toml`** — `/j-*` slash commands (TOML format, not Markdown)
- **`agents/*.md`** — specialist subagents invoked via `@agent-<name>`
- **`hooks/*.sh`** — lifecycle scripts (JSON stdout protocol)
- **`policies/default.toml`** — fine-grained shell command allow/deny rules
- **`settings.json`** — Gemini-schema config (model, hooks, policy path)

Skills and references are NOT duplicated under `.gemini/` — Gemini natively auto-discovers them at `~/.agents/skills/` and reads `~/.agents/references/` by path. The existing `setup.sh -c` sync covers everything.

#### Recommended Plugins

**LSP Plugins** (code intelligence — goToDefinition, findReferences, hover, diagnostics):

| Plugin | Language | Prerequisite |
|--------|----------|-------------|
| `pyright-lsp` | Python | `brew install pyright` (in setup.sh) |
| `typescript-lsp` | TS/JS | `npm install -g typescript-language-server typescript` (in setup.sh) |
| `swift-lsp` | Swift | Xcode (bundled) |
| `gopls-lsp` | Go | `go install golang.org/x/tools/gopls@latest` (in setup.sh) |
| `rust-analyzer-lsp` | Rust | `brew install rust-analyzer` (in setup.sh) |

**Workflow Plugins:**

| Plugin | Adds |
|--------|------|
| `commit-commands` | `/commit`, `/commit-push-pr`, `/clean_gone` |
| `claude-md-management` | `/revise-claude-md` for CLAUDE.md maintenance |

Install after running `setup.sh` launch claude and run:
```bash
/plugin install pyright-lsp@claude-plugin-directory
/plugin install typescript-lsp@claude-plugin-directory
/plugin install swift-lsp@claude-plugin-directory
/plugin install rust-analyzer-lsp@claude-plugin-directory
/plugin install gopls-lsp@claude-plugin-directory
/plugin install commit-commands@claude-plugin-directory
/plugin install claude-md-management@claude-plugin-directory
```

