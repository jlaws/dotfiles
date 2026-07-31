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

`setup.sh` is a thin shim that runs the `macos_setup` Python package (Python 3.12+, standard
library only — no virtualenv or pip installs at runtime). It performs three main tasks and
**archives every file and setting it changes** so a run can be undone later (see
[Uninstall / Reset](#uninstall--reset)).

### 1. Sync Dotfiles
Copies the dotfiles and agent configs to your home directory. Before overwriting any existing
file, the original is saved into the run's archive; newly created files are tracked so they can
be removed on uninstall. The Vim runtime tree includes the Solarized colorscheme and creates
`~/.vim/backup`, `~/.vim/undo`, and `~/.vim/swap`.

### 2. Install Packages (via Homebrew)
- **GNU utilities**: `coreutils`, `findutils`, `gnu-sed`, `moreutils`
- **Updated tools**: `vim`, `grep`, `openssh`, `screen`, `wget`
- **Git tools**: `git`, `git-lfs`, `gh`
- **Shell utilities**: `autojump`, `mermaid-cli`
- **Language tools**: `uv`
- **Rust tooling**: `rustup`, stable toolchain, `rust-analyzer` component
- **Language servers**: `node`, `pyright`

Homebrew packages are **not** removed on uninstall.

### 3. Configure macOS
Sets hundreds of macOS preferences including:
- **UI/UX**: Faster animations, expanded save/print dialogs, disabled auto-correct
- **Input**: Tap to click, fast key repeat, natural scrolling disabled
- **Finder**: Show hidden files, path bar, status bar, list view default
- **Dock**: Auto-hide, no recent apps, fast animations
- **Safari**: Developer tools enabled, privacy settings, no auto-fill
- **Security**: Password required immediately after sleep

Before each setting is changed, its prior value (and a full snapshot of the affected preference
domain) is recorded in the archive so it can be restored.

## Uninstall / Reset

Every run writes a timestamped archive under `~/.dotfile-archive/<YYYY-MM-DD-HHMMSS>/`
containing the files it replaced, a snapshot of each macOS preference domain it touched, and a
`manifest.json` recording what changed. A `latest` symlink points at the newest archive.

```zsh
./setup.sh --list-archives        # show archives with timestamps
./setup.sh --uninstall            # revert the most recent run (latest)
./setup.sh --uninstall 2026-07-03-141530   # revert a specific run
./setup.sh --dry-run -m           # preview what a step would change, write nothing
```

Reset is **guarded**: a file or macOS setting is only reverted if its current value still
matches what setup applied. If you changed it afterward, it is left untouched and logged as
`user-modified, left as-is`. Files that setup replaced are restored from the archive; files it
newly added are removed. System-level settings (`pmset`, `nvram`, `systemsetup`, firewall) are
reverted on a best-effort basis — readable values are restored, and anything that can't be read
back precisely is returned to a known macOS default and logged. Empty parent directories created
while installing managed files are retained.

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
| `.vim/` | Vim Solarized colorscheme, syntax files, and state directories |
| `.hushlogin` | Suppress "Last login" message in terminal |
| `.gitattributes` | Git file handling attributes |
| `.wgetrc` | Wget configuration |
| `ghosty_config.txt` | Ghostty terminal configuration reference |
| `setup.sh` | Entry-point shim that runs the `macos_setup` package |
| `macos_setup/` | Python package: install, archive, and uninstall/reset logic |

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

Each re-run creates its own timestamped archive, so you can always roll back to the state before
the most recent run with `./setup.sh --uninstall`.

### Claude

#### Commands & Skills

The `.claude/` directory is self-contained with `/j-*` commands, specialist agents, skills, and references.

### Codex

The `.codex/` directory contains Codex-native agents, prompts, hooks, and command rules. Reusable workflows and `$cmd-j-*` command skills live under `.agents/`, which Codex discovers directly. For example, invoke `$cmd-j-tdd` or `$cmd-j-plan` in Codex. The files under `.codex/prompts/` remain available through `/prompts:j-tdd` style slash commands.

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
| `rust-analyzer-lsp` | Rust | `rustup component add rust-analyzer` (in setup.sh) |

## Credits

Built on the shoulders of others. See [REFERENCES.md](REFERENCES.md) for sources of inspiration and borrowed patterns — dotfiles lineage, the agent knowledge base, and vendored files.

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
/plugin install commit-commands@claude-plugin-directory
/plugin install claude-md-management@claude-plugin-directory
```
