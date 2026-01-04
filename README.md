# Joe's dotfiles

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

### 2. Install Powerline Fonts (Optional)

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

### 3. Clone This Repository

```zsh
git clone https://github.com/jlaws/dotfiles.git ~/Workspace/dotfiles
cd ~/Workspace/dotfiles
```

### 4. Run the Setup Script

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

### 2. Install Homebrew Packages
- **GNU utilities**: `coreutils`, `findutils`, `gnu-sed`, `moreutils`
- **Updated tools**: `vim`, `grep`, `openssh`, `wget`
- **Git tools**: `git`, `git-lfs`
- **Shell utilities**: `autojump`

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
- **Terminal/iTerm2**: Set font to a Powerline font
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

## Credits

Inspired by [Mathias Bynens' dotfiles](https://github.com/mathiasbynens/dotfiles).
