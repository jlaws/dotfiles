# Task: Remove Hardcoded and Unresolved Paths from the Shell Environment

## Context
Found while removing the rustup and elan installers (`refactor/drop-rust-elan-install`). The shell
files assume one account and one Homebrew layout, so they misbehave for any other user of this repo:
- `.extra:18`: `export PATH="/Users/jlaws/.antigravity/antigravity/bin:$PATH"` names one account's
  home directory instead of `$HOME`.
- `.extra:20`: `export PATH="${HOMEBREW_PREFIX}/opt/openssl/bin:$PATH"`. No dotfile runs
  `brew shellenv` or sets `HOMEBREW_PREFIX`, so this usually expands to `/opt/openssl/bin`.
- `.zshrc:78`: `fpath=(~/.zfunc /opt/homebrew/share/zsh/site-functions $fpath)` hardcodes the Apple
  Silicon prefix; an Intel Mac uses `/usr/local`.
- `.zshrc:80`: `source $ZSH/oh-my-zsh.sh` sources `~/.oh-my-zsh`, which `setup.sh` never installs,
  so a fresh machine prints an error on every new shell.

## Problem
A standard user, a second account, or an Intel Mac gets PATH entries that point nowhere, and a fresh
machine gets a shell startup error.

## Scope
1. Use `$HOME` in `.extra:18`.
2. Set `HOMEBREW_PREFIX` from `brew --prefix` (or `brew shellenv`) before `.extra:20` and
   `.zshrc:78` use it, and guard both when Homebrew is absent.
3. Either install oh-my-zsh in setup or guard `.zshrc:80` with `[ -f "$ZSH/oh-my-zsh.sh" ]`.
4. Add a test that sources the shell files in a clean environment and asserts no error output.
