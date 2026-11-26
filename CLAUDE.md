# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

Personal macOS dotfiles and development environment configuration. Combines traditional Unix dotfile management with a shared Codex/Gemini knowledge base (`.agents/`) and tool-specific configs for Claude Code, Codex, and Google Antigravity.

## Commands

### Setup & Installation
```bash
./setup.sh                 # Interactive: syncs dotfiles, installs packages, configures macOS
./setup.sh --force         # Non-interactive: skips all confirmation prompts
./setup.sh --list-archives # List timestamped archives of prior runs
./setup.sh --uninstall     # Revert the most recent run (guarded)
./setup.sh --dry-run -m    # Preview a step's changes without writing
```

`setup.sh` is a shim that runs the `macos_setup` Python package (stdlib only; it runs on the system `python3`, 3.9 or later; the dev venv is 3.12+).

### Linting & Tests
```bash
make check    # ruff format --check + ruff check + ty (via .venv); healthy run ends "All checks passed!"
make fix      # auto-fix + format
make test     # stdlib unittest suite (python -m unittest); healthy run ends "Ran <count> tests in <secs>s" then "OK"
make verify   # check + test -- the full gate (make test alone already fails on audit WARN findings); ends like make test
python3 .claude/skills/skill-audit/scripts/audit.py .   # repo copy; ends "Findings:   0 fail  |  0 warn"
```

### KB evals (see `evals/README.md`)
```bash
make eval-stage      # free: stage kb and vanilla plugins under $TMPDIR
make eval-lint       # free: lint eval cases; ends "0 problems"
make eval-pilot      # bills ANTHROPIC_API_KEY: pilot cases
make eval-smoke ASSETS="..."      # bills ANTHROPIC_API_KEY: one-run smoke for named assets
make eval-baseline MODEL=...      # bills ANTHROPIC_API_KEY: full baseline for one model; CAP=<usd> stops new runs once purpose baseline has spent that
make eval-compare OLD=... NEW=... # free: compare two baseline summaries
python3 -m evals.kbeval settle --purpose P --staging S --model M --suite X --batch B [--id R] --actual-usd N   # free: settle one open reservation to its real cost (--reason defaults to interrupted; --id picks one of several; prints the settle row as JSON, exit 0; `settle: <error>`, exit 1)
make eval-spend      # free: snapshot the spend ledger into evals/spend.json; exits 1 over the ceiling
make eval-headroom   # free: classify effect and workflow cases from evals/baselines/; each suite ends "<n> cases: <s> saturated, <d> dead, <k> kb-below-vanilla, <x> discriminating, <u> unmeasured"
make eval-affected   # free: assets a diff touches, as one line for ASSETS=, and the projected smoke cost on stderr
python3 -m evals.kbeval.hillclimb adopt --asset A... --before P... --after P... --effects P   # free: prints one "ADOPT <asset> <verdict>" line per asset, then any neighbour lines
python3 -m evals.kbeval summarize # free: check results and write evals/baselines/ (writes nothing if a check fails)
```

### What setup.sh does:
Steps run in privilege order: every per-user change first (archive saved), then Homebrew (needs an admin-group account; skipped with a warning otherwise), then one `sudo` prompt for system-wide settings. Cancelling the prompt keeps the per-user changes; uninstall follows the same order.

1. **Syncs dotfiles and agent configurations** to `~` (root dotfiles, `.vim/` runtime, and agent skills/commands/agents replaced; machine-specific agent configuration files merged with repo keys; installed `.md`/`.toml` agent files get `~/.agents/references/`, `~/.claude/references/`, and `~/.gemini/antigravity-cli/skills/` expanded to absolute paths), archiving replaced or merged originals first. Deprecated Codex prompt copies and repo-owned Claude/Codex logging hooks are removed without deleting unrelated local prompts/hooks or historical logs
2. **Installs tooling**: Homebrew packages (no Rust or Lean tooling: rustup, `rust-analyzer`, and elan are installed by the user or the project that needs them; the Claude Code, Codex, and Antigravity CLIs are not installed): coreutils, moreutils, findutils, fd, wget, just, vim, grep, jq, openssh, git, gh, autojump, mermaid-cli, poppler, agent-browser, uv, node, rg, mole; then symlinks `gsha256sum` to `sha256sum`, downloads the Chrome binary `agent-browser` drives, installs `typescript-language-server` and `typescript` via npm, and finishes with `brew cleanup`
3. **Configures macOS**: ~200 `defaults` settings for Finder, Dock, Safari, security, etc., snapshotting each domain first; user/host-scope settings and `~/Library` apply with the per-user steps, `sudo`-scope settings and system items (`pmset`, `nvram`, firewall, `/Volumes`) after the `sudo` prompt

Every run writes a timestamped archive to `~/.dotfile-archive/<timestamp>/` (files, per-domain
plist snapshots, `manifest.json`). `--uninstall` reverts a run; reverts are **guarded** — a file
or setting is only restored if it still matches what setup applied, otherwise left as-is.
Homebrew packages are not uninstalled.

## Repository Structure

```
dotfiles/
├── Root dotfiles (.zshrc, .extra, .gitconfig, .vimrc, .editorconfig, etc.)
├── .vim/               # Solarized colorscheme, syntax files, and state directories
├── ghosty_config.txt  # Ghostty terminal configuration reference
├── setup.sh           # Entry-point shim → python3 -m macos_setup
├── macos_setup/       # Python package: install + archive + uninstall/reset
├── tests/             # stdlib unittest suite for `macos_setup`, the KB, and `evals`
├── evals/             # KB eval harness, cases, baselines (repo-only, not synced)
├── docs/adr/          # Decision records (README.md + template.md are scaffolding, not ADRs)
├── pyproject.toml     # ruff config, project metadata
├── Makefile           # lint, format, fix, test targets
├── .agents/           # Shared Codex/Gemini KB (agentskills.io spec)
│   ├── skills/        # cmd-j-* entry points + reusable workflows
│   └── references/    # Shared domain knowledge by category
├── .claude/           # Claude Code (self-contained)
│   ├── CLAUDE.md, settings.json
│   ├── agents/, commands/, skills/, references/
├── .codex/            # Codex config + native agents/rules
│   ├── AGENTS.md, config.toml
│   ├── agents/, rules/
└── .gemini/           # Antigravity (skills/settings in antigravity-cli/, agents in config/agents/)
    ├── GEMINI.md
    ├── config/agents/, antigravity-cli/
```

## Key Files

| File | Purpose |
|------|---------|
| `.zshrc` | Resolves the Homebrew prefix, loads Oh My Zsh if installed, sources `.extra` |
| `.extra` | 60+ aliases, functions, PATH setup, file descriptor limits (249 lines) |
| `.gitconfig` | Git aliases (`l`, `s`, `d`, `go`, `dm`, `amend`) |
| `.vimrc` | Solarized Dark, relative line numbers, centralized backup/undo/swap files |
| `.vim/` | Solarized colorscheme, syntax files, and state directories |
| `setup.sh` | Shim that execs `python3 -m macos_setup` |
| `macos_setup/` | Install/archive/uninstall package (stdlib only) |
| `evals/` | KB eval harness (`kbeval`), cases, fixtures, and budget; baselines under `evals/baselines/` |
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
- Agent configuration merges (`.claude/settings.json`, `.codex/config.toml`, `.gemini/antigravity-cli/settings.json`) run during `sync_agents`; they must remain Python 3.9 stdlib-only with zero runtime pip dependencies. Skills and commands are overwritten in place. Sync prunes `j-*` entries the repo no longer ships from the trees in `AGENT_ORPHAN_GLOBS` (with `.agents/skills/cmd-j-*`), beside the `AGENT_REMOVAL_GLOBS` patterns. `j-` and `cmd-j-` names are reserved for repo-shipped commands; a symlinked entry is kept with a warning, and a root the repo ships nothing in is not pruned; any other renamed or deleted asset needs an `AGENT_REMOVALS` entry in `macos_setup/dotfiles.py`.
- Agent skills follow the [agentskills.io](https://agentskills.io/specification) spec (SKILL.md with YAML frontmatter)
- Codex commands use the canonical `.agents/skills/cmd-j-*` skills. Deprecated `.codex/prompts/j-*` copies are removed on sync. Codex role files omit model and effort; `.codex/config.toml` owns session and default delegate choices, and task dispatch supplies overrides. V2 permits six concurrent spawned threads; depth three remains prose guidance
- `.claude/` is written for the Claude 5 generation, `.agents/skills/` is tailored for Codex (GPT-6.2), and `.gemini/` is tailored for Google Antigravity (Gemini 4), while `.agents/references/` remains the shared Codex/Antigravity reference tree. Asset sets (and supporting files across `.agents/skills/` and `.gemini/antigravity-cli/skills/`) are kept in parity, enforced by `tests/test_agent_config.py`. Declare any single-tree asset in that file's exception lists
- **References are kept in parity too, by section rather than by byte.** Every file under `.claude/references/` has an `.agents/` twin at the same path exposing the same headings; bodies may differ, because the trees serve different tools. Edit one tree and you must edit the other. This check ships without an exception list by design — see `docs/adr/workflow/reference-tree-section-parity.md`
- A shared reference must not cite `.claude/references/`: Codex and Gemini have no such checkout, so the pointer resolves nowhere. Cite `references/<category>/<file>.md`, which is also the form `audit.py`'s XR-7 validates from the reference root. Citing assets use one form per tree and kind (`REFERENCE_FORMS` in `tests/test_agent_config.py`): `~/.agents/references/<category>/` in `.agents`, `.codex`, and `.gemini` (sync expands `~` to the home path), `~/.claude/references/<category>/` in `.claude/commands` and `.claude/agents`, and `${CLAUDE_SKILL_DIR}/../../references/<category>/` in `.claude/skills`; never a path that resolves against the project
- `tests/test_agent_config.py` owns anything spanning two trees, not just parity: Codex agents must declare `name`/`description`/a non-empty `developer_instructions`, Gemini agents must declare `model: inherit` and explicit Antigravity `tools:`, skills in `.agents/skills/` and `.gemini/antigravity-cli/skills/` carry only `name` and `description` in frontmatter and ship the same supporting files per workflow skill, and no tracked asset in any tree may name a script the repo does not ship. `audit.py` is Claude-tree only by its own docstring, so cross-tree checks belong in the test file
- `.claude/CLAUDE.md`, `.codex/AGENTS.md`, and `.gemini/GEMINI.md` are synced to `~` and loaded in every repo. Keep guidance that only applies to this repo in this file instead
- KB assets under `.claude/`, `.agents/`, `.codex/`, and `.gemini/` contain no em dash and no prose ` -- ` outside code spans and fences (a CLI argument separator stays in a code span); `tests/test_agent_config.py` enforces both. Use a comma, semicolon, colon, or parentheses
- Do not hardcode counts of KB assets (agents, commands, references, skills) — they go stale
- `.claude/CLAUDE.md`, `.codex/AGENTS.md`, and `.gemini/GEMINI.md` are re-sent on every request, so
  each carries a byte ceiling in `tests/test_agent_config.py` (`ALWAYS_LOADED_CEILINGS`). Bump it
  deliberately when you add a documented rule; do not bump it for phrasing creep. These three hold
  pointers, never restatements — guidance belongs in a skill or reference that loads on demand
