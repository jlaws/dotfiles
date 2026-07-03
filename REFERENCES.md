# References

Sources of inspiration and borrowed patterns behind this repo's dotfiles, configs, and agent knowledge base. Vendored third-party files retain their own license headers; the repo `LICENSE` covers original work.

## Dotfiles lineage

- **mathiasbynens/dotfiles** — https://github.com/mathiasbynens/dotfiles — macOS `defaults write` patterns in `setup.sh`, `.gitconfig` alias style, `.wgetrc`, and heritage reference comments. Direct signal: `.gitattributes` (line 3).
- **Solarized (Ethan Schoonover)** — https://ethanschoonover.com/solarized — vim colorscheme (`colorscheme solarized` in `.vimrc`); vendored as `.vim/colors/solarized.vim` under its own MIT header.
- **oh-my-zsh** — https://github.com/ohmyzsh/ohmyzsh — zsh framework loaded by `.zshrc`.

## Agent knowledge base & tooling

- **superpowers (obra / Jesse Vincent)** — https://github.com/obra/superpowers — skill-system conventions: `SKILL.md` structure, `superpowers:skill-name` cross-reference style, and the git-worktree workflow. Reflected across `.agents/skills/` and `.claude/skills/`.
- **Agent Skills specification (agentskills.io)** — https://agentskills.io/specification — the `.agents/` knowledge base follows this spec (`SKILL.md` + YAML frontmatter, `agent-*`/`cmd-*` naming). Cited in `CLAUDE.md` and `.agents/README.md`.
- **Google Style Guides** — https://google.github.io/styleguide/ — `Source:` citations throughout `references/languages/*` (Python, Go, Swift API Design, and others).
- **Keep a Changelog** — https://keepachangelog.com/ — changelog format used in `references/documentation/changelog-patterns.md`.

## Vendored third-party files

Bundled verbatim; each retains its original attribution and license header.

- `.vim/colors/solarized.vim` — Ethan Schoonover (MIT). Also references tpope/vim-pathogen (https://github.com/tpope/vim-pathogen) and urso/dotrc.
- `.vim/syntax/json.vim` — Jeroen Ruigrok van der Werven.
