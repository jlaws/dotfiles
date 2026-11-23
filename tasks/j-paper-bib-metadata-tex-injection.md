# Task: Treat fetched citation metadata as untrusted TeX

## Context
Found in the diff review of `feat/kb-skill-spec-and-writing-rules` (2026-11-23). Pre-existing and
unrelated to that PR's rules, so it was filed rather than fixed there.

## Problem
- `j-paper` section d (all four copies: `.claude/commands/j-paper.md`,
  `.agents/skills/cmd-j-paper/SKILL.md`, `.codex/prompts/j-paper.md`,
  `.gemini/antigravity-cli/skills/j-paper/SKILL.md`) takes title, authors, and abstract from fetched
  arXiv, DOI, and venue pages and appends them to the `.bib` file.
- Whoever controls that metadata controls the TeX: `\input{...}` can pull a local file into the PDF,
  and `\write18` runs a command when the user builds with `--shell-escape`.
- The untrusted-data rules in the harness configs cover prompt injection, not TeX control sequences
  in written fields.

## Scope
1. Decide the rule: which control sequences a fetched field may carry. A blanket escape breaks
   legitimate accents such as `{\"o}`.
2. Add it to `references/research/literature-review.md` (both trees) and point `j-paper` section d at it.
3. Add an eval case or a test pin for the rule.
