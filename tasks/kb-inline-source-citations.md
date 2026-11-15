# Task: Move Remaining Inline Source Citations to REFERENCES.md

## Context
Found while planning `feat/harness-best-practices`.

## Problem
`.agents/skills/writing-skills/SKILL.md:193` cites Meincke et al. (2025) with figures inline. The `:213` and `:244` headings carry "(from Anthropic's ...)" provenance. Credits belong only in `REFERENCES.md`. The same text is in `.gemini/antigravity-cli/skills/writing-skills/SKILL.md`.

## Scope
1. Credit each source in `REFERENCES.md`, one line each: `name - URL - what, where`.
2. Drop the inline provenance in the `.agents` file and its `.gemini` copy, keeping them byte-identical.
3. Run `make verify`.
