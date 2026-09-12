# Architecture Decision Records

Conventions, and this repo's topic list. **Not an index** — ordering comes from each ADR's
`created` date, and there is no sequence counter anywhere.

Format, frontmatter, templates, and the amendment rules live in
`architecture-decision-records.md` under `references/architecture/` — `.claude/` for Claude,
`~/.agents/` for Codex and Gemini. Read that before writing one; this file does not restate it.
`template.md` beside this file is that reference's Standard ADR template, copied verbatim and ready
to start from.

## Topics

A topic is a single directory level naming the area a decision constrains.

| Topic | Constrains |
|-------|-----------|
| `context-efficiency/` | What enters and leaves the context window — output shaping, compression, retrieval |
| `workflow/` | How work gets done — planning, review, build/scope decisions |

Add a topic by adding a row here and a directory. Pick names from this repo's own boundaries.

## Scope

An ADR is for a decision that is significant **and** not easily reversible. Minor or cheap-to-undo
choices are recorded inline in the asset they affect. Every ADR fills in `## Reversal Conditions`,
so a future reader can tell whether those conditions have arrived.

`README.md` and `template.md` are scaffolding, not decisions. Exclude both by name whenever
`docs/adr/**/*.md` is used to enumerate ADRs.
