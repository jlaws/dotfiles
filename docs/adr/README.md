# Architecture Decision Records

Conventions, and this repo's topic list. **Not an index** — ordering comes from each ADR's
`created` date, and there is no sequence counter anywhere.

Format, frontmatter, templates, and the amendment rules live in
`.claude/references/architecture/architecture-decision-records.md`. Read that before writing one;
this file does not restate it, and there is deliberately no `template.md` copy here — duplicating a
template the reference already carries is the kind of thing `code-efficiency-ladder` rung 2 exists
to prevent.

## Topics

A topic is a single directory level naming the area a decision constrains.

| Topic | Constrains |
|-------|-----------|
| `context-efficiency/` | What enters and leaves the context window — output shaping, compression, retrieval |
| `workflow/` | How work gets done — planning, review, build/scope decisions |

Add a topic by adding a row here and a directory. Pick names from this repo's own boundaries.

## Scope

An ADR is for a decision that is significant **and** not easily reversible. Minor or cheap-to-undo
choices are recorded inline in the asset they affect. Every ADR names the conditions that would
reverse it, so a future reader can tell whether those conditions have arrived.
