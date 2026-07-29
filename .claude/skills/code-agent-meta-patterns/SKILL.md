---
name: code-agent-meta-patterns
description: "Use when designing agent instructions, hooks, or workflows."
allowed-tools: Read, Grep, Glob, Bash, Write, Edit
---

# Claude Code Meta-Patterns

Mechanics of the harness that are easy to get wrong, and the conventions this repo settled on.

## What belongs in CLAUDE.md

| Include | Leave out |
|---------|-----------|
| Project conventions a reader could not guess | Anything a strong model already knows |
| Build, test, and lint commands | API documentation — link to it |
| Architecture decisions and their rationale | Content already in `package.json` or `pyproject.toml` |
| A file and directory purpose map | Copy-pasted README |
| Non-obvious constraints and real footguns | Instructions that change week to week |

CLAUDE.md earns its place by holding what the model cannot infer from the repo. Everything else
competes with the task for attention.

## Layering

CLAUDE.md files cascade, and higher specificity wins:

```
~/.claude/CLAUDE.md              # global: style, git conventions, universal workflow
project/CLAUDE.md                # repo overview
project/.claude/CLAUDE.md        # repo behavioral rules
project/src/api/CLAUDE.md        # module-specific conventions
```

Putting project-specific rules in the global file applies them to every project. Keep global truly
global.

**Volatile content breaks the prompt cache.** Timestamps, changing counts, and rotating metrics in
CLAUDE.md invalidate the cached prefix on every session. Keep it stable.

## Progressive disclosure is the main lever

Everything in CLAUDE.md and every asset `description` is loaded on every turn. Bodies are not. Where a
detail belongs depends entirely on that:

| Surface | Cost |
|---|---|
| CLAUDE.md | Every turn |
| Skill, command, and agent `description` | Every turn |
| Skill and command bodies | Only when invoked |
| `references/` | Only when a body names one and an agent reads it |

Skill frontmatter offers two more levers: `paths` glob-gates a skill to matching files, and
`disable-model-invocation` keeps it out of the always-loaded listing entirely.

## Command design

Commands route; skills hold the logic. A command gathers context, then invokes a skill or delegates to
an agent. When a command starts explaining *how* to do the thing, that content belongs in a skill —
otherwise it is duplicated wherever else the skill applies.

`$ARGUMENTS` passes user context through. Declare `argument-hint` whenever a command reads it, so
autocomplete tells the user what to type.

Skill authoring conventions — frontmatter, descriptions, structure, testing — live in the
`writing-skills` skill rather than here.

## Hooks

See `references/workflow/hook-patterns.md` for PreToolUse and PostToolUse JSON examples.

Hooks run on the critical path, so a slow one makes every affected tool call painful. Keep them fast,
scope matchers narrowly, fail loudly rather than silently, and test manually before wiring them in.

## Permissions

See `references/workflow/permission-management.md` for the settings hierarchy and JSON examples. Deny
overrides allow; `settings.local.json` holds personal overrides that should not be committed.

## Delegation

Delegate work that is genuinely independent and large enough to be worth a separate context — a wide
multi-file investigation, several unrelated tracks that can run at once. Work you could finish in a few
tool calls costs more to delegate than to do.

Use `run_in_background` for long builds and test runs, then collect results when notified.

Further patterns: `references/workflow/context-efficiency`.

## Gotchas

- **Stale instructions actively mislead.** A CLAUDE.md naming deleted files or retired conventions is
  worse than one that says nothing. Prune when the underlying thing changes.
- **Description mismatch misroutes.** If a description no longer matches the body, the asset gets
  invoked at the wrong time or skipped when it was needed.
- **Overlapping descriptions make invocation unpredictable.** Two assets with similar triggers compete.
  Deduplicate, or make the triggers clearly distinct.
- **One occurrence is not a pattern.** If you have needed something once, it was a conversation, not a
  skill.
