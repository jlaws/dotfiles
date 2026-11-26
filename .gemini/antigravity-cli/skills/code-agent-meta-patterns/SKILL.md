---
name: code-agent-meta-patterns
description: "Use when designing agent instructions, hooks, or workflows."
---

# Coding Agent Meta-Patterns

Mechanics of agent harnesses (`CLAUDE.md`, `AGENTS.md`, `GEMINI.md`, skills, subagents, commands, and hooks) that are easy to get wrong, and the conventions this repo settled on.

## What Belongs in Harness Instructions

| Include | Leave out |
|---------|-----------|
| Project conventions a reader could not guess | Anything a strong model already knows |
| Build, test, and lint commands | API documentation (link to it) |
| Architecture decisions and their rationale | Content already in `package.json` or `pyproject.toml` |
| A file and directory purpose map | Copy-pasted README |
| Non-obvious constraints and real footguns | Instructions that change week to week |

Always-loaded instruction files (`CLAUDE.md`, `AGENTS.md`, `GEMINI.md`) earn their place by holding what the model cannot infer from the repo. Everything else competes with the task for attention.

In a project's instruction file, list each command with its quiet flag and the line a healthy run prints, so a run can be checked without reading its full output. Add compaction preservation notes naming what compaction must keep.

## Layering

Instruction files cascade, and higher specificity wins:

```
~/.claude/CLAUDE.md | ~/.codex/AGENTS.md | ~/.gemini/GEMINI.md   # global: style, git, workflow
project/CLAUDE.md | project/AGENTS.md | project/GEMINI.md        # repo overview & commands
project/src/api/CLAUDE.md | AGENTS.md                            # module-specific conventions
```

Putting project-specific rules in a global file applies them to every project. Keep global truly global.

**Volatile content breaks the prompt cache.** Timestamps, changing counts, and rotating metrics in always-loaded files invalidate the cached prefix on every session. Keep them stable.

## Progressive Disclosure Is the Main Lever

Everything in the root instruction file and every asset `description` is loaded on every turn. Bodies are not. Where a detail belongs depends on that:

| Surface | Cost |
|---|---|
| `CLAUDE.md` / `AGENTS.md` / `GEMINI.md` | Every turn |
| Skill, command, and agent `description` | Every turn |
| Skill and command bodies | Only when invoked |
| `references/` | Only when a body names one and an agent reads it |

Where a harness supports invocation gating (such as Claude `paths` / `disable-model-invocation` or Codex `agents/openai.yaml` `allow_implicit_invocation: false`), use it to keep user-only side-effect commands out of implicit model routing.

## Command Design

Commands route; skills hold the logic. A command gathers context, then invokes a skill or delegates to an agent. When a command starts explaining *how* to do the thing, that content belongs in a skill; otherwise it is duplicated wherever else the skill applies.

Skill authoring conventions (frontmatter, descriptions, structure, testing) live in `writing-skills`.

## Hooks

See `~/.agents/references/workflow/hook-patterns.md` for PreToolUse and PostToolUse configuration examples.

Hooks run on the critical path, so a slow one makes every affected tool call painful. Keep them fast, scope matchers narrowly, fail loudly rather than silently, and test manually before wiring them in.

## Agent Memory

Never enable persistent agent memory on an agent dispatched with a frozen, report-only packet; in this KB that is every specialist subagent. Durable lessons go into a versioned skill or reference instead.

## Permissions

See `~/.agents/references/workflow/permission-management.md` for the settings hierarchy and permission rule syntax across Claude Code, Codex, and Google Antigravity. Deny overrides allow; keep machine-local overrides out of version control.

## Delegation

Delegate work that is genuinely independent and large enough to be worth a separate context: a wide multi-file investigation, or several unrelated tracks at once. Work you could finish in a few tool calls costs more to delegate than to do.

Run long builds and test suites asynchronously when other independent work can proceed, then collect results on wakeup.

Further patterns: `~/.agents/references/workflow/context-efficiency.md`.

## Gotchas

- **Stale instructions actively mislead.** An instruction file naming deleted files or retired conventions is worse than one that says nothing. Prune when the underlying thing changes.
- **Description mismatch misroutes.** If a description no longer matches the body, the asset gets invoked at the wrong time or skipped when it was needed.
- **Overlapping descriptions make invocation unpredictable.** Two assets with similar triggers compete. Deduplicate, or make the triggers clearly distinct.
- **One occurrence is not a pattern.** If you have needed something once, it was a conversation, not a skill.
