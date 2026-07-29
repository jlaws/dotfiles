---
name: writing-skills
description: "Use when creating, editing, or validating agent skills."
allowed-tools: Read, Grep, Glob, Bash, Write, Edit
---

# Writing Skills

A skill exists to teach what a strong model does not already know. Everything else crowds out the
task it was loaded to help with.

**Codex and Gemini read `~/.agents/skills`; Claude reads `~/.claude/skills`.** The trees have
diverged on purpose: `.claude/` is written for the Claude 5 generation, `.agents/` serves the other
tools. `tests/test_agent_config.py` pins which skills exist in each, not what they say.

## What earns a place in a skill

Keep a section only if it passes one of these:

- The model would get it wrong or vague from memory — version-gated behavior, a non-obvious default
- It is a number or threshold with provenance, not a guess
- It is an opinionated choice that closes a decision the model would otherwise re-litigate
- It is specific to this repo: a real path, a real gotcha, a real convention

Cut what fails all four. Generic best practice, framework tours, and derivable taxonomies are the
usual offenders. A skill that could apply to any repo teaches little about this one.

## The four shifts

| Instead of | Write |
|---|---|
| A rule | The goal and the reason, so judgment covers cases you did not foresee |
| Examples enumerating usage | An interface whose shape implies correct use |
| Everything upfront | A pointer, with the detail in a file that loads when needed |
| The same rule on three surfaces | One statement, in the place that owns it |

Rigid prohibitions are usually wrong for some real situation, and the model cannot notice that if the
skill also forbids noticing. Say what you want and why it matters.

## When to create a skill

**Create when** the technique was not obvious, it is reusable, and someone else would benefit.

**Do not create for** one-off solutions, well-documented standard practice, project conventions
(those go in CLAUDE.md), or anything a script can enforce — automate those instead.

## Structure

```
skills/
  skill-name/
    SKILL.md              # required
    scripts/              # mechanical steps; call these instead of describing them
    references/           # heavy detail, loaded on demand
```

Split a file out when it is heavy reference or a reusable tool. Keep the rest inline, one level deep.

**For mechanical steps — validation, formatting, deterministic checks — ship a script and call it.**
Reserve the model for judgment. A prose checklist of things a script could verify is both longer and
less reliable. List a script's dependencies; do not assume they are installed.

Where a choice exists, give one default with an escape hatch rather than a menu of equal options.

## Frontmatter

`name` (letters, numbers, hyphens) and `description` are the essentials. Others worth knowing:

| Field | Use |
|---|---|
| `allowed-tools` / `disallowed-tools` | Scope tool access for the turn |
| `model`, `effort` | Override tier or reasoning depth; tier aliases float across model generations |
| `paths` | Glob-gate the skill so it loads only for matching files |
| `disable-model-invocation` | Keep it out of the always-loaded listing; user-invocable only |
| `context: fork`, `agent` | Run in an isolated subagent context |
| `argument-hint`, `arguments` | Autocomplete and `$name` substitution |

`compatibility` is not a Claude Code field and does nothing.

`paths` and `disable-model-invocation` are the progressive-disclosure levers: they decide whether a
skill costs context on every turn or only when it is relevant.

**Description = triggering conditions only.** Never summarize the workflow there; testing showed
Claude follows a description shortcut instead of reading the body. Shared skills under `.agents/`
also carry a hard 64-character budget and must start with "Use when", enforced by `tests/`.

```yaml
# BAD: summarizes the workflow
description: Use when executing plans - executes tasks sequentially with review between tasks

# GOOD: just the trigger
description: Use when executing implementation plans with independent tasks
```

## Discoverability

- Name by what you do: `condition-based-waiting`, not `async-test-helpers`. Gerunds read well.
- Put searchable words in the description: symptoms, error text, tool names.
- Prefer concrete triggers over language-specific ones.

**Cross-references:** name the skill in prose (`use the code-review-patterns skill`). Never use `@`
links — they force-load and burn context.

## Flowcharts

Use one only for a non-obvious decision, a process loop, or "when to use A vs B". Never for reference
material, code examples, or linear instructions. Conventions and rendering: `graphviz-conventions.dot`
and `render-graphs.js` in this directory.

## Test it against a real agent

A skill you have not watched an agent use is a guess. Run the task without the skill and note where
the agent actually goes wrong; write the skill to address those specific failures; re-run.

If the agent read the skill and still chose wrong, ask it why. "The skill was clear, I ignored it"
means the principle is not doing any work. "It should have said X" is a direct edit. "I missed
section Y" is a structure problem. Detailed methodology: `references/CLAUDE_MD_TESTING.md`.

When wording shapes behavior, read it back cold a few times. If it supports several readings, rewrite
until they converge.

Resist answering every failure with a stronger prohibition. When an agent skips a step, the usual
cause is that the skill never said why the step mattered.

## Before you ship

- [ ] Every section passes one of the four tests above
- [ ] Mechanical steps live in a script, not in prose
- [ ] `description` is a trigger, and any `.agents/` copy fits the 64-character budget
- [ ] Supporting files are referenced from SKILL.md
- [ ] KB self-docs current: CLAUDE.md Knowledge Base Structure, and MEMORY.md if the asset set changed
- [ ] `cmd-j-*` skills have Claude, Codex, and Gemini command counterparts
- [ ] `make test` and `make check` pass
