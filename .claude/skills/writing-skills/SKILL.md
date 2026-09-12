---
name: writing-skills
description: "Use when creating, editing, or validating agent skills."
allowed-tools: Read, Grep, Glob, Bash, Write, Edit
---

# Writing Skills

A skill exists to teach what a strong model does not already know. Everything else crowds out the
task it was loaded to help with.

**Codex reads `~/.agents/skills`, Gemini/Antigravity reads `~/.gemini/antigravity-cli/skills`, and Claude reads `~/.claude/skills`.** The trees have
diverged on purpose: `.claude/` is written for the Claude 5 generation, `.agents/` serves the other
tools. A parity test, where the repo has one, pins which skills exist in each, not what they say.

## What earns a place in a skill

Keep a section only if it passes one of these:

- The model would get it wrong or vague from memory — version-gated behavior, a non-obvious default
- It is a number or threshold with provenance, not a guess
- It is an opinionated choice that closes a decision the model would otherwise re-litigate
- It is specific to this repo: a real path, a real gotcha, a real convention

Cut what fails all four. Generic best practice, framework tours, and derivable taxonomies are the
usual offenders. A skill that could apply to any repo teaches little about this one.

## Claims and evidence

A number in a KB asset is a claim the reader will act on. Every quantitative claim carries how it was
measured, or is marked unmeasured.

| Instead of | Write |
|---|---|
| "Tables are ~40% more efficient than prose" | "Tables are denser than prose (unmeasured here; the ratio depends on the content)" |
| "This cuts review time in half" | "Measured on <what>, <date>: <before> -> <after>" |

Distinguish a claim from a target. "min 80% coverage" is a gate and needs no provenance; "cuts tokens
2-3x" is a claim and does.

Say when a practice loses, not only when it wins. A rule with no stated cost reads as free, and the
reader stops looking for the cost. Where a technique has fixed overhead — a skill body preloaded on
every dispatch, a subagent's dispatch cost — say at what size the overhead exceeds the benefit.

Numbers from a third party are that party's claim, not yours. Attribute them or leave them out.

**A counterfactual is not a measurement.** "This saved N tokens" about work you did is unknowable —
the unwritten version was never written, so there is no baseline to subtract from. Either run a
holdout (leave a fraction of the work unshaped and compare) or label the figure an estimate and give
its range. Naming a saving you cannot have observed is the most common way a real technique acquires
a fake number.

**Link the counter-evidence.** Where someone has published a measurement against a claim you are
making, cite it and say which part survives your fix and which part stands. A claim with the
strongest objection linked beside it is more trustworthy than one without, and a reader who finds
the objection elsewhere first will discount everything around it.

**Retract in place.** A figure that stops reproducing gets struck where it was published, with what
replaced it — not quietly deleted. Deleting it hides that it was ever believed, which is the thing a
later reader most needs to know.

**A feature is a claim too.** Behavior shipped on a hunch carries the same burden as a number:
either evidence that it fires, or the label. Say it is speculative and say what would retire the
label. The opposite failure is worse and quieter — a rule the repo advertises but nothing enforces.

**Cherry-picking is only dishonest when it is silent.** A best-case example is a fair answer to
"what does this look like when it works", provided the general number sits next to it.

**If you do build a measurement, two things make it defensible.** Include cases where the technique
should *not* win, and fail loudly if it starts claiming a win there — a suite of only favourable
inputs proves the fixtures were chosen well, not that the technique works. And compare against a
*length-matched naive control*: for a compressor, plain truncation to the same output size. Beating
nothing is easy; beating the dumbest thing that produces the same volume is the real question.

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
- [ ] The repo's lint and test commands pass
