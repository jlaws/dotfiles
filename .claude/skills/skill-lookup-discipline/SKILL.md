---
name: skill-lookup-discipline
description: "Use when starting a task that may match an existing skill."
allowed-tools: Read, Grep, Glob
---

# Skill-Lookup Discipline

Check for an applicable skill before acting, not after. A skill exists because someone already hit the
problem and wrote down what worked — finding it afterward means redoing that work, and the rework costs
far more than the check.

Check even when a match seems unlikely. Reading a skill and deciding it does not fit is cheap; the
expensive case is discovering halfway through that one applied.

## The check

1. **Identify** the kind of work: feature, bug, refactor, review, plan.
2. **Scan** `.claude/skills/` and `.claude/references/` for applicable patterns.
3. **Prune** to the smallest set that covers the task. Loading extras dilutes attention, which costs
   quality and not just tokens.
4. **Load** the matches via the Skill tool.
5. **Proceed**, following what they say.

The check comes before exploration, because skills often tell you *how* to explore. "Let me look at the
codebase first" inverts the order and usually means exploring in a way a skill would have improved.

## Which first

Process skills before implementation skills. Process skills decide how to approach the work, so they
change what the implementation skills are applied to:

- "Build feature X" → `design-first`, then the implementation skills
- "Fix this bug" → `debugging-methodology`, then domain references
- "Review this PR" → `code-review-patterns`

## Announce it

Say which skill and why before acting: **`Using [skill] to [purpose].`** This surfaces a wrong choice
before you have built on it. If a skill carries a numbered process, track its steps as todos so none
gets dropped.

## How closely to follow a skill

Some skills encode a discipline where the sequence is the point — TDD's write-the-test-first,
`debugging-methodology`'s find-the-cause-before-fixing. Departing from the sequence discards the thing
that makes it work, so if you think you should, say why rather than doing it quietly.

Others are principles to fit to the situation, like `code-quality` or `refactoring-and-debt`. Apply the
reasoning, not the letter.

An instruction that says *what* to do ("add X", "fix Y") is not an instruction to skip *how*.

## When to skip the check

Conversational replies with no action, reading a file you were asked to read, and informational git
commands (`status`, `log`, `diff`).

## Integration

The session-start hook triggers this skill, so it applies to every session.
