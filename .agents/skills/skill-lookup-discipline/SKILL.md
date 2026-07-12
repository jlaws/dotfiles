---
name: skill-lookup-discipline
description: "Use when starting any task — ensures applicable skills are checked before writing code, running commands, or making decisions. Do NOT use as a substitute for reading specific skills."
---

# Skill-Lookup Discipline

**Core principle:** Check for applicable skills BEFORE any implementation action.

**Violating the letter of this rule is violating the spirit of this rule.**

## The Rule

Invoke relevant skills BEFORE any response or action. Even a 1% chance a skill might apply means you should check. If the skill turns out to be wrong for the situation, you don't need to follow it.

```
BEFORE acting on any task:
1. IDENTIFY: What kind of work is this? (feature, bug, refactor, review, plan, etc.)
2. CHECK: Scan .agents/skills/ and .agents/references/ for applicable patterns
3. PRUNE: Keep only the smallest set of skills that covers the task — loading extras dilutes attention
4. LOAD: Invoke matching skills via the Skill tool
5. THEN: Proceed with the task following loaded skill guidance
```

## Skill Priority

When multiple skills could apply, use this order:

1. **Process skills first** (design-first, debugging, verification) — these determine HOW to approach the task
2. **Implementation skills second** (code-quality, code-review-patterns, TDD) — these guide execution

Examples:
- "Build feature X" → design-first skill first, then implementation skills
- "Fix this bug" → debugging skill first, then domain-specific references
- "Review this PR" → code-review-patterns first

## Announce the Skill

When you invoke a skill, say so before acting: **`Using [skill] to [purpose].`** This makes the workflow visible and surfaces a wrong-skill choice early, before you've acted on it.

If the skill contains a checklist or numbered process, create matching todos so each step is tracked to completion.

## Skill Types

**Rigid** (TDD, verification-before-completion, debugging): Follow exactly. Don't adapt away discipline.

**Flexible** (code-quality, refactoring-and-debt): Adapt principles to context.

The skill itself tells you which type it is.

## Red Flags — STOP

These thoughts mean you're rationalizing skipping a skill check:

| Thought | Reality |
|---------|---------|
| "This is just a simple question" | Questions can be tasks. Check for skills. |
| "I need more context first" | Skill check comes BEFORE exploration. |
| "Let me explore the codebase first" | Skills tell you HOW to explore. Check first. |
| "I can handle this quickly" | Quick tasks are where skipped skills cause the most rework. |
| "This doesn't need a formal skill" | If a skill exists, use it. |
| "I remember this skill" | Skills evolve. Read current version. |
| "The skill is overkill" | Simple things become complex. Use it. |
| "I'll just do this one thing first" | Check BEFORE doing anything. |
| "I know what that means" | Knowing the concept ≠ following the workflow. |

## User Instructions

Instructions say WHAT, not HOW. "Add X" or "Fix Y" doesn't mean skip workflows. The user expects discipline.

## When NOT to Check

- Pure conversational responses (greetings, explanations with no action)
- Reading files the user explicitly asked you to read
- Git status / log / diff commands (informational only)

## Integration

This skill is auto-triggered by the session-start hook. It applies to every session.
