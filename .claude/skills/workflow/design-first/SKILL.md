---
name: design-first
description: "Design-before-implementation gate for features and components. Use when creating features, building components, adding functionality, or modifying behavior — explores intent, requirements, and design before writing code. Do NOT use for bug fixes (use debugging), refactoring (use refactoring-and-debt), or pure research tasks."
compatibility: claude-code
allowed-tools: Read, Grep, Glob, Bash
---

# Design-First

**Core principle:** Understand what you're building before building it.

## Hard Gate

```
DO NOT write any code, scaffold any project, or take any implementation
action until you have presented a design and the user has approved it.
This applies to EVERY feature regardless of perceived simplicity.
```

## Anti-Pattern: "This Is Too Simple To Need A Design"

Every feature goes through this process. A utility function, a config change, a single component — all of them. "Simple" projects are where unexamined assumptions cause the most wasted work. The design can be short (a few sentences for truly simple work), but you MUST present it and get approval.

## The Process

### Phase 1: Explore Context

- Check existing code, docs, recent commits
- Understand the current architecture and patterns
- Identify constraints and dependencies

### Phase 2: Ask Clarifying Questions

- One question at a time — don't overwhelm
- Prefer multiple-choice questions when possible
- Focus on: purpose, constraints, success criteria, edge cases
- Keep asking until you understand what you're building

### Phase 3: Propose Approaches

- Present 2-3 different approaches with trade-offs
- Lead with your recommendation and explain why
- Be honest about complexity and risk of each option
- Apply YAGNI ruthlessly — remove unnecessary features from all proposals

### Phase 4: Present Design

- Scale each section to its complexity (a few sentences if straightforward, more if nuanced)
- Ask after each major section whether it looks right so far
- Cover as applicable: architecture, components, data flow, error handling, testing approach
- Be ready to revise if something doesn't fit

### Phase 5: Document

- Save validated design to `docs/plans/YYYY-MM-DD-<topic>-design.md`
- Commit the design document
- Skip file output only if user explicitly says no

### Phase 6: Transition to Implementation

- Hand off to `workflow/writing-plans` skill for implementation planning
- Or proceed directly if the task is small enough for inline implementation

## Process Flow

```
Explore context → Ask questions (one at a time) → Propose 2-3 approaches
→ Present design → User approves? (no → revise, yes → continue)
→ Write design doc → Transition to implementation
```

## Key Principles

- **One question at a time** — don't overwhelm with multiple questions
- **Multiple choice preferred** — easier to answer than open-ended when possible
- **YAGNI ruthlessly** — remove unnecessary features from all designs
- **Explore alternatives** — always propose 2-3 approaches before settling
- **Incremental validation** — present design, get approval before moving on
- **Be flexible** — go back and clarify when something doesn't make sense

## Red Flags

- Writing code before design approval
- Skipping the question phase ("I know what they want")
- Presenting only one approach
- Over-designing before asking clarifying questions
- Assuming requirements instead of asking
