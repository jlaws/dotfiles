---
name: design-first
description: "Use when designing features or behavior before coding."
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
- Each question carries your **recommended answer** — let the user confirm, not explain from scratch
- **Research before asking** — if the codebase can answer it (Glob/Grep/Read), find out first; only ask what the code can't tell you
- Order questions by dependency — resolve the ones that unlock others first
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
- **Self-review before committing** — scan the written design for placeholders, internal contradictions, and scope creep (features beyond what was agreed). Fix or flag anything you find.
- Commit the design document
- Skip file output only if user explicitly says no
- Design docs are not user docs — when the feature ships, keep README/API/usage docs current too (see `documentation-validation`)

### Phase 6: Transition to Implementation

- Hand off to `writing-plans` skill for implementation planning
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

## Design Size Scale

Scale the process to the task's complexity:

| Task Size | Questions | Approaches | Design Document |
|-----------|-----------|------------|-----------------|
| Tiny (utility function, config) | 1-2 quick questions | 1-2 sentences each | Skip — inline approval is fine |
| Small (single component, endpoint) | 2-3 questions | 2-3 brief approaches | Optional — user's choice |
| Medium (feature, multi-file change) | 3-5 questions | 2-3 detailed approaches with trade-offs | Yes — save to docs/plans/ |
| Large (system, architecture change) | Iterative questioning | Detailed approaches with diagrams | Yes — save and commit before proceeding |

The process is the same regardless of size — the depth scales, not the steps. Even tiny tasks get clarifying questions.

## Red Flags

- Writing code before design approval
- Skipping the question phase ("I know what they want")
- Presenting only one approach
- Over-designing before asking clarifying questions
- Assuming requirements instead of asking
- Jumping to implementation details before understanding the "what" and "why"
