---
name: writing-skills
description: "Use when creating, editing, or validating agent skills."
---

# Writing Skills

**Writing skills IS Test-Driven Development applied to process documentation.**

**Codex uses `~/.agents/skills`; Gemini/Antigravity uses `~/.gemini/antigravity-cli/skills`; Claude uses `~/.claude/skills`.**

You write test cases (pressure scenarios), watch them fail (baseline behavior), write the skill (documentation), watch tests pass (agents comply), and refactor (close loopholes).

**Core principle:** If you didn't watch an agent fail without the skill, you don't know if the skill teaches the right thing.

## When to Create a Skill

**Create when:** Technique wasn't intuitively obvious, reusable across projects, pattern applies broadly, others would benefit.

**Don't create for:** One-off solutions, standard well-documented practices, project-specific conventions (use CLAUDE.md), mechanically enforceable constraints (automate instead).

## Skill Types

- **Technique**: Concrete method with steps (condition-based-waiting)
- **Pattern**: Way of thinking about problems (flatten-with-flags)
- **Reference**: API docs, syntax guides, tool documentation

## Directory Structure

```
skills/
  skill-name/
    SKILL.md              # Main reference (required)
    supporting-file.*     # Only if needed
```

Separate files for: heavy reference (100+ lines), reusable tools. Keep everything else inline.

## SKILL.md Structure

**Frontmatter:** Only `name` (letters/numbers/hyphens) and `description` (at most 64 characters, third-person, starts with "Use when...")

**CRITICAL:** Description = triggering conditions ONLY. Never summarize the skill's workflow in description. Testing showed Claude follows description shortcuts instead of reading skill body.

```yaml
# BAD: Summarizes workflow
description: Use when executing plans - executes tasks sequentially with code review between tasks

# GOOD: Just triggers
description: Use when executing implementation plans with independent tasks in the current session
```

**Body structure:**
```markdown
# Skill Name
## Overview (1-2 sentences, core principle)
## When to Use (flowchart IF decision non-obvious, bullets with symptoms)
## Core Pattern (before/after code)
## Quick Reference (table/bullets)
## Common Mistakes (what goes wrong + fixes)
```

## Keyword Discipline (RFC 2119)

Use MUST / MUST NOT / SHOULD / MAY with their RFC 2119 meanings so a hard gate reads differently from a suggestion. Reserve MUST/MUST NOT for non-negotiable rules, SHOULD for strong defaults with an escape hatch, MAY for options. Letting "should" creep into a hard gate is how discipline erodes.

## Claims and Evidence

A number in a skill is a claim the reader will act on, so every quantitative claim carries how it was measured or is marked unmeasured. Distinguish a claim from a target: "min 80% coverage" is a gate and needs no provenance, while "cuts tokens 2-3x" is a claim and does. Numbers from a third party are that party's claim, not yours; attribute them or leave them out.

| Instead of | Write |
|---|---|
| "Tables are ~40% more efficient than prose" | "Tables are denser than prose (unmeasured here; the ratio depends on the content)" |
| "This cuts review time in half" | "Measured on <what>, <date>: <before> -> <after>" |

Say when a practice loses, not only when it wins. A rule with no stated cost reads as free, and the reader stops looking for the cost. Where a technique has fixed overhead, such as a skill body preloaded on every dispatch or the cost of dispatching a subagent, say at what size the overhead exceeds the benefit.

**A counterfactual is not a measurement.** "This saved N tokens" about work you did is unknowable -- the unwritten version was never written, so there is no baseline to subtract from. Either run a holdout (leave a fraction unshaped and compare) or label the figure an estimate and give its range. Naming a saving you cannot have observed is the most common way a real technique acquires a fake number.

**Link the counter-evidence.** Where someone has published a measurement against a claim you are making, cite it and say which part survives your fix and which part stands. A claim with the strongest objection beside it is more trustworthy than one without, and a reader who finds the objection elsewhere first will discount everything around it.

**Retract in place.** A figure that stops reproducing gets struck where it was published, with what replaced it -- not quietly deleted. Deleting it hides that it was ever believed, which is what a later reader most needs to know.

**A feature is a claim too.** Behavior shipped on a hunch carries the same burden as a number: either evidence that it fires, or the label. Say it is speculative and say what would retire the label. The opposite failure is quieter -- a rule the repo advertises but nothing enforces.

**Cherry-picking is only dishonest when it is silent.** A best-case example fairly answers "what does this look like when it works", provided the general number sits next to it.

**If you do build a measurement, two things make it defensible.** Include cases where the technique should *not* win, and fail loudly if it starts claiming a win there -- a suite of only favourable inputs proves the fixtures were chosen well, not that the technique works. And compare against a length-matched naive control: for a compressor, plain truncation to the same output size. Beating nothing is easy; beating the dumbest thing that produces the same volume is the real question.

## Co-located Scripts

For mechanical steps (validation, formatting, deterministic checks), ship a `scripts/` file in the skill directory and call it, rather than describing the steps in prose. Reserve the LLM for judgment; let a script do what a script does better. Keep scripts one level deep and list any dependencies.

## Claude Search Optimization (CSO)

- Use concrete triggers/symptoms in description, not language-specific symptoms
- Include keywords Claude would search: error messages, symptoms, tool names
- Name by what you DO: `condition-based-waiting` not `async-test-helpers`
- Gerunds work well: `creating-skills`, `debugging-with-logs`

**Token Efficiency:**
- Getting-started workflows: <150 words
- Frequently-loaded skills: <200 words
- Other skills: <500 words
- Move details to `--help`, use cross-references, compress examples

**Cross-References:** Name the skill in prose (`use the code-review-patterns skill`). Never use `@` links (force-loads, burns context).

## Flowchart Usage

Use ONLY for: non-obvious decisions, process loops, "when to use A vs B".
Never for: reference material, code examples, linear instructions.

Conventions and rendering: see `graphviz-conventions.dot` and `render-graphs.js` in this skill directory.

## The Iron Law (Same as TDD)

```
NO SKILL WITHOUT A FAILING TEST FIRST
```

Applies to new skills AND edits. Write skill before testing? Delete it. Start over. No exceptions.

## RED-GREEN-REFACTOR for Skills

**RED:** Run pressure scenario WITHOUT skill. Document exact failures and rationalizations.

**GREEN:** Write minimal skill addressing those specific failures. Re-test with skill.

**REFACTOR:** Agent found new rationalization? Add explicit counter. Re-test until bulletproof.

## Testing Skills

For detailed testing methodology, see `references/CLAUDE_MD_TESTING.md`.

Run scenarios without skill (RED), write skill addressing failures (GREEN), close loopholes (REFACTOR).

### RED Phase: Baseline Testing

Run pressure scenario WITHOUT skill. Document exact failures.

**Process:**
- [ ] Create pressure scenarios (3+ combined pressures)
- [ ] Run WITHOUT skill
- [ ] Document choices and rationalizations verbatim
- [ ] Identify patterns

### Pressure Types

| Pressure | Example |
|----------|---------|
| Time | Emergency, deadline, deploy window |
| Sunk cost | Hours of work, "waste" to delete |
| Authority | Senior says skip it |
| Exhaustion | End of day, want to go home |
| Pragmatic | "Being pragmatic vs dogmatic" |

**Best tests combine 3+ pressures.**

### Key Elements
1. Concrete A/B/C choices (not open-ended)
2. Real constraints (specific times, consequences)
3. Real file paths
4. Make agent act ("What do you do?" not "What should you do?")

### REFACTOR Phase: Close Loopholes

Capture new rationalizations verbatim. For each, add:
1. **Explicit negation** in rules
2. **Rationalization table** entry
3. **Red flag** entry
4. **Updated description** with violation symptoms

Re-test after each refactor. Continue until no new rationalizations.

### Meta-Testing

After agent chooses wrong: "You read the skill and chose wrong. How could it be clearer?"

Three responses:
1. "Skill WAS clear, I chose to ignore" -> Need stronger foundational principle
2. "Skill should have said X" -> Add their suggestion
3. "I didn't see section Y" -> Make it more prominent

### Interpretation Test

Before shipping behavior-shaping wording, run the same prompt about 5 times (or across 5 fresh contexts). Five different interpretations means the wording is ambiguous — rewrite until the readings converge.

## Persuasion Principles for Skill Design

LLMs respond to the same persuasion principles as humans. Meincke et al. (2025) tested 7 principles with N=28,000 AI conversations. Persuasion techniques doubled compliance rates (33% -> 72%, p < .001).

### Effective Principles

- **Authority**: Imperative language ("YOU MUST", "Never", "Always"), non-negotiable framing. For discipline-enforcing skills.
- **Commitment**: Require announcements, force explicit choices. For ensuring skills are followed.
- **Scarcity**: Time-bound requirements ("Before proceeding"), sequential dependencies. For immediate verification.
- **Social Proof**: Universal patterns ("Every time", "Always"), failure modes. For documenting universal practices.
- **Unity**: Collaborative language ("our codebase", "we're colleagues"). For collaborative workflows.
- **Reciprocity & Liking**: Use sparingly or avoid.

| Skill Type | Use | Avoid |
|------------|-----|-------|
| Discipline-enforcing | Authority + Commitment + Social Proof | Liking, Reciprocity |
| Guidance/technique | Moderate Authority + Unity | Heavy authority |
| Collaborative | Unity + Commitment | Authority, Liking |
| Reference | Clarity only | All persuasion |

## Anthropic Best Practices

### Skill Categories (from Anthropic's Guide)

| Category | Description | Examples |
|----------|-------------|---------|
| 1. Document & Asset Creation | Generate files from templates/specs | Commit messages, PR descriptions, config files |
| 2. Workflow Automation | Multi-step processes with tool use | Code review, deployment, refactoring |
| 3. MCP Enhancement | Extend Claude with external tool integrations | API wrappers, database queries, service connectors |

### Success Criteria Methodology
1. Define what "good output" looks like before writing the skill
2. Create 3+ concrete evaluation scenarios with expected outcomes
3. Test against scenarios, measure pass rate
4. Iterate until pass rate meets threshold (aim for >80%)

### Core Principles
- **Concise is key**: Only add what Claude doesn't already know. Challenge each piece: "Does Claude need this?"
- **Degrees of freedom**: High (text instructions) for multiple valid approaches; Medium (pseudocode) for preferred patterns; Low (exact scripts) for fragile operations

### Anti-Patterns
- Offering too many library options (provide a default with escape hatch)
- Assuming packages installed (list dependencies)
- Deeply nested references (keep one level deep)
- Time-sensitive information (use "old patterns" section)

### Evaluation-Driven Development
1. Run Claude on tasks without Skill, document failures
2. Create 3+ evaluation scenarios
3. Establish baseline performance
4. Write minimal instructions to pass evaluations
5. Iterate: evaluate, compare baseline, refine

### Quick Checklist (from Anthropic's Reference A)
- [ ] Frontmatter: `name`, `description` (with trigger phrases), `allowed-tools`
- [ ] Description starts with "Use when..." and includes negative triggers
- [ ] Body is concise (under 500 words for most skills)
- [ ] At least one code example or concrete output
- [ ] No redundant information Claude already knows
- [ ] Cross-references use relative paths, not `@` links

## Skill Creation Checklist

**RED Phase:**
- [ ] Create pressure scenarios (3+ combined pressures for discipline skills)
- [ ] Run WITHOUT skill - document baseline failures verbatim
- [ ] Identify rationalization patterns

**GREEN Phase:**
- [ ] YAML frontmatter: name (letters/numbers/hyphens), description (Use when..., third-person)
- [ ] Address specific baseline failures
- [ ] Keywords throughout for search
- [ ] One excellent code example (not multi-language)
- [ ] Run WITH skill - verify compliance

**REFACTOR Phase:**
- [ ] Add counters for new rationalizations
- [ ] Build rationalization table and red flags list
- [ ] Re-test until bulletproof

**Deployment:**
- [ ] Update KB self-docs: `description` frontmatter and cross-references current; CLAUDE.md KB-structure section and MEMORY.md index (if present) updated if the asset set changed (see `documentation-validation`)
- [ ] Sync shared workflow bodies to `.claude`; ensure `cmd-j-*` skills have Claude, Codex, and Gemini command counterparts
- [ ] Commit and push
