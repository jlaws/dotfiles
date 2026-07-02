---
name: j-new
description: "Scaffold a new .claude/ command, skill, or agent with correct structure and best practices. Use when creating any new .claude/ asset. Do NOT use for editing existing assets (edit directly)."
argument-hint: "<type: command|skill|agent> [name]"
---

Load skill `writing-skills` for asset-authoring methodology and `skill-audit` for the conformance rules applied in Phase 5, before scaffolding.

Create: $ARGUMENTS

If no arguments provided, ask which asset type to create (command, skill, or agent).

---

## Phase 1: Parse Arguments

Extract from `$ARGUMENTS`:
- **Type**: first word must be `command`, `skill`, or `agent`
- **Name**: optional second word (kebab-case)

If type is missing or invalid, ask the user.

### Name Validation

If name provided, verify:
- Lowercase letters, numbers, hyphens only
- Max 64 characters
- Does NOT contain "claude" or "anthropic" (reserved)
- No leading/trailing hyphens, no double hyphens

If name fails validation, explain why and ask for a corrected name.

## Phase 2: Gather Requirements

### 2A. Common (all types)

1. **Name** (if not provided in args)
2. **One-line purpose**: What does this asset do? (used to craft description)

### 2B. Type-Specific

**For commands:**
1. **Category**: List existing categories via `Glob(".claude/commands/*/")` and let user pick or create new
2. **Takes arguments?** If yes, ask for argument-hint text
3. **Invokes skills?** If yes, which ones? List available via `Glob(".claude/skills/*/")` and `Glob(".claude/skills/*/*/SKILL.md")`
4. **Invokes agents?** If yes, which ones? List available via `Glob(".claude/agents/*.md")`

**For skills:**
1. **Category**: List existing categories via `Glob(".claude/skills/*/")` and let user pick or create new
2. **Skill type**: Technique (concrete steps), Pattern (mental model), or Reference (API/syntax docs)
3. **Allowed tools**: Which tools should this skill use without per-use approval? Options: Read, Grep, Glob, Bash, Write, Edit, WebFetch, WebSearch
4. **Dependencies**: Other skills this loads? List available skills
5. **Auto-invocable?** Should Claude invoke this automatically based on description match, or manual-only (`disable-model-invocation: true`)?

**For agents:**
1. **Tools**: Which tools should this agent access? Options: Read, Grep, Glob, Bash, Write, Edit, WebFetch, WebSearch
2. **Skills to preload**: Which skills should be injected into agent context? List available
3. **Reference library**: Which `.claude/references/` paths should the agent consult? List available categories via `Glob(".claude/references/*/")` 
4. **Model override?** Default (inherit), sonnet, opus, haiku
5. **Persistent memory?** None, user, project, or local scope

## Phase 3: Generate Description

Craft description following the WHAT + WHEN + DO NOT pattern:

```
"{What it does} -- {brief qualifier}. Use when {trigger conditions}. Do NOT use for {anti-patterns} ({alternative} instead)."
```

### Description Rules

| Rule | Constraint |
|------|-----------|
| Max length (skills) | 250 chars (truncated in skill listing beyond this) |
| Max length (commands/agents) | 1024 chars |
| No XML characters | No `<` or `>` in frontmatter values |
| Trigger-only for skills | NEVER summarize the workflow in the description |
| Include negative triggers | "Do NOT use for..." with redirect to alternative |

**Bad skill description** (summarizes workflow):
```
description: "Use when executing plans -- executes tasks sequentially with code review between tasks"
```

**Good skill description** (triggers only):
```
description: "Use when executing implementation plans with independent tasks in the current session"
```

Present the draft description to the user for approval before proceeding.

## Phase 4: Scaffold the Asset

Generate the file using the correct template for the asset type.

### 4A. Command Template

**File path:** `.claude/commands/{category}/{name}.md`

```markdown
---
name: {name}
description: "{generated description}"
argument-hint: "{hint}"  # only if takes arguments
---

{Purpose context}: $ARGUMENTS  # only if takes arguments

If no arguments provided, {fallback behavior}.

---

## Phase 1: {First Phase Name}

{Instructions for gathering context, understanding the request}

## Phase 2: {Second Phase Name}

{Core methodology -- the main work}

## Phase 3: {Third Phase Name}

{Verification, output, or handoff}

---

### Cross-References

- **skill:{skill-path}** -- {why referenced}
- **agent:{agent-name}** -- {why referenced}
```

### 4B. Skill Template

**File path:** `.claude/skills/{category}/{name}/SKILL.md`

Create the directory first, then write SKILL.md:

```markdown
---
name: {name}
description: "{generated description}"
compatibility: claude-code
allowed-tools: {tools list}
skills:  # only if has dependencies
  - {dependency-path}
---

# {Skill Title}

## Overview

{One-two sentence core principle.}

## When to Use

- {Symptom or trigger condition 1}
- {Symptom or trigger condition 2}
- {Symptom or trigger condition 3}

## Core Pattern

{The main technique, method, or pattern with concrete steps.}

```{language}
# Example demonstrating the pattern
```

## Quick Reference

| {Column 1} | {Column 2} |
|-------------|-------------|
| {Key point} | {Detail} |

## Common Mistakes

- **{Mistake 1}**: {What goes wrong} -- {Fix}
- **{Mistake 2}**: {What goes wrong} -- {Fix}
```

For **workflow/procedural skills**, also add:

```markdown
## Red Flags

- {Warning sign that the skill is being bypassed}

## The Iron Law

```
{NON-NEGOTIABLE RULE IN CAPS}
```
```

### 4C. Agent Template

**File path:** `.claude/agents/{name}.md`

```markdown
---
name: {name}
description: "{generated description}"
tools: {tools list}
skills:
  - {skill-name}
  - verification-before-completion
model: {model}  # only if overridden
memory: {scope}  # only if enabled
---

You are a {role description}. {One sentence about approach/expertise.}

Reference library at .claude/references/{category}/:
- {reference-1}, {reference-2}, {reference-3}

Read the relevant reference file(s) for the user's topic before responding.
{Additional role-specific instructions.}
```

## Phase 5: Inline Validation

Before writing, validate the generated content against skill-audit rules:

### All Types
- [ ] Frontmatter has `---` delimiters
- [ ] `name` field exists and is kebab-case
- [ ] `name` matches filename (commands/agents) or folder name (skills)
- [ ] `description` field exists and is under length limit
- [ ] Description has WHAT + WHEN trigger pattern
- [ ] No `<`/`>` in frontmatter values
- [ ] No "claude"/"anthropic" in name

### Commands Only
- [ ] File lives in a category subdirectory
- [ ] If body uses `$ARGUMENTS`, frontmatter has `argument-hint`
- [ ] If body invokes a skill, that skill exists
- [ ] If body invokes an agent, that agent exists
- [ ] Body has 10+ words

### Skills Only
- [ ] SKILL.md is the filename (exact casing)
- [ ] No README.md in the folder
- [ ] Body has 50+ words
- [ ] At least one code block or example
- [ ] Description is trigger-only (no workflow summary)

### Agents Only
- [ ] File is flat in agents/ (no subdirectory)
- [ ] `tools` field exists with valid tool names
- [ ] If `skills:` references skills, each resolves to existing SKILL.md
- [ ] Body has 20+ words

Report any WARN or FAIL findings. Auto-fix where possible (e.g., truncate description, fix casing).

## Phase 6: Present & Confirm

Show the complete generated file to the user. Ask for approval before writing.

After writing:
1. Confirm the file was created at the correct path
2. If skill: confirm the directory was created
3. Suggest: "Run `/j-skill-audit {type}s` to verify full conformance"

---

### Frontmatter Quick Reference

| Field | Command | Skill | Agent |
|-------|---------|-------|-------|
| name | Required | Required | Required |
| description | Required | Required | Required |
| argument-hint | If $ARGUMENTS | Optional | N/A |
| compatibility | N/A | "claude-code" | N/A |
| allowed-tools | N/A | Recommended | N/A |
| tools | N/A | N/A | Recommended |
| skills | N/A | Optional (deps) | Optional (preload) |
| model | N/A | Optional | Optional |
| memory | N/A | N/A | Optional (user/project/local) |
| disable-model-invocation | N/A | Optional | N/A |
| user-invocable | N/A | Optional | N/A |
| context | N/A | Optional (fork) | N/A |
| effort | N/A | Optional | Optional |
| color | N/A | N/A | Optional |
| maxTurns | N/A | N/A | Optional |
| permissionMode | N/A | N/A | Optional |

### Valid Tool Names

Read, Write, Edit, Grep, Glob, Bash, LSP, WebFetch, WebSearch, NotebookEdit, Skill

### Body Conventions

| Type | Convention |
|------|-----------|
| Command | `$ARGUMENTS` substitution, `---` divider, phased methodology, cross-references section |
| Skill | Overview, When to Use, Core Pattern (with code example), Quick Reference table, Common Mistakes |
| Agent | Role statement (1-2 sentences), reference library pointers, delegate methodology to skills |

### Cross-References

- **skill:writing-skills** -- skill authoring TDD methodology, CSO, persuasion principles
- **skill:skill-audit** -- validation rules for all asset types
