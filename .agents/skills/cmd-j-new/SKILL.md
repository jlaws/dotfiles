---
name: cmd-j-new
description: "Use when invoking the j-new workflow."
---

# Scaffold Asset

Invoke the `writing-skills` skill for asset-authoring methodology and the `skill-audit` skill for the conformance rules applied in Phase 5, before scaffolding.

Create: the user's provided input

If no arguments provided, ask which asset type to create (command, skill, or agent).

---

## Phase 1: Parse Arguments

Extract from the user's provided input:
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
1. **Takes input?** If yes, define the expected user input and fallback behavior
2. **Invokes skills?** If yes, list available `.agents/skills/*/SKILL.md`
3. **Invokes agents?** If yes, list available `.codex/agents/*.toml`

**For skills:**
1. **Skill type**: Technique (concrete steps), Pattern (mental model), or Reference (API/syntax docs)
2. **Dependencies**: Other skills this loads? List available skills
3. **Auto-invocable?** Should Codex invoke this from description matching, or only through an explicit `$` mention?

**For agents:**
1. **Skills to load**: Which shared workflows should the agent invoke?
2. **Reference library**: Which `~/.agents/references/` paths should the agent consult?
3. **Permissions**: Set the sandbox and approval policy on the parent session. The role inherits them.

Role definitions MUST omit `model` and `model_reasoning_effort`. Select model and effort for the assigned task at dispatch, following the caller's routing policy. A role describes expertise, not a model tier.

## Phase 3: Generate Description

Craft a trigger-only description:

```
"Use when {specific trigger conditions}."
```

### Description Rules

| Rule | Constraint |
|------|-----------|
| Max length (skills) | 64 characters |
| Max length (commands/agents) | 1024 chars |
| No XML characters | No `<` or `>` in frontmatter values |
| Trigger-only for skills | Describe only when to use it; a workflow summary in the description lets the model skip the body |
| Distinct trigger | Name the task or symptom that selects this skill |

**Bad skill description** (summarizes workflow):
```
description: "Use when executing plans - executes tasks sequentially with code review between tasks"
```

**Good skill description** (triggers only):
```
description: "Use when executing implementation plans with independent tasks in the current session"
```

Present the draft description to the user for approval before proceeding.

## Phase 4: Scaffold the Asset

Generate the file using the correct template for the asset type.

### 4A. Command Template

**File path:** `.agents/skills/cmd-j-{name}/SKILL.md`

Create the shared skill. Codex invokes it as `$cmd-j-{name}`. Do not create a Codex prompt fallback. Claude commands and Gemini command skills remain native counterparts.

```markdown
---
name: cmd-j-{name}
description: "Use when invoking the j-{name} workflow."
---

{Purpose context}: the user's provided input

If no arguments provided, {fallback behavior}.

---

## Phase 1: {First Phase Name}

{Instructions for gathering context, understanding the request}

## Phase 2: {Second Phase Name}

{Core methodology - the main work}

## Phase 3: {Third Phase Name}

{Verification, output, or handoff}

---

### Cross-References

- **skill:{skill-path}**: {why referenced}
- **agent:{agent-name}**: {why referenced}
```

For an explicit-only command, add `agents/openai.yaml` beside `SKILL.md` with:

```yaml
policy:
  allow_implicit_invocation: false
```

Keep this policy metadata when migrating commands. Command skills expose only `name` and `description` in their frontmatter.

### 4B. Skill Template

**File path:** `.agents/skills/{name}/SKILL.md`

Create the directory first, then write SKILL.md:

```markdown
---
name: {name}
description: "{generated description}"
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

## Gotchas

- **{Failure mode 1}**: {What goes wrong} - {Fix}
- **{Failure mode 2}**: {What goes wrong} - {Fix}
```

Do not add rationalization tables, "Red Flags: STOP" banners, or all-caps "Iron Law" blocks; state the goal and consequence instead (`writing-skills`, The Four Shifts).

### 4C. Agent Template

**File path:** `.codex/agents/{name}.toml`

```toml
name = "{name}"
description = "{generated description}"
developer_instructions = """
You are a {role description}. {One sentence about approach.}
Load these shared skills when relevant: {skill names}.
Read relevant files under `~/.agents/references/{category}/`.
{Additional role-specific instructions.}
"""
```

## Phase 5: Inline Validation

Before writing, validate the generated content against skill-audit rules:

### All Types
- [ ] Markdown frontmatter has `---` delimiters; TOML role definitions use TOML fields
- [ ] `name` field exists and is kebab-case
- [ ] `name` matches the skill folder or agent convention
- [ ] `description` field exists and is under length limit
- [ ] Description has WHAT + WHEN trigger pattern
- [ ] No `<`/`>` in frontmatter values
- [ ] No "claude"/"anthropic" in name

### Commands Only
- [ ] Shared skill lives at `.agents/skills/cmd-j-{name}/SKILL.md`
- [ ] No Codex prompt fallback is created
- [ ] Explicit-only commands carry `agents/openai.yaml` with `policy.allow_implicit_invocation: false`
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
- [ ] TOML file is flat in `.codex/agents/`
- [ ] `name`, `description`, and `developer_instructions` exist
- [ ] Any referenced skills or reference paths resolve
- [ ] `model` and `model_reasoning_effort` are absent so dispatch routes by task

Report any WARN or FAIL findings. Auto-fix where possible (e.g., truncate description, fix casing).

## Phase 6: Present & Confirm

Show the complete generated file to the user. Ask for approval before writing.

After writing:
1. Confirm the file was created at the correct path
2. If skill: confirm the directory was created
3. Suggest: "Run `$cmd-j-skill-audit {type}s` to verify full conformance"
4. Register and document the new asset. Commands require the Codex `cmd-j-*` skill, a native Claude command, and an Antigravity `j-*` command skill. Agents require native definitions in all three tool trees. Workflow skills ship in `.agents/skills/`, `.gemini/antigravity-cli/skills/`, and `.claude/skills/` with matching supporting files between `.agents/skills/` and `.gemini/antigravity-cli/skills/`; skill bodies and descriptions adapt per harness while shared reference twins expose the same ordered headings. See `documentation-validation`.
5. If the repository keeps an eval suite for its assets, add the new asset's cases (trigger and effect cases for a skill or command, effect cases for an agent). Offer to run its smoke target; it may bill.

---

### Frontmatter Quick Reference

| Field | Command | Skill | Agent |
|-------|---------|-------|-------|
| name | Required | Required | Required |
| description | Required | Required | Required |
| developer_instructions | N/A | N/A | Required |
| model | N/A | N/A | Omit; select per task at dispatch |
| `agents/openai.yaml` `policy.allow_implicit_invocation: false` | Shared command skill | Optional; set for a side-effect command nothing invokes | N/A |
| model_reasoning_effort | N/A | N/A | Omit; select per task at dispatch |

### Body Conventions

| Type | Convention |
|------|-----------|
| Command | Shared skill accepts user input and is the Codex command; no prompt fallback |
| Skill | Overview, When to Use, Core Pattern (with code example), Quick Reference table, Gotchas |
| Agent | Focused TOML `developer_instructions` with shared skill and reference pointers |

### Cross-References

- **skill:writing-skills**: authoring criteria, frontmatter spec, discoverability
- **skill:skill-audit**: validation rules for all asset types
