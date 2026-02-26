---
description: "Audit skills for conformance to Anthropic's skill-building guide — checks naming, frontmatter, description quality, structure, and content best practices. Use when creating or modifying skills to validate compliance."
---

# Skill Audit

Audit all skills under `.claude/skills/` for conformance to Anthropic's skill-building standards.

## Scoping

Determine audit scope from `$ARGUMENTS`:
- **Empty** → audit ALL skills under `.claude/skills/`
- **Category name** (e.g., `security`) → audit only `.claude/skills/{category}/`
- **Specific skill path** (e.g., `security/auth-implementation-patterns`) → audit only that skill

## Phase 1: Discovery

1. Enumerate all skill directories under `.claude/skills/` (each subdirectory of a category that contains a `SKILL.md`)
2. For each skill, note:
   - Category (parent directory name)
   - Skill folder name
   - Whether `SKILL.md` exists
   - Any supporting files (templates, examples, etc.)

## Phase 2: Automated Checks

For each skill, run ALL of the following checks. Track results as PASS, WARN, or FAIL.

### Structural Checks

| # | Check | Severity | Rule |
|---|-------|----------|------|
| S1 | `SKILL.md` exists with exact casing | FAIL | Required file name |
| S2 | No `README.md` in the skill folder | FAIL | Must use SKILL.md, not README.md |
| S3 | Folder name is kebab-case | FAIL | No spaces, underscores, or capitals |

### Frontmatter Checks

| # | Check | Severity | Rule |
|---|-------|----------|------|
| F1 | YAML frontmatter present (`---` delimiters) | FAIL | Required |
| F2 | `name` field exists | FAIL | Required field |
| F3 | `name` is kebab-case (no spaces, underscores, capitals) | FAIL | Naming convention |
| F4 | `name` matches folder name exactly | FAIL | Must be identical |
| F5 | `description` field exists | FAIL | Required field |
| F6 | `description` is under 1024 characters | WARN | Guide recommendation |
| F7 | No XML angle brackets (`<` or `>`) in any frontmatter field | FAIL | Breaks YAML parsing |
| F8 | No "claude" or "anthropic" in the `name` field | FAIL | Reserved terms |
| F9 | `description` includes WHAT it does AND WHEN to use it (look for trigger phrases like "Use when", "Use for", "Triggered by") | WARN | Best practice for discoverability |

### Content Quality Checks

| # | Check | Severity | Rule |
|---|-------|----------|------|
| C1 | Instructions are specific/actionable (flag if body is under 50 words — likely too thin) | WARN | Must provide concrete guidance |
| C2 | SKILL.md under 5,000 words | WARN | Progressive disclosure — large skills should be split or use supporting files |
| C3 | If supporting files exist, they are referenced in SKILL.md | WARN | Bundled files must be used |
| C4 | Contains at least one code block or example | WARN | Best practice |
| C5 | Contains error handling / edge case guidance | WARN | Expected for workflow-type skills |

## Phase 3: Report

Output a structured report with these sections:

### Summary

```
Skills Audited: {total}
  PASS: {count}  |  WARN: {count}  |  FAIL: {count}
```

### Failures (if any)

List every skill with at least one FAIL, grouped by category:

```
## {category}

### {skill-name}
- [FAIL] F2: Missing `name` field in frontmatter
- [FAIL] F4: `name` ("my-skill") does not match folder ("myskill")
- [WARN] C4: No code blocks or examples found
```

### Warnings Only (if any)

List skills that passed all FAIL checks but have warnings, same format.

### Clean Skills

List skill names that passed all checks (compact, comma-separated).

## Execution Notes

- Read each SKILL.md fully before evaluating
- Parse frontmatter carefully — extract between first and second `---` lines
- For F9 (WHAT + WHEN), check if the description explains both purpose and trigger conditions; don't require exact phrasing
- For C5, only flag workflow/procedural skills (those with multi-step instructions), not reference/catalog skills
- Count words by splitting on whitespace after stripping markdown syntax
- Report actual values when a check fails (e.g., show the actual `name` vs folder name)
