---
name: skill-audit
description: "Audit the .agents/ knowledge base — skills and references for conformance and integrity. Use when validating conformance after creating or modifying any .agents/ asset, checking naming conventions, or verifying knowledge base integrity. Do NOT use for general code quality (use code-quality) or code review (use code-review-patterns)."
---

# Knowledge Base Audit

Full conformance audit of all assets under `.agents/`.

## Scoping

Determine scope from arguments:
- **Empty** -> audit ALL asset types
- **Asset type** (`skills`, `commands`, `agents`, `references`, `config`) -> audit only that type
- **Category/path** (e.g., `skills/workflow`, `agents/code-reviewer`, `commands/ai-ml/ml`) -> audit only matching assets

## Phase 1: Discovery

Enumerate all assets by type:

| Type | Location | Pattern |
|------|----------|---------|
| Skills | `.agents/skills/{category}/{name}/SKILL.md` | Folder with SKILL.md |
| References | `.agents/references/{category}/{name}.md` | Markdown files |

For each asset, record: type, category, name, file path, any supporting files.

## Phase 2: Automated Checks

Track every result as PASS, WARN, or FAIL. Report actual values on failure.

---

### 2A. Skill Checks

For each skill directory under `.agents/skills/`:

**Structural**

| # | Check | Sev | Rule |
|---|-------|-----|------|
| SK-S1 | `SKILL.md` exists with exact casing | FAIL | Required file |
| SK-S2 | No `README.md` in skill folder | FAIL | Must use SKILL.md |
| SK-S3 | Folder name is kebab-case | FAIL | No spaces, underscores, capitals |

**Frontmatter**

| # | Check | Sev | Rule |
|---|-------|-----|------|
| SK-F1 | YAML frontmatter present (`---` delimiters) | FAIL | Required |
| SK-F2 | `name` field exists | FAIL | Required |
| SK-F3 | `name` is kebab-case | FAIL | Letters, numbers, hyphens only |
| SK-F4 | `name` matches folder name | FAIL | Must be identical |
| SK-F5 | `description` field exists | FAIL | Required |
| SK-F6 | `description` under 1024 chars | WARN | Guide limit |
| SK-F7 | No XML `<`/`>` in frontmatter fields | FAIL | Breaks YAML parsing |
| SK-F8 | `name` max 64 chars | FAIL | Agent Skills spec limit |
| SK-F9 | Description has WHAT + WHEN (trigger phrase like "Use when") | WARN | Discoverability |

**Content**

| # | Check | Sev | Rule |
|---|-------|-----|------|
| SK-C1 | Body over 50 words | WARN | Too thin otherwise |
| SK-C2 | Under 5,000 words | WARN | Split large skills |
| SK-C3 | Supporting files referenced in SKILL.md | WARN | Bundled files must be used |
| SK-C4 | At least one code block or example | WARN | Best practice |
| SK-C5 | Error/edge-case guidance (workflow skills only) | WARN | Expected for procedural skills |

---

### 2B. Reference Checks

For each `.md` file under `.agents/references/`:

**Structural**

| # | Check | Sev | Rule |
|---|-------|-----|------|
| RF-S1 | Filename is kebab-case | FAIL | Naming convention |
| RF-S2 | File lives inside a category subdirectory | WARN | Expected: `references/{category}/{name}.md` |

**Content**

| # | Check | Sev | Rule |
|---|-------|-----|------|
| RF-C1 | File has at least one markdown heading (`#`) | WARN | Should be structured |
| RF-C2 | Over 50 words of content | WARN | Likely stub |
| RF-C3 | Referenced by at least one skill | WARN | Orphan reference |

---

### 2C. Cross-Reference Integrity

Verify links between assets resolve:

| # | Check | Sev | Rule |
|---|-------|-----|------|
| XR-1 | Skill body cross-references to other skills resolve | WARN | Broken skill ref |
| XR-2 | Skill supporting files in `references/` subdirs are referenced in SKILL.md | WARN | Orphan supporting file |

## Phase 3: Report

### Summary

```
Knowledge Base Audit
====================
Skills:     {n} audited  |  {pass} pass  |  {warn} warn  |  {fail} fail
References: {n} audited  |  {pass} pass  |  {warn} warn  |  {fail} fail
Cross-Refs: {n} checks   |  {pass} pass  |  {warn} warn  |  {fail} fail
─────────────────────────────────────────────────────────────────────────
Total:      {N} checks   |  {P} pass     |  {W} warn     |  {F} fail
```

### Failures (grouped by asset type, then category)

```
## Skills / {category}
### {skill-name}
- [FAIL] SK-F4: `name` ("my-skill") does not match folder ("myskill")
```

### Warnings Only

Same format, for assets with zero FAILs but one or more WARNs.

### Clean Assets

Comma-separated list per type:
```
Skills: code-quality, code-review-patterns, ...
References: auth-implementation-patterns, ...
```

## Execution Notes

- Read each file fully before evaluating
- Parse frontmatter between first and second `---` lines
- For description WHAT+WHEN checks, accept varied phrasing (don't require exact "Use when")
- For SK-C5, only flag workflow/procedural skills, not reference/catalog skills
- For RF-C3 (orphan check), search all skills for the reference filename
- For cross-ref checks, normalize paths (strip `.agents/`, `.md`, leading/trailing slashes)
- Count words by splitting on whitespace after stripping markdown syntax
- Report actual values on every failure (show expected vs actual)
- When scoped to a single asset type, still run cross-reference checks relevant to that type
