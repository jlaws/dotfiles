---
name: skill-audit
description: "Audit the .claude/ knowledge base — skills, commands, agents, references, config, cross-references, and documentation currency. Use when validating conformance after creating or modifying any .claude/ asset, checking naming conventions, or verifying knowledge base integrity. Do NOT use for general code quality (use code-quality) or code review (use code-review-patterns)."
compatibility: claude-code
allowed-tools: Read, Grep, Glob, Bash
---

# Knowledge Base Audit

Full conformance audit of KB assets across all tool trees. `.agents/` is the source of truth; `.claude/`, `.cursor/`, `.codex/`, `.gemini/` carry mirrored copies.

## Scoping

Determine scope from arguments:
- **Empty** -> audit ALL asset types
- **Asset type** (`skills`, `commands`, `agents`, `references`, `config`) -> audit only that type
- **Name/path** (e.g., `skills/writing-plans`, `agents/code-reviewer`, `commands/j-arch`) -> audit only matching assets

## Phase 1: Discovery

Enumerate all assets by type:

| Type | Location | Pattern |
|------|----------|---------|
| Skills | `.claude/skills/{name}/SKILL.md` | Folder with SKILL.md |
| Commands | `.claude/commands/{name}.md` | Flat markdown files |
| Agents | `.claude/agents/{name}.md` | Markdown files |
| References | `.claude/references/{category}/{name}.md` | Markdown files |
| Config | `.claude/CLAUDE.md`, `.claude/settings.json`, `.claude/settings.local.json` | Fixed paths |

For each asset, record: type, category, name, file path, any supporting files.

**Multi-tree scope.** `.agents/skills` and `.agents/references` are canonical; `.claude/`, `.cursor/`, `.codex/`, and `.gemini/` carry mirrored copies. Audit `.agents/` as the source of truth, then diff each mirror against it and flag drift (a mirror changed without its source, or vice versa). Skills/references duplicated in `.claude/` should differ only by frontmatter.

## Phase 2: Automated Checks

Track every result as PASS, WARN, or FAIL. Report actual values on failure.

---

### 2A. Skill Checks

For each skill directory under `.claude/skills/`:

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
| SK-F8 | No "claude"/"anthropic" in `name` | FAIL | Reserved terms |
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

### 2B. Agent Checks

For each `.md` file under `.claude/agents/`:

**Structural**

| # | Check | Sev | Rule |
|---|-------|-----|------|
| AG-S1 | Filename is kebab-case (`.md` extension) | FAIL | Naming convention |
| AG-S2 | No subdirectories under `agents/` | WARN | Agents are flat files |

**Frontmatter**

| # | Check | Sev | Rule |
|---|-------|-----|------|
| AG-F1 | YAML frontmatter present | FAIL | Required |
| AG-F2 | `name` field exists | FAIL | Required |
| AG-F3 | `name` matches filename (without `.md`) | FAIL | Must be identical |
| AG-F4 | `description` field exists | FAIL | Required |
| AG-F5 | `description` under 1024 chars | WARN | Keep concise |
| AG-F6 | `tools` field exists | WARN | Should declare tool access |
| AG-F7 | `tools` only lists valid tool names (Read, Grep, Glob, Bash, Write, Edit, NotebookEdit, WebFetch, WebSearch) | WARN | Invalid tools ignored at runtime |

**Content**

| # | Check | Sev | Rule |
|---|-------|-----|------|
| AG-C1 | Body over 20 words | WARN | Must provide role instructions |
| AG-C2 | If `skills:` field references skills, each skill path resolves to an existing SKILL.md | FAIL | Broken skill ref |
| AG-C3 | If body references `.claude/references/`, each path resolves | WARN | Broken reference link |

---

### 2C. Command Checks

For each `.md` file under `.claude/commands/`:

**Structural**

| # | Check | Sev | Rule |
|---|-------|-----|------|
| CM-S1 | Filename is kebab-case (`.md` extension) | FAIL | Naming convention |
| CM-S2 | File is flat directly under `commands/` (no category subdir) | WARN | Expected: `commands/{name}.md` |
| CM-S3 | Filename starts with `j-` prefix | WARN | Custom commands use `j-` prefix to disambiguate from built-ins |

**Frontmatter**

| # | Check | Sev | Rule |
|---|-------|-----|------|
| CM-F1 | YAML frontmatter present | FAIL | Required |
| CM-F2 | `name` field exists | FAIL | Required |
| CM-F3 | `name` matches filename (without `.md`) | FAIL | Must be identical |
| CM-F4 | `description` field exists | FAIL | Required |
| CM-F5 | `description` under 1024 chars | WARN | Keep concise |
| CM-F6 | `description` includes WHAT + WHEN trigger | WARN | Discoverability |

**Content**

| # | Check | Sev | Rule |
|---|-------|-----|------|
| CM-C1 | Body over 10 words | WARN | Must provide instructions |
| CM-C2 | If body uses `$ARGUMENTS`, frontmatter has `argument-hint` | WARN | Helps user know what to pass |
| CM-C3 | If body invokes a skill (e.g., "load skill", "follow skill", "invoke skill"), skill exists | FAIL | Broken skill ref |
| CM-C4 | If body invokes an agent, agent exists under `.claude/agents/` | WARN | Broken agent ref |

---

### 2D. Reference Checks

For each `.md` file under `.claude/references/`:

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
| RF-C3 | Referenced by at least one agent, command, or skill | WARN | Orphan reference |

---

### 2E. Config Checks

**CLAUDE.md**

| # | Check | Sev | Rule |
|---|-------|-----|------|
| CF-C1 | `.claude/CLAUDE.md` exists | FAIL | Required config |
| CF-C2 | Has at least one heading | WARN | Should be structured |
| CF-C3 | Under 10,000 words | WARN | Token budget — large files waste context |

**Settings**

| # | Check | Sev | Rule |
|---|-------|-----|------|
| CF-S1 | `.claude/settings.json` exists | WARN | Expected for configured projects |
| CF-S2 | `settings.json` is valid JSON | FAIL | Parse error breaks config |
| CF-S3 | If `settings.local.json` exists, it is valid JSON | FAIL | Parse error breaks config |

---

### 2F. Cross-Reference Integrity

Verify links between assets resolve:

| # | Check | Sev | Rule |
|---|-------|-----|------|
| XR-1 | Agent `skills:` entries resolve to existing skill folders | FAIL | Broken skill ref |
| XR-2 | Agent body reference paths (`.claude/references/...`) resolve | WARN | Broken ref link |
| XR-3 | Command body skill invocations resolve | FAIL | Broken skill ref |
| XR-4 | Command body agent references resolve | WARN | Broken agent ref |
| XR-5 | Skill body cross-references to other skills resolve | WARN | Broken skill ref |
| XR-6 | Skill supporting files in `references/` subdirs are referenced in SKILL.md | WARN | Orphan supporting file |

### 2G. Documentation Currency

| # | Check | Sev | Rule |
|---|-------|-----|------|
| DOC-1 | Mirror parity — shared skill/reference bodies identical across `.agents`↔`.claude` (differ only by frontmatter); each `j-*` command and each agent has its `.gemini`/`.codex` counterpart and its `cmd-`/`agent-` source skill | WARN | Drift means one tree's docs are stale |
| DOC-2 | Description reflects behavior — the `description` still matches what the body does after edits (no stale/misleading trigger) | WARN | Stale description misroutes invocation |
| DOC-3 | Registration — a new or renamed asset is discoverable in `.claude/CLAUDE.md` Knowledge Base Structure (and MEMORY index if the repo has one); no lingering references to a renamed/removed asset | WARN | Unregistered or dangling asset |

## Phase 3: Report

### Summary

```
Knowledge Base Audit
====================
Skills:     {n} audited  |  {pass} pass  |  {warn} warn  |  {fail} fail
Agents:     {n} audited  |  {pass} pass  |  {warn} warn  |  {fail} fail
Commands:   {n} audited  |  {pass} pass  |  {warn} warn  |  {fail} fail
References: {n} audited  |  {pass} pass  |  {warn} warn  |  {fail} fail
Config:     {n} checks   |  {pass} pass  |  {warn} warn  |  {fail} fail
Cross-Refs: {n} checks   |  {pass} pass  |  {warn} warn  |  {fail} fail
─────────────────────────────────────────────────────────────────────────
Total:      {N} checks   |  {P} pass     |  {W} warn     |  {F} fail
```

### Failures (grouped by asset type, then category)

```
## Skills / {category}
### {skill-name}
- [FAIL] SK-F4: `name` ("my-skill") does not match folder ("myskill")

## Agents
### {agent-name}
- [FAIL] AG-F3: `name` ("reviewer") does not match filename ("code-reviewer")

## Commands
### {command-name}
- [FAIL] CM-C3: References skill "nonexistent" — not found

## Cross-References
- [FAIL] XR-1: Agent "ml-engineer" → skill "ai-ml/training" — not found
```

### Warnings Only

Same format, for assets with zero FAILs but one or more WARNs.

### Clean Assets

Comma-separated list per type:
```
Skills: code-quality, code-review-patterns, ...
Agents: ml-engineer, test-writer, ...
Commands: j-arch, j-new, ...
References: auth-implementation-patterns, ...
```

## Health Score (qualitative)

Beyond PASS/WARN/FAIL, rate the KB on four axes and name the weakest — that is where to invest next.

| Axis | Question |
|------|----------|
| Groundedness | Do skills/refs use concrete repo paths and repo-specific examples, or generic advice? |
| Coverage | Are the common task types covered by a skill, with no large gaps? |
| Freshness | Any asset unreferenced by an agent/command/config, or pointing at deleted files? |
| Structure | Frontmatter valid, cross-references resolve, naming conventions hold, within line budgets? |

**Documentation currency:** mirror parity across trees and description-behavior match are part of freshness — see checks DOC-1..3.

**Groundedness rule:** prefer concrete file paths and repo-specific examples over generic advice — a skill that could apply to any repo teaches little about this one. Flag generic-only skills for grounding.

**Staleness:** an asset never referenced by any agent, command, or config across all trees is a removal/merge candidate — report it.

## Execution Notes

- Read each file fully before evaluating
- Parse frontmatter between first and second `---` lines
- For description WHAT+WHEN checks, accept varied phrasing (don't require exact "Use when")
- For SK-C5, only flag workflow/procedural skills, not reference/catalog skills
- For RF-C3 (orphan check), search all agents, commands, and skills for the reference filename
- For cross-ref checks, normalize paths (strip `.claude/`, `.md`, leading/trailing slashes)
- Count words by splitting on whitespace after stripping markdown syntax
- Report actual values on every failure (show expected vs actual)
- When scoped to a single asset type, still run cross-reference checks relevant to that type
