---
name: skill-audit
description: "Use when auditing agent knowledge-base integrity."
---

# Knowledge Base Audit

Conformance and quality audit across shared workflows (`.agents/`) and tool-native agent trees (`.claude/`, `.codex/`, `.gemini/`).

## Scoping

Determine scope from arguments:
- **Empty** -> audit ALL asset types (`skills`, `commands`, `agents`, `references`, `config`)
- **Asset type** (`skills`, `commands`, `agents`, `references`, `config`) -> audit only that type
- **Name/path** (e.g., `skills/writing-plans`, `agents/code-reviewer`, `commands/j-arch`) -> audit only matching assets
- **`adoption`** -> run the transcript adoption pass alone (never part of an unscoped run, because it reads conversation history)
- **`memory`** -> Claude Code auto-memory stores under `~/.claude` only (Codex and Antigravity have no such store)

## Phase 1: Discovery & Automated Checks

Enumerate assets across the repository's trees:

| Type | Location |
|------|----------|
| Skills | `.agents/skills/{name}/SKILL.md`, `.claude/skills/{name}/SKILL.md`, `.gemini/antigravity-cli/skills/{name}/SKILL.md` |
| Commands | `.agents/skills/cmd-j-*/SKILL.md` (Codex), `.claude/commands/{name}.md`, `.gemini/antigravity-cli/skills/j-*/SKILL.md` |
| Agents | `.codex/agents/{name}.toml`, `.claude/agents/{name}.md`, `.gemini/config/agents/{name}.md` |
| References | `.agents/references/{category}/{name}.md`, `.claude/references/{category}/{name}.md` |
| Config | `.codex/AGENTS.md`, `.codex/config.toml`, `.claude/CLAUDE.md`, `.claude/settings.json`, `.gemini/GEMINI.md`, `.gemini/antigravity-cli/settings.json` |

When the repository ships `.claude/skills/skill-audit/scripts/audit.py` and `tests/test_agent_config.py`, run them first instead of re-deriving mechanical checks by hand:
- `audit.py` checks naming (kebab-case, `j-` command prefix, no `README.md` in skill folders), YAML frontmatter (`name` == stem/folder, quoted `description`, no `<`/`>`, valid tool names, no unscoped tool pre-approvals, floating model aliases on Claude agents, no `model` on commands), `SKILL.md` size (`<= 20,000` bytes so compaction keeps the body), reference/skill/agent link resolution, anchor resolution, orphan references, and JSON config validity.
- `tests/test_agent_config.py` enforces multi-tree parity and safety invariants:
  - Shared non-command skills in `.agents/skills/` and `.gemini/antigravity-cli/skills/` are byte-identical across all files, carry only `name` and `description` in frontmatter, and keep `.agents/` descriptions `<= 64` chars starting with `"Use when"`.
  - `.gemini/config/agents/*.md` declare `subagent: true`, `mainAgent: true`, `model: inherit`, and match body skill lists to frontmatter `skills:`.
  - `.codex/agents/*.toml` declare `name`, `description`, and non-empty `developer_instructions`, omitting `model` and `model_reasoning_effort`.
  - `.claude/references/` and `.agents/references/` hold the same files and expose the exact same ordered Markdown headings (`docs/adr/workflow/reference-tree-section-parity.md`).
  - Reference citations use `~/.agents/references/<category>/` in `.agents`, `.codex`, and `.gemini`, never project-relative paths or `.claude/references/` in shared assets.
  - Always-loaded configs stay under `ALWAYS_LOADED_CEILINGS` while keeping safety lines, and KB prose contains no em dashes, no ` -- ` prose dashes, and no inline source credits.

In a repository without those scripts (this shared skill ships no `scripts/` directory), walk the rules above directly across every discovered asset and classify each check as `PASS`, `WARN`, or `FAIL` with actual values cited.

## Phase 2: Adoption (On Request Only)

Only when explicitly invoked with `adoption`, inspect readable session transcripts by hand (or via `.claude/skills/skill-audit/scripts/adoption.py . --exclude dotfiles` when auditing a Claude tree that ships it). Five limits govern what counts:
- Count conversation prose only: exclude tool calls, tool output, hook output, per-session skill catalogs, and record envelopes (`cwd`, `gitBranch`).
- Scan nested subagent session directories as well as top-level project transcripts.
- Exclude sessions from the repository that edits the knowledge base itself, which name every asset.
- Transcripts are pruned, so `last_seen` is bounded by retention.
- An asset invoked solely via tool calls leaves zero prose mentions: `no evidence` means not found in prose, not unused. Weigh it alongside reference reachability when judging orphans; treat `cold` (found outside the active window) as positive evidence of use.

## Phase 3: Judge What Scripts Cannot

Rate the knowledge base across five qualitative axes and name the weakest (where to invest next):

| Axis | Question |
|------|----------|
| Groundedness | Do skills and references use real repo paths and repo-specific examples, or generic advice that would fit any codebase? |
| Coverage | Is each common task type served by some asset, with no large gap? |
| Freshness | Does anything describe retired models, renamed tools, or deleted files, or have a `description` that drifted from its body? |
| Structure | Do assets read as goals, reasons, and interfaces rather than pre-frontier rulebooks? |
| Evidence | Does every quantitative claim carry how it was measured, or say it is unmeasured? Are third-party numbers left out with sources in `REFERENCES.md`? Are counterfactual savings labeled estimates and hunches labeled speculative? |

### High-Signal Defects to Flag

- **Generic content**: sections a strong model already knows unprompted (textbook taxonomies, framework tours, restated SOLID/DRY definitions). Keep only version-gated facts, numbers with provenance, opinionated tie-breakers, and repo-specific gotchas.
- **Rules where goals belong**: all-caps "Iron Laws", pre-refuted excuse tables, and "Red Flags: STOP" banners that prevent judgment on valid edge cases (`writing-skills`, The Four Shifts).
- **Repetition across surfaces**: the same rule restated across `CLAUDE.md`/`AGENTS.md`/`GEMINI.md`, a command, and a skill instead of one owner plus pointers.
- **Inlined skill bodies**: commands pasting methodology instead of loading the skill.
- **Stale model or tool facts**: retiring model IDs, outdated effort flags, or tool names belonging to another harness (`WebFetch`/`Explore` in Gemini/Codex skills, legacy `@google/gemini-cli` tools in Antigravity).
- **Unsourced numbers**: guessed percentages, multipliers, or week estimates stated without measurement provenance (`writing-skills`, Claims and Evidence).
- **Orphans & registration**: references unreachable from any agent, command, or skill, or renamed/new assets missing from harness structure docs or `MEMORY.md` (when present).

## Phase 4: Report

Lead with automated check counts (`PASS` / `WARN` / `FAIL`), then judgment findings grouped by axis (worst first), citing `file:line`, quoted text, and failed criterion. Close with the weakest axis and the single highest-value next change.
