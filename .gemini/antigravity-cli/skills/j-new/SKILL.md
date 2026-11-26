---
name: j-new
description: "Scaffold a new command, skill, or agent with correct structure and best practices. Use when creating a new agent asset. Do NOT use for editing existing assets (edit directly)."
---

# New

Load skill `writing-skills` before scaffolding; it holds the authoring criteria this command applies.

Create: $ARGUMENTS

If no arguments provided, ask which asset type to create (command, skill, or agent).

---

## Phase 1: Parse and validate

From `$ARGUMENTS`, take the type (`command`, `skill`, or `agent`) and an optional kebab-case name. Ask
if the type is missing or invalid.

A name must be lowercase letters, numbers, and single internal hyphens, and must not contain "claude" or
"anthropic" (reserved). Commands take a `j-` prefix. If a name fails, explain why and ask for another.

## Phase 2: Gather requirements

Ask for a one-line purpose for every type; it drives the description. Then, by type:

**Command:** what input does it take, and what is the fallback when no input is supplied? Which skills does it invoke,
and which agents does it delegate to? List candidates under `~/.gemini/antigravity-cli/skills/*/SKILL.md` and
`~/.gemini/config/agents/*.md`.

**Skill:** does it run a bundled script, and what file operations does it need? Does it depend on other skills? Should it load from task matching, or only from an explicit invocation? Keep tool permissions and invocation policy in the owning tool's supported configuration, outside shared skill frontmatter.

**Agent:** which tools, which skills to preload, and which `~/.agents/references/` categories should it
consult (`~/.agents/references/*/`)? Leave persistent memory unset: agents here are dispatched with a frozen, report-only packet.

Gemini role definitions use `model: inherit`. Codex TOML roles omit both `model` and `model_reasoning_effort`; the caller selects them for the assigned task at dispatch. Preserve each tool's native role requirements. Shared and Gemini skill frontmatter carries only `name` and `description`.

## Phase 3: Draft the description

A skill description is **triggering conditions only**: never a workflow summary, because the model will
follow the summary instead of reading the body. Commands and agents describe what they are and when to
reach for them, and may scope out near-misses to keep routing clean.

Descriptions are loaded on every turn, so keep them to what routing actually needs. Shared skills under
`.agents/` additionally must fit 64 characters and start with "Use when"; `.claude/` skills have no such
budget. No `<` or `>` anywhere in frontmatter.

Present the draft for approval before writing.

## Phase 4: Scaffold

Paths are flat. There are no category subdirectories:

| Type | Path |
|---|---|
| Command | `.gemini/antigravity-cli/skills/j-{name}/SKILL.md` |
| Skill | `~/.gemini/antigravity-cli/skills/{name}/SKILL.md` |
| Agent | `~/.gemini/config/agents/{name}.md` |

**Command**: route, do not explain. Gather context, then invoke a skill or delegate to an agent. If you
find yourself writing the methodology inline, it belongs in a skill instead:

```markdown
---
name: j-{name}
description: "{description}"
---

Load skill `{skill}` before starting.

{Purpose}: the user's provided input

If no arguments provided, {fallback}.

## Phase 1: {name}
{what to gather}

## Phase 2: {name}
{the work, or the delegation}
```

**Skill**: state the goal and why it matters, so judgment covers cases you did not foresee. Put
mechanical steps in a `scripts/` file and call it; put heavy detail in `references/` so it loads on
demand:

```markdown
---
name: {name}
description: "{trigger}"
---

# {Title}

{The core principle, and why it matters.}

## When to use
{Concrete symptoms or triggers.}

## {The method}
{Steps or pattern. One worked example beats three.}

## Gotchas
- **{failure mode}**: {what happens} -- {what to do instead}
```

Do not add rationalization tables, "Red Flags: STOP" lists, or an all-caps non-negotiable rule. Those
narrow the model's judgment and are usually wrong for some real situation. State the goal and the
consequence instead.

**Agent**: a role statement plus pointers. Delegate methodology to skills:

```markdown
---
name: {name}
description: "{description}"
tools: [read_file, run_command]
subagent: true
mainAgent: true
model: inherit
skills:
  - {skill}
---

You are a {role}. {One sentence on approach.}

Reference library at ~/.agents/references/{category}/:
- {reference}, {reference}, {reference}

Read the relevant reference file(s) for the user's topic before responding.
```

Every reference stem listed must exist; the audit script fails on a dangling index entry.

## Phase 5: Validate

This tree ships no audit script, so walk the rules before writing:

- frontmatter opens and closes with `---`, `name` is kebab-case, and `description` fits its budget
- `name` matches the skill folder, the command filename, or the agent convention
- `description` is a trigger, not a workflow summary, and carries no `<` or `>`
- every skill, agent, and reference the body names resolves to a real file
- shared and Gemini command skills carry only `name` and `description` in frontmatter
- Gemini roles use `model: inherit`; Codex TOML roles omit `model` and `model_reasoning_effort`
- the body carries at least one concrete example, not only prose

Then confirm by judgment: does every section teach something a strong model would otherwise get wrong
or vague? Cut what does not.

## Phase 6: Present and register

Show the file, get approval, write it, and confirm the path.

Then register it. A new command needs its native Claude command, Gemini command skill, and shared `cmd-j-*` skill. That shared skill is the Codex command; do not create a Codex prompt fallback. Preserve explicit-only Codex command policy in its `agents/openai.yaml`. A new agent needs native definitions in all three tool trees. Shared non-command skills must match their Gemini copies byte-for-byte. Claude skill bodies may differ; shared reference twins must expose the same headings. Where a parity test guards asset sets, the repo's own CLAUDE.md names it; declare any single-tree asset in that test's exception list.

Update the GEMINI.md Knowledge Base Structure section if a new asset directory was added, then run the repo's
lint and test commands. See `documentation-validation`.

If the repository keeps an eval suite for its assets, add the new asset's cases (trigger and effect
cases for a skill or command, effect cases for an agent). Offer to run its smoke target; it may bill.

### Frontmatter reference

| Field | Command | Skill | Agent |
|-------|---------|-------|-------|
| `name`, `description` | Required | Required | Required |
| `tools` | N/A | N/A | Recommended |
| `skills` | N/A | N/A (name dependencies in the body) | Optional (preload) |
| `model` | Omit | Omit | `inherit` |
| `subagent`, `mainAgent` | N/A | N/A | `true` |

A skill under `.gemini/antigravity-cli/skills/` or `.agents/skills/` carries only `name` and `description`. Codex explicit-only invocation policy lives in `agents/openai.yaml` with `policy.allow_implicit_invocation: false`.

Use only tool names supported by the target harness; do not copy Claude-only fields into shared or Gemini skills.

Field semantics: `writing-skills`, SKILL.md Structure.

### Cross-References

- **skill:writing-skills**: authoring criteria, frontmatter spec, discoverability
- **skill:skill-audit**: the validation script and the judgment axes
