---
name: j-new
description: "Scaffold a new .claude/ command, skill, or agent with correct structure and best practices. Use when creating any new .claude/ asset. Do NOT use for editing existing assets (edit directly)."
argument-hint: "<type: command|skill|agent> [name]"
model: sonnet
---

Load skill `writing-skills` before scaffolding — it holds the authoring criteria this command applies.

Create: $ARGUMENTS

If no arguments provided, ask which asset type to create (command, skill, or agent).

---

## Phase 1: Parse and validate

From `$ARGUMENTS`, take the type (`command`, `skill`, or `agent`) and an optional kebab-case name. Ask
if the type is missing or invalid.

A name must be lowercase letters, numbers, and single internal hyphens, and must not contain "claude" or
"anthropic" (reserved). Commands take a `j-` prefix. If a name fails, explain why and ask for another.

## Phase 2: Gather requirements

Ask for a one-line purpose for every type — it drives the description. Then, by type:

**Command:** does it take arguments (and what should `argument-hint` say)? Which skills does it invoke,
and which agents does it delegate to? List candidates with `Glob(".claude/skills/*/SKILL.md")` and
`Glob(".claude/agents/*.md")`.

**Skill:** which tools does it need without per-use approval? Does it depend on other skills? Should
Claude invoke it automatically, or only the user (`disable-model-invocation: true`)? Does it apply only
to certain files (`paths:`)?

**Agent:** which tools, which skills to preload, and which `.claude/references/` categories should it
consult (`Glob(".claude/references/*/")`)? Persistent memory scope, if any?

Leave `model` unset so the asset inherits the session model. Set it only when a specific tier is
genuinely required; tier aliases (`opus`, `sonnet`, `haiku`, `fable`) float across model generations, so
prefer an alias over a pinned ID. Set `effort` when the work is reliably cheaper (`low`, `medium`) or
reliably demanding (`xhigh`).

## Phase 3: Draft the description

A skill description is **triggering conditions only** — never a workflow summary, because Claude will
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
| Command | `.claude/commands/{name}.md` |
| Skill | `.claude/skills/{name}/SKILL.md` |
| Agent | `.claude/agents/{name}.md` |

**Command** — route, do not explain. Gather context, then invoke a skill or delegate to an agent. If you
find yourself writing the methodology inline, it belongs in a skill instead:

```markdown
---
name: {name}
description: "{description}"
argument-hint: "{hint}"        # only if it reads $ARGUMENTS
---

Load skill `{skill}` before starting.

{Purpose}: $ARGUMENTS

If no arguments provided, {fallback}.

## Phase 1: {name}
{what to gather}

## Phase 2: {name}
{the work, or the delegation}
```

**Skill** — state the goal and why it matters, so judgment covers cases you did not foresee. Put
mechanical steps in a `scripts/` file and call it; put heavy detail in `references/` so it loads on
demand:

```markdown
---
name: {name}
description: "{trigger}"
allowed-tools: {tools}
---

# {Title}

{The core principle, and why it matters.}

## When to use
{Concrete symptoms or triggers.}

## {The method}
{Steps or pattern. One worked example beats three.}

## Gotchas
- **{failure mode}**: {what happens} — {what to do instead}
```

Do not add rationalization tables, "Red Flags — STOP" lists, or an all-caps non-negotiable rule. Those
narrow the model's judgment and are usually wrong for some real situation. State the goal and the
consequence instead.

**Agent** — a role statement plus pointers. Delegate methodology to skills:

```markdown
---
name: {name}
description: "{description}"
tools: {tools}
skills:
  - {skill}
---

You are a {role}. {One sentence on approach.}

Reference library at .claude/references/{category}/:
- {reference}, {reference}, {reference}

Read the relevant reference file(s) for the user's topic before responding.
```

Every reference stem listed must exist — the audit script fails on a dangling index entry.

## Phase 5: Validate

Run the mechanical checks rather than reviewing by hand:

```bash
python3 .claude/skills/skill-audit/scripts/audit.py . --type {type}s
```

Fix what it reports. Then confirm by judgment: does every section teach something a strong model would
otherwise get wrong or vague? Cut what does not.

## Phase 6: Present and register

Show the file, get approval, write it, and confirm the path.

Then register it. A new command needs its Codex and Gemini counterparts plus the shared `cmd-j-*` skill;
a new agent needs native definitions in all three tool trees; a shared workflow skill needs an
`.agents/skills/` source. `.claude/` skill and reference *bodies* have intentionally diverged from
`.agents/` and are not kept in sync — only the asset sets are, which
`tests/test_agent_config.py` enforces. Declare any single-tree asset in its exception list there.

Update the CLAUDE.md Knowledge Base Structure section if the asset set changed, then run `make test` and
`make check`. See `documentation-validation`.

### Frontmatter reference

| Field | Command | Skill | Agent |
|-------|---------|-------|-------|
| `name`, `description` | Required | Required | Required |
| `argument-hint` | If `$ARGUMENTS` | Optional | N/A |
| `allowed-tools` / `disallowed-tools` | N/A | Recommended | N/A |
| `tools` | N/A | N/A | Recommended |
| `skills` | N/A | Optional (deps) | Optional (preload) |
| `model`, `effort` | Optional | Optional | Optional |
| `paths` | N/A | Optional | N/A |
| `disable-model-invocation`, `user-invocable` | N/A | Optional | N/A |
| `context: fork`, `agent` | N/A | Optional | N/A |
| `memory` | N/A | N/A | Optional (user/project/local) |
| `color`, `maxTurns`, `permissionMode` | N/A | N/A | Optional |

Valid tool names: Read, Write, Edit, Grep, Glob, Bash, WebFetch, WebSearch, NotebookEdit, Task, Skill.

`compatibility` is not a Claude Code field and does nothing — do not emit it.

### Cross-References

- **skill:writing-skills** — authoring criteria, frontmatter spec, discoverability
- **skill:skill-audit** — the validation script and the judgment axes
