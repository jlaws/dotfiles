---
name: project-scaffolding
description: "Use when scaffolding a new or unconfigured repository."
allowed-tools: Read, Write, Edit, Grep, Glob, Bash, WebFetch, WebSearch
---

# Project Scaffolding

A scaffold is a set of claims about a project: this is the stack, these are the versions, this is how
you run the tests. Every claim comes from the repo in front of you or from a live registry. A scaffold
assembled from recall looks right on the day it is written and is wrong in ways that surface later,
when someone trusts it.

## When to use

- A repo is empty or nearly so and its shape has not been decided.
- A repo has code but no `CLAUDE.md`, `AGENTS.md`, or `GEMINI.md`.
- Agent config exists but is stale, partial, or contradicts the code.

## What the target repo does not need

The user's `~/.claude`, `~/.codex`, `~/.gemini`, and `~/.agents` trees already load in every
repository. Generated config carries only what is true of *this* repo: how to build, test, and lint
it; its architecture in a paragraph; the gotchas a newcomer trips on; its directory map.

Restating global guidance is the most common failure in this workflow. It costs context on every
request and drifts from the source the moment either side changes.

## Phase 1: Read before asking

Inventory the target before forming any opinion: `README*` and other docs, dependency manifests and
lockfiles, existing agent config, existing lint/format/test config, the directory shape, and recent
history if any exists. Then classify:

| State | Path through this skill |
|---|---|
| Empty or docs only | Full interview (Phase 2), then everything downstream |
| Code, no agent config | Infer and confirm, then write config and fill toolchain gaps |
| Partial config | Diff each file you would change, confirm, apply only what was approved |

A README is the highest-value artifact in the repo. If one exists, the purpose, audience, and stack
are usually already stated. Ask about what it leaves open, not about what it says.

## Phase 2: Interview only the gaps

For an empty repo, use the `design-first` skill to establish purpose, users, constraints, and success
criteria before writing anything. For a repo that already has code, state your inferred purpose and
stack and ask for correction. Re-asking what the repo already answered is the fastest way to lose the
user's attention for the questions that matter.

Every question carries your recommended answer.

## Phase 3: Derive the toolchain, name no tool from memory

Detect the ecosystem from the manifests, load the matching file under `references/languages/` for
idioms and packaging, and get versions from the registry rather than recall. "Latest" held in model
weights is stale by construction; the gap is invisible and grows every day after the cutoff.

Registry queries are mechanism, not opinion. Use the ecosystem's own:

| Ecosystem | Current version comes from |
|---|---|
| Python | `uv python list`, `pip index versions <pkg>` |
| Node | `npm view <pkg> version`, `npm view <pkg> dist-tags` |
| Rust | `cargo search <crate>`, `cargo info <crate>` |
| Go | `go list -m -versions <module>` |
| Anything else | That ecosystem's package index; WebSearch only when no registry answers |

Which linter, formatter, type checker, and test runner to adopt is the project's decision, not this
skill's. Propose a toolchain drawn from the ecosystem's current consensus and the relevant
`references/languages/` file, show the exact commands it implies, and confirm before writing.

`language-specialist` can help with idioms and project layout, but it has no web tools. Never take a
version number from it.

## Phase 4: Write the three agent config files

`CLAUDE.md`, `AGENTS.md`, and `GEMINI.md`, each holding the repo-specific content above, with
tool-specific notes where the tools genuinely differ. Codex reads `AGENTS.md` hierarchically from the
project root down, so directory-level notes belong in nested files rather than in one long root file.

Keep each one short enough that a reader gets through it. Guidance that applies to every repo belongs
in the user's global trees, not here.

## Phase 5: Project settings

Write `.claude/settings.json` with a `permissions.allow` list covering the project's routine commands,
such as its test, lint, and build invocations. That file is the one meant to be committed;
`settings.local.json` is personal and gets git-excluded automatically. Allow entries merge with the
user's own rules rather than replacing them.

Do not write `.codex/config.toml`. Codex does read project-level config, but its only documented
project levers are `approval_policy` and `sandbox_mode`, and loosening either on the project's behalf
is a security decision the repo's author should make deliberately.

Do not write `.gemini/settings.json`. A project file there overrides the user's settings rather than
merging with them, so a generated file can silently shadow global configuration.

Record both omissions in the generated config, with the reason. Otherwise a later reader "fixes" the
gap by adding files that do nothing useful.

## Phase 6: Supporting artifacts

- `.gitignore` for the detected stack. This is a plain file; run no git commands.
- `README.md` only if one is absent, following `references/documentation/readme-template.md`.
- `docs/adr/` seeded with the conventions in `references/architecture/architecture-decision-records.md`:
  topic directories, no sequence numbers, ordering by the `created` date, and `## Reversal Conditions`
  filled in on every record.
- One seed ADR capturing the stack decision from Phase 3, since the interview already produced its
  context, options, and rationale.

Write no CI workflows and no dependency-bot configuration unless asked.

## Phase 7: One gate, then proof

Show the exact install commands, ask once, then run them. Afterwards run the project's lint and test
gate a single time and report its real output.

Setup that has not been executed is a hypothesis. A scaffold whose gate has never run is where the
mistakes hide, because every file individually looks plausible.

## Gotchas

- **Copying global guidance into the target repo**: costs context on every request and drifts. Write
  only what is specific to this repo.
- **Version numbers from recall**: silently stale. Every version traces to a registry query.
- **Installing before the gate**: the user loses the chance to redirect the stack after seeing it.
  One confirmation, covering all of it, before anything is installed.
- **Overwriting existing config**: show a diff per file and confirm. Someone chose what is already
  there.
- **Unrequested scaffolding**: CI, git operations, dependency bots, and service configs are separate
  decisions with their own consequences. Scaffold what was asked for.
