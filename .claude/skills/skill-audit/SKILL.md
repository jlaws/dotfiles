---
name: skill-audit
description: "Use when auditing agent knowledge-base integrity."
allowed-tools: Read, Grep, Glob, Bash
---

# Knowledge Base Audit

Three parts. One script settles everything mechanical, a second measures which assets actually show
up in transcripts, and you spend your attention on the judgment calls neither can make.

The scope argument routes: `skills`, `commands`, `agents`, `references`, and `config` narrow the
mechanical checks and the judgment pass; `adoption` runs the transcript report alone; a path audits
that asset. No argument runs everything except adoption, which is never automatic.

## Run the mechanical checks

```bash
python3 .claude/skills/skill-audit/scripts/audit.py .
```

Scope it with `--type skills|agents|commands|references|config` (repeatable). It exits non-zero when
any FAIL trips, and covers naming, frontmatter validity, tool names, model aliases, declared-skill
and agent resolution, reference reachability, relative-path and anchor resolution, and settings-file
syntax.

Read its output; do not re-derive it by hand. If a check is wrong or missing, fix the script rather
than working around it in prose.

## Measure adoption

Reached by `/j-skill-audit adoption`, or by any scope run where the orphan question comes up. It
reads the harness's own transcripts, so run it only when asked. `audit.py` never imports it and no
build target invokes it; its own tests do execute it, but every one passes `--transcripts`, so the
real transcript root is reached only when a human types the command.

```bash
python3 .claude/skills/skill-audit/scripts/adoption.py . --exclude dotfiles
```

`--exclude` takes a substring of the transcript *project directory* name, not a path. Those
directories are the project path with slashes turned to dashes, so `dotfiles` matches this repo.
Pass it with `=` when the value starts with a dash: `--exclude=-Users-me-Workspace-repo`.

| Flag | Default | Effect |
|---|---|---|
| `--transcripts DIR` | `~/.claude/projects` | where to read |
| `--since DAYS` | 30 | window, minimum 1 |
| `--all` | off | ignore the window; 1.63 GB and 1 min 52 s on this machine (2026-09-12) |
| `--active-days DAYS` | 7 | a mention newer than this counts as `active` |
| `--exclude SUBSTRING` | fixture dirs only | skip a project, repeatable |
| `--json` | off | machine-readable, including on the degraded paths |

Exit 0 on a report and on a missing transcript root, 1 when the repo argument holds no `.claude`
tree, 2 on a bad flag. It prints no progress; a full run is a silent two-minute wait.

Read the limits it prints before the table, because they decide what the numbers mean:

- Only prose counts. Tool calls, tool output, hook output, the harness's per-session catalogue, and
  the record envelope are all excluded. Counting any of them marks everything `active`.
- The cost: an asset invoked only through a tool call, never named in prose, leaves no mention.
- A session that edited the knowledge base names every asset in it, which is what `--exclude` is
  for. Without it, this repo's own sessions drown the signal.
- Transcripts are pruned, so `last seen` is bounded by retention, not by real last use.

`no evidence` means the name was never found. `cold` means it was found, but not inside
`--active-days`. Neither means unused, and neither is on its own a reason to delete anything. Use
them as evidence for the **Orphans** judgment below: an asset that is both unreferenced and
unmentioned is a strong removal candidate, while one that is unmentioned but well-referenced is more
likely a discoverability problem. A `cold` asset has positive evidence -- treat it as used.

## Judge what the script cannot

The script proves structure. These five questions decide whether the knowledge base is worth loading.
Rate each and name the weakest — that is where to invest next.

| Axis | Question |
|------|----------|
| Groundedness | Do skills and references use real repo paths and repo-specific examples, or generic advice that would fit any codebase? |
| Coverage | Is each common task type served by some asset, with no large gap? |
| Freshness | Does anything describe behavior that changed, or point at a deleted file? |
| Structure | Do the assets read as goals and interfaces, or as rulebooks? |
| Evidence | Does every quantitative claim carry how it was measured, or say it is unmeasured? Is a third party's number attributed rather than adopted? |

### What to look for

**Generic content.** A section a strong model would already produce unprompted costs tokens and crowds
out the task. The keep test: would the model answer differently or less specifically without this?
Version-gated behavior, numbers with provenance, and opinionated choices that close a decision earn
their place. Taxonomies, framework tours, and restated best practice do not.

**Rules where goals belong.** Rigid prohibitions ("never", "always", "no exceptions"), tables of
pre-refuted excuses, and red-flag lists are the pre-Claude-5 style. They are usually wrong for some
real situation, and they stop the model from noticing. Flag them for rewriting as a goal plus the
reason. The `writing-skills` skill has the four shifts.

**Repetition across surfaces.** The same instruction in CLAUDE.md, a skill, and a command means three
places to update and three chances to drift. Name every location and pick the one that owns it.

**Inlined skill bodies.** A command that pastes a skill's content instead of declaring the skill is the
most expensive form of this. Check commands against the skills they overlap.

**Stale model facts.** Pinned model IDs, capability claims about specific tiers, and effort defaults
carried over from an older generation. Tier aliases (`opus`, `sonnet`, `haiku`, `fable`) float safely,
as does `inherit`, which defers to the parent rather than naming a tier; full IDs like
`claude-sonnet-4-5-20250929` do not. `AG-F8` enforces this for agents. Commands also carry `model:`
and are not checked, so read those by hand.

**Unsourced numbers.** A percentage or multiplier reads as measured fact and the reader acts on it.
Distinguish a claim ("tables are ~40% more efficient") from a threshold ("min 80% coverage") or a
rollout stage — only the claim needs provenance. Where a practice has fixed overhead, say at what size
it exceeds the benefit. This is deliberately an axis and not a script check: a regex over `\d+%|\d+x`
cannot tell a claim from a threshold or a rollout stage, so it flags far more legitimate targets than
real ones and the only way to green is to reword them.

**Orphans.** The script lists references that no agent, command, or skill indexes. Each is a removal,
merge, or index-fix candidate — decide which, rather than leaving it unreachable. The adoption report
is the evidence for that call where the asset is a skill, agent, or command.

## Tree layout

`.agents/` is the Codex and Gemini source for shared skills and references. Each tool owns its own
agents and commands.

**`.claude/` has intentionally diverged.** Its skill and reference bodies are written for the Claude 5
generation and no longer mirror `.agents/`. Body and description differences between the two trees are
expected and are not findings. A parity test, named in the repo's own CLAUDE.md where one exists, pins
only which assets exist in each tree, with single-tree assets declared explicitly.

Still worth checking by hand, because no test covers it:

- every `cmd-j-*` shared skill has Claude, Codex, and Gemini command counterparts
- native agent name sets match across the three tools
- a renamed or removed asset leaves no dangling mention, and the CLAUDE.md Knowledge Base Structure
  section plus the MEMORY.md index reflect the current asset set
- a `description` still matches what its body does after edits

## Report

Lead with the script's counts, then the judgment findings grouped by axis, worst first. For each give
the file, the quoted text, and which criterion it fails. Close with the weakest axis and the single
highest-value next change.
