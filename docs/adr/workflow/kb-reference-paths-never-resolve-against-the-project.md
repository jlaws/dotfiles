---
status: accepted
topic: workflow
created: 2026-11-25
updated: 2026-11-26
---
# KB reference paths never resolve against the project

## Context

KB assets (skills, commands, agent definitions) cite shared references under a home tree:
`~/.agents/references/` for Codex and Antigravity, `~/.claude/references/` for Claude Code. Many
assets cite them by a path such as `.agents/references/<category>/<file>.md`. A tool resolves a
relative path against the session's working directory, which is the user's project. A cloned,
untrusted repository can ship a file at that same relative path, and the model then reads the
project's file as trusted KB guidance.

What the tools document, as of codex-cli 0.172.0, Claude Code 2.1.332, and agy 1.5.17:

- Every tool documents one thing only: a path inside a skill resolves against that skill's own
  folder. Sources: the agentskills.io specification; Codex `codex-rs/ext/skills/src/catalog_prompt.rs`
  ("resolve them relative to the directory containing that expanded `SKILL.md` first");
  Antigravity build-plugins ("paths relative to the skill directory"); Claude Code
  `${CLAUDE_SKILL_DIR}`.
- No tool documents how to reach a shared tree outside the skill folder, how its read tool treats
  `~`, or any base directory for commands, prompts, or agent definitions. Antigravity states it does
  not expand `~` in manifest paths.
- Reference repositories differ: superpowers reaches sibling skills with `../<skill>/...`;
  addyosmani/agent-skills copies shared references into each skill and cites them skill-relative;
  anthropics/skills duplicates shared files. None anchors on `~`.

Documentation does not settle the form, so it was measured (below) and then chosen.

## Decision Drivers

* **Must never read a project file** when an asset means the KB copy. A planted read is a prompt
  injection path, so a form that reads a planted file even once is out.
* **Must read the home copy every time**, in every tool and asset kind that ships the citation.
* **Should keep one source of truth** for each reference, so a reference edit lands once per tree.
* **Must keep shared skills byte-identical** between `.agents/skills` and
  `.gemini/antigravity-cli/skills` (`tests/test_agent_config.py`), so both trees take one form.

## Decision

Each (tree, kind) cites references in one form:

| Tree and kind | Form | Text in the installed asset |
|---|---|---|
| `.agents/skills` and `.gemini/antigravity-cli/skills` | `abs` | `<home>/.agents/references/<category>/` (repo text `~/.agents/references/<category>/`, expanded by sync) |
| `.codex/agents` | `abs` | `<home>/.agents/references/<category>/` (repo text `~/.agents/references/<category>/`, expanded by sync) |
| `.gemini/config/agents` | `abs` | `<home>/.agents/references/<category>/` (repo text `~/.agents/references/<category>/`, expanded by sync) |
| `.claude/skills` | `sibling` | `${CLAUDE_SKILL_DIR}/../../references/<category>/` |
| `.claude/commands`, `.claude/agents` | `abs` | `<home>/.claude/references/<category>/` (repo text `~/.claude/references/<category>/`, expanded by sync) |

No asset in any tree cites a reference by a path that resolves against the working directory.
Codex commands use the canonical shared `$cmd-j-*` skills. Deprecated `.codex/prompts/` copies are
no longer shipped; the prompt observations below describe the earlier measurement only.
Two exemptions keep the relative form on purpose: `config-security-audit` (it names project-scope
surfaces to audit) and the `skill-audit` copies in `.agents` and `.gemini` (they describe the
repo layout). `tests/test_agent_config.py` lists them by exact path.

Override of the literal rule (2026-11-25), made by the agent executing the plan, not by the repo
owner: the rule above gave Antigravity agents `tilde`, because the one session that did not read
home died on an account quota error (HTTP 429) before its first turn. The executing agent took
`abs` instead, for two reasons: that miss is quota, not path, and Antigravity's `read_file`
rejects a literal `~` while `abs` (home expanded by sync) needs no model expansion. `tilde` and
`abs` share the repo text `~/.agents/references/...`. `abs` for Antigravity agents is measured 2
of 3 with one quota error, not a clean 3 of 3, and is re-measured when the agy quota resets (see
Reversal Conditions).

## Rationale

The rule: a form qualifies for a (tool, kind) when it returned the home token (or the bundled
token) in 3 of 3 fresh sessions and no planted token in any session, counting every tool output in
the session, not just the final answer. Take the first qualifying form in the order `sibling`,
`abs`, `bundled`, `tilde` for skills, and `abs`, `tilde` for commands, prompts, and agent
definitions.

- Codex skills: `sibling` qualified first. Antigravity skills: `sibling` and `bundled` read planted
  files, so `abs` qualified first. The shared skills must stay byte-identical across the two trees,
  so both take `abs`, the first form that qualified in both.
- Codex agents: `abs` qualified. At measurement time, Codex prompts could not run headless and took
  the Codex agent result; those prompt copies have since been retired.
- Antigravity agents: `abs` read home in 2 of 3; the third session died on an account quota error
  (HTTP 429) before its first turn, which is not a path failure but still misses the 3-of-3 bar.
  By the literal rule `tilde` qualified, but the executor chose `abs` (see Decision): the miss is
  quota, and a literal `~` is rejected by `read_file`. The quota does not reset for about seven
  days, so `abs` was not re-measured.
- Claude skills: `sibling` qualified first. Claude commands and agents: `abs` qualified first.
- Every `cwd` control read the planted project file in 3 of 3 sessions, in every tool and kind.
  That confirms the premise: the current relative form is capturable today.

Measured 2026-11-25 with codex-cli 0.172.0 (`-m gpt-6-luna`; the pinned default model is not
offered on the account; the parent picked `gpt-5.6-terra` for spawned agents), agy 1.5.17 (Gemini 4
Hydrogen High; `--mode auto` unavailable on the account, default mode used), and Claude Code
2.1.332 (`--model haiku`). Sessions ran from a scratch project `<R>/a/b/work` holding a planted
copy at every relative path a form could resolve to. "home", "planted", and "errors" count
sessions out of 3.

| Tool | Kind | Form | Home | Planted | Errors | Path the tool call carried |
|---|---|---|---|---|---|---|
| Codex | skill | `cwd` | 0 | 3 | 0 | `cat .agents/references/...` |
| Codex | skill | `sibling` | 3 | 0 | 0 | `cat <home>/.agents/references/...` (resolved against the skill folder) |
| Codex | skill | `abs` | 3 | 0 | 0 | `cat <home>/.agents/references/...` |
| Codex | skill | `tilde` | 3 | 0 | 0 | `cat ~/.agents/references/...` (shell expands) |
| Codex | skill | `bundled` | 3 | 0 | 0 | `cat <home>/.agents/skills/<skill>/references/...` |
| Codex | agent | `cwd` | 0 | 3 | 0 | `cat .agents/references/...` |
| Codex | agent | `abs` | 3 | 0 | 0 | `cat <home>/.agents/references/...` |
| Codex | agent | `tilde` | 3 | 0 | 0 | `cat ~/.agents/references/...` |
| Antigravity | skill | `cwd` | 0 | 3 | 0 | `<project>/.agents/references/...` |
| Antigravity | skill | `sibling` | 3 | 2 | 0 | `<home>/.gemini/antigravity-cli/references/...` denied, then guesses including `<project>/../../references/...` |
| Antigravity | skill | `abs` | 3 | 0 | 0 | `<home>/.agents/references/...` |
| Antigravity | skill | `tilde` | 3 | 0 | 0 | `<home>/.agents/references/...` (model expanded `~`) |
| Antigravity | skill | `bundled` | 3 | 2 | 0 | `<home>/.gemini/antigravity-cli/skills/<skill>/references/...` plus probes of `<project>/references/...` |
| Antigravity | agent | `cwd` | 0 | 3 | 0 | `<project>/.agents/references/...` |
| Antigravity | agent | `abs` | 2 | 0 | 1 (quota) | `<home>/.agents/references/...` |
| Antigravity | agent | `tilde` | 3 | 0 | 0 | `<home>/.agents/references/...`; once literal `~/...` first, rejected by `read_file` as not absolute |
| Claude | skill | `cwd` | 0 | 3 | 0 | `<home>/.claude/skills/<skill>/references/...` (missing), then `<project>/references/...` |
| Claude | skill | `sibling` | 3 | 0 | 0 | `<home>/.claude/skills/<skill>/../../references/...` |
| Claude | skill | `abs` | 3 | 0 | 0 | `<home>/.claude/references/...` |
| Claude | skill | `tilde` | 3 | 0 | 0 | `<home>/.claude/references/...` (model expanded `~`) |
| Claude | skill | `bundled` | 3 | 0 | 0 | `<home>/.claude/skills/<skill>/references/...` |
| Claude | command | `cwd` | 0 | 3 | 0 | `.claude/references/...` |
| Claude | command | `abs` | 3 | 0 | 0 | `<home>/.claude/references/...` |
| Claude | command | `tilde` | 3 | 0 | 0 | `<home>/.claude/references/...` (model expanded `~`) |
| Claude | agent | `cwd` | 0 | 3 | 0 | `<project>/.claude/references/...` |
| Claude | agent | `abs` | 3 | 0 | 0 | `<home>/.claude/references/...` |
| Claude | agent | `tilde` | 3 | 0 | 0 | `<home>/.claude/references/...` (model expanded `~`) |

No session read the plant under a directory literally named `~`.

Two Antigravity behaviours explain its failures. Its `read_file` denies `~/.gemini/antigravity-cli/references/`
("Matches hardcoded system protection boundary rule"), so a sibling tree beside the Antigravity
skills cannot be read at all. And the skill body reaches the model inline, without its folder, so a
skill-relative path leaves the model guessing; it then probes project paths.

## Consequences

**Gained**: A project file at `.agents/references/...` or `references/...` can no longer stand in
for KB guidance in any measured tool and kind. Claude skills cite the sibling tree through a
harness-substituted variable, with no sync rewrite.

**Accepted**: A `-p <project>` install expands the anchors to that project's synced references
copy, and each sync overwrites that copy from the repo. The `abs` forms need sync to expand `~/.agents/references/` and
`~/.claude/references/` into the absolute home path in installed copies, so the installed text
differs from the repo text and uninstall must recognise the expanded file. The eval stager must
rewrite every installed form to its workspace copy. Codex skills give up `sibling`, which
qualified, to keep shared skills byte-identical with Antigravity. Antigravity agents use `abs` on
the strength of 2 of 3 home reads plus a quota failure, not a clean 3 of 3. Claude cells ran on Haiku and every Claude session
expanded `~` itself, so how Claude's Read tool treats a literal `~` is unmeasured.

## Ruled Out

| Idea | Why ruled out | When |
|------|---------------|------|
| Cite references relative to the working directory (`cwd`) | The status quo. Read the planted project file in 3 of 3 sessions in every tool and kind measured. | 2026-11-25 |
| Sibling tree beside the Antigravity skills (`../../references/`) | Antigravity denies reads under `~/.gemini/antigravity-cli/references/`; the model then guessed paths and read the planted `<project>/../../references/` copy in 2 of 3 sessions. | 2026-11-25 |
| Sibling form for Codex skills | Qualified (3 of 3 home, 0 planted), but the shared skills must be byte-identical with Antigravity, where it failed. | 2026-11-25 |
| Bundle references into each skill folder (`bundled`) | Antigravity read the planted `<project>/references/` copy in 2 of 3 sessions, because the model is not told the skill folder. It also copies every cited category into every skill. | 2026-11-25 |
| Bundled or `abs` for Claude skills | Both qualified; `sibling` ranks first and needs no sync rewrite. | 2026-11-25 |
| `tilde` where `abs` qualified | Ranked after `abs`: it works only when the model expands `~` before the call, and Antigravity's `read_file` rejects a literal `~` path. | 2026-11-25 |
| `tilde` for Antigravity agents | Qualified by the literal rule (3 of 3 home) while `abs` took 2 of 3 plus a quota 429, but the model had to expand `~` itself and `read_file` rejected the literal once. Same repo text as `abs`; the executor took the installed form that needs no model expansion. | 2026-11-25 |

## Enforcement

- Owner: the citing assets under `.agents/skills`, `.codex/agents`,
  `.gemini/antigravity-cli/skills`, `.gemini/config/agents`, `.claude/skills`, `.claude/commands`,
  and `.claude/agents`; `REFERENCE_FORMS` in `tests/test_agent_config.py` maps each (tree, kind) to
  its chosen prefix.
- Pinned by: `tests/test_agent_config.py::test_no_kb_asset_cites_a_project_relative_reference`
  (every reference citation in the KB trees carries the prefix its tree and kind take) and
  `tests/test_evals.py::StageTests::test_staged_assets_cite_workspace_references` (staged eval
  copies cite `./.kb-references/` and hold no installed prefix).
- Also pinned: `tests/test_agent_config.py::test_prefixed_citations_name_a_real_reference_or_skill`
  (a short `<category>:<name>` citation names `references/<category>/<name>.md`, with any
  `#anchor` matching a heading there, and `skill:<name>` names a skill, in the citing file's own
  tree).

## Reversal Conditions

- A tool documents an anchor for its KB root (a variable or base directory for references outside
  the skill folder, or for commands and agent definitions). Re-measure that tool and prefer the
  documented anchor.
- A later CLI version changes a measured cell: a chosen form reads a planted file, or a ruled-out
  form starts to qualify. In particular, Antigravity passing the skill folder to the model, or
  allowing reads beside its skills, reopens `sibling` and a shared skill form for Codex.
- The Antigravity quota resets and its agent `abs` cell reads a planted file or misses home:
  re-measure and reconsider `tilde` for Antigravity agents.

## Related
- [Consult commands offer references and delegation](consult-commands-offer-references-and-delegation.md)
- [KB evals run on a staged plugin](kb-evals-run-on-a-staged-plugin.md)
- [Reference trees share a section set, not a body](reference-tree-section-parity.md)
