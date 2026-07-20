# References

Sources of inspiration and borrowed patterns behind this repo's dotfiles, configs, and agent knowledge base. Vendored third-party files retain their own license headers; the repo `LICENSE` covers original work.

## Dotfiles lineage

- **mathiasbynens/dotfiles** - https://github.com/mathiasbynens/dotfiles - macOS `defaults write` patterns in `setup.sh`, `.gitconfig` alias style, `.wgetrc`, and heritage reference comments. Direct signal: `.gitattributes` (line 3).
- **Solarized (Ethan Schoonover)** - https://ethanschoonover.com/solarized - vim colorscheme (`colorscheme solarized` in `.vimrc`); vendored as `.vim/colors/solarized.vim` under its own MIT header.
- **oh-my-zsh** - https://github.com/ohmyzsh/ohmyzsh - zsh framework loaded by `.zshrc`.

## Agent knowledge base & tooling

The agent KB (`.agents/` skills and references shared by Codex and Gemini, plus tool-native Claude, Codex, and Gemini assets) borrows patterns from the sources below, grouped by theme.

### Skill system & authoring

- **superpowers (obra / Jesse Vincent)** - https://github.com/obra/superpowers - skill-system conventions (`SKILL.md` structure, `superpowers:skill-name` cross-references) and the git-worktree workflow; plus spec-first-then-quality review ordering, the cannot-verify-from-diff reviewer flag, clean-baseline-before-work, and the junior-engineer plan-clarity bar in `subagent-driven-development`, `using-git-worktrees`, and `writing-plans`.
- **Agent Skills specification (agentskills.io)** - https://agentskills.io/specification - the `.agents/` KB follows this spec (`SKILL.md` + YAML frontmatter, `cmd-j-*` command skills, and reusable workflows). Cited in `CLAUDE.md` and `.agents/README.md`.
- **ctx (stevesolun)** - https://github.com/stevesolun/ctx - per-session context budget, minimal-bundle skill selection, and skill-rot staleness detection in `skill-lookup-discipline` and `skill-audit`.
- **Skilgen (skilgen)** - https://github.com/skilgen/skilgen - the 4-axis skill-system health rubric (Groundedness/Coverage/Freshness/Structure) and groundedness authoring rule in `skill-audit`.
- **Agent SOP (strands-agents)** - https://github.com/strands-agents/agent-sop - RFC 2119 keyword discipline and the tiered artifact-output convention (summary/planning/tasks/scratchpad) in `writing-skills` and the harness configs.
- **everything-claude-code / ECC (affaan-m)** - https://github.com/affaan-m/everything-claude-code - the config-security audit of agent files and env-var hook runtime profiles in the `config-security-audit` skill and `references/workflow/hook-patterns.md`.
- **claude-codex-settings (fcakyon)** - https://github.com/fcakyon/claude-codex-settings - the agent anti-patterns self-audit checklist in `code-agent-meta-patterns` and the PreCompact context-preservation hook pattern.

### Review, verification & debugging

- **Claude Code dynamic workflows (Anthropic docs)** - https://code.claude.com/docs/en/workflows - orchestration patterns in `dispatching-parallel-agents` and the harness configs: concurrency/total-spawn caps, gauge-cost-on-a-slice, adversarial cross-check, stop-on-no-progress, and the multi-angle + per-claim-voting deep-research method in `references/research/literature-review.md`.
- **Anthropic defending-code-reference-harness** - https://github.com/anthropics/defending-code-reference-harness - the staged recon-find-verify-dedupe-report-patch loop, execution-verify N/N, independent grader agents, and the re-attack patch-validation gate in `references/security/vulnerability-review-pipeline.md`.
- **AI Code Review When Models Debate (Milvus)** - https://milvus.io/blog/ai-code-review-gets-better-when-models-debate-claude-vs-gemini-vs-codex-vs-qwen-vs-minimax.md - the adversarial debate mode and evidence-citation (file:line, reason-for-position-change) rules in `code-review-patterns` and `cmd-j-diff-review`.
- **Claude-Codex Skills (levnikolaevich)** - https://github.com/levnikolaevich/claude-code-skills - the independent-review panel, PASS/CONCERNS/FAIL/BLOCKED verdict grammar, weighted evidence hierarchy, and read-only reviewer contract in `code-review-patterns` and `verification-before-completion`.
- **Distributed Systems Testing Skills (shenli)** - https://github.com/shenli/distributed-system-testing - the richer verdict taxonomy, no-silent-passes rule, and SUT/harness/checker/environment blame classification in `verification-before-completion` and `debugging-methodology`.
- **Claude Autoresearch (uditgoenka)** - https://github.com/uditgoenka/autoresearch - mechanical-verification-only, fresh-post-action evidence, and baseline-worktree regression gating in `verification-before-completion` and the security review pipeline.
- **Clawd Cursor (AmrDab)** - https://github.com/AmrDab/clawd-cursor - the fresh-post-action-evidence / DEVIATION rule in `verification-before-completion` and the Allow/Confirm/Block PreToolUse gate in `references/workflow/hook-patterns.md`.
- **dev-browser (SawyerHood)** - https://github.com/SawyerHood/dev-browser - the observe-then-proceed UI verification rule (screenshot / accessibility-tree snapshot) in `verification-before-completion`.

### Context & token efficiency

- **Caveman (juliusbrussee)** - https://github.com/juliusbrussee/caveman - the "compress speech not thought" framing, byte-for-byte code/command/error preservation, and no-invented-abbreviations rule in the harness Output Formatting.
- **RTK Rust Token Killer (rtk-ai)** - https://github.com/rtk-ai/rtk - the command-output-shaping strategies (noise removal, grouping, truncation, dedup) and verbose-to-compact command mapping in `references/workflow/context-efficiency.md`.
- **squeez (claudioemmanuel)** - https://github.com/claudioemmanuel/squeez - the reversible-summarization (persist full output, cite path) and net-win compression gate in `references/workflow/context-efficiency.md`.
- **context-mode (mksglu)** - https://github.com/mksglu/context-mode - the fixed HANDOFF/compaction snapshot schema and progressive-checkpoint cadence in `session-handoff`.
- **Token Savior (Mibayy)** - https://github.com/Mibayy/token-savior - the bash-output-compaction and symbol-first/delta-read navigation guidance in `references/workflow/context-efficiency.md`.
- **claude-token-optimizer (nadimtuhin)** - https://github.com/nadimtuhin/claude-token-optimizer - the tiered always-on-core vs on-demand model behind the working-set context-budget rule.
- **Semantic Anchors (llm-coding)** - https://llm-coding.github.io/Semantic-Anchors - the name-the-methodology-instead-of-explaining rule (and naming BLUF) in the harness Context Efficiency guidance.
- **Skill Codex (skills-directory)** - https://github.com/skills-directory/skill-codex - non-TTY CLI invocation hygiene (`</dev/null`, redirect noise) and the effort-to-timeout heuristic in the harness Bash guidance.

### Research & documents

- **OpenDataLoader** - https://opendataloader.org/ - deterministic-parse-first document ingestion, multi-column reading-order caveats, table-fidelity checks, and page-anchored claims in `references/research/paper-analysis-methodology.md`.
- **PaperOrchestra (Ar9av)** - https://github.com/Ar9av/PaperOrchestra - the citation-verification method (fuzzy match + orphan detection) and anti-gaming self-review halt in `references/research/literature-review.md`.
- **grill-me skill (Matt Pocock, aihero.dev)** - https://www.aihero.dev/my-grill-me-skill-has-gone-viral - the recommend-an-answer and research-before-asking clarifying-question rules in the harness configs and `design-first`.

### MCP & integrations

- **Awesome MCP Servers (punkpeye)** - https://github.com/punkpeye/awesome-mcp-servers - MCP server selection/transport/vetting taxonomy in `references/architecture/mcp-client-configuration.md`.
- **Configuring MCP servers in Claude Code (Builder.io)** - https://builder.io/blog/claude-code-mcp-servers - MCP config scopes/precedence, env-var credential interpolation, and tool-search token reduction in `references/architecture/mcp-client-configuration.md`.

### Style & domain guidance

- **Google Style Guides** - https://google.github.io/styleguide/ - `Source:` citations throughout `references/languages/*` (Python, Go, Swift API Design, and others).
- **Keep a Changelog** - https://keepachangelog.com/ - changelog format used in `references/documentation/changelog-patterns.md`.
- **Hemingway writing style (Cole Schafer)** - https://www.coleschafer.com/blog/ernest-hemingway-writing-style - prose-style guidance ("Writing style (Hemingway)" blocks) in the harness configs and `references/documentation/technical-writing-for-devtools.md`.

## Vendored third-party files

Bundled verbatim; each retains its original attribution and license header.

- `.vim/colors/solarized.vim` - Ethan Schoonover (MIT). Also references tpope/vim-pathogen (https://github.com/tpope/vim-pathogen) and urso/dotrc.
- `.vim/syntax/json.vim` - Jeroen Ruigrok van der Werven.
