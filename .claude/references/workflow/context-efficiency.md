# Context Efficiency Best Practices

Actionable patterns for minimizing token waste and maximizing context window effectiveness in Claude Code workflows.

## U-Shaped Attention

LLMs attend most strongly to content at the **beginning** and **end** of the context window. Content in the middle gets lower attention weight ("lost in the middle" effect).

| Placement | Content Type |
|-----------|-------------|
| Beginning | Critical rules, constraints, "never do X" instructions |
| Middle | Reference material, examples, supporting details |
| End | Current task, most recent instructions, action items |

**Implications for CLAUDE.md**: Put #1 rules and behavioral defaults at top. Put communication style and knowledge base structure (reference material) toward the middle. Current task context naturally lands at the end via conversation.

## Token Density by Format

Not all formats are equally efficient. Prefer higher-density formats when conveying structured information.

| Format | Relative Density | Best For |
|--------|-----------------|----------|
| Tables | Highest (~40% more efficient than prose) | Comparisons, decision matrices, option lists |
| Code blocks | High | Commands, configurations, examples |
| Bullet points | Medium | Sequential steps, short items |
| Prose paragraphs | Lowest | Explanations, nuanced reasoning |

**Rule**: If information can be a table, make it a table. Reserve prose for explanations that require nuance.

## Two-Phase Retrieval

Search first, read second. Never bulk-read files speculatively.

| Phase | Tool | Purpose |
|-------|------|---------|
| 1. Search | Glob, Grep | Identify which files are relevant |
| 2. Read | Read | Load only confirmed-relevant files |

**Anti-patterns:**
- Reading 10 files "just in case" when Grep could narrow to 2
- Reading an entire file when only one function is relevant (use `offset`/`limit`)
- Using Read to scan for keywords (use Grep instead)

**Delta reads:** when re-reading a file you just changed, read only the changed range (`git diff`, or Read with `offset`/`limit`) instead of the whole file. For a large unfamiliar file, read signatures/imports first, then Grep to the exact symbol and read only that range.

## Data Cleaning

External content (web pages, logs, API responses) carries significant bloat. Clean before injecting into context.

| Source | Strip | Keep |
|--------|-------|------|
| Web pages | Nav, ads, sidebars, footers, scripts | Article body, code blocks, headings |
| Logs | Repetitive entries, stack frame noise | First occurrence, root cause lines |
| API responses | Metadata, pagination, null fields | Relevant data fields |
| Documentation | Boilerplate headers, version badges | Content sections, examples |

**HTML to Markdown conversion reduces tokens 2-3x.** WebFetch does this automatically; when processing raw HTML, strip tags before reasoning.

## Command Output Shaping

Shape tool and command output before it enters context — most of it is noise.

| Tactic | Do |
|--------|-----|
| Strip noise | Remove ANSI colors, progress bars, spinners |
| Collapse passing runs | A green suite becomes one line: "142 passed, 0 failed" |
| Dedupe | Fold repeated log lines / stack frames to first occurrence + count |
| Cap large output | Write big output to a scratch file, then Grep it — do not inline megabytes |
| Prefer compact flags | `git status --porcelain`, `git log --oneline`, `ls -1`, `--quiet` |

**Hard constraint:** preserve failures, exit codes, and error strings **verbatim**. Shaping is for noise, never for evidence — `verification-before-completion`'s evidence hierarchy ranks a reproduced run as the strongest support for a claim, so the real output has to survive intact to be cited.

## Context Budget

| Layer | Line Limit | Review Cadence |
|-------|-----------|---------------|
| CLAUDE.md (global) | <150 lines | Quarterly |
| CLAUDE.md (project) | <150 lines | Monthly |
| Skills | 150-300 lines | On modification |
| References | 200-400 lines | On modification |

**Over budget?** Extract detail into a reference file and link to it. Never inline >50 lines of reference material into CLAUDE.md or skills.

## Context Isolation

When a task involves heavy research that could bloat the main context, consider these patterns:

| Pattern | Benefit |
|---------|---------|
| Focused research scope | Ask a specific question, not "explore everything about X" |
| Result summarization | Capture findings as a summary, discard raw search output |
| File-based handoff | Write findings to a scratch file rather than accumulating in conversation |
| Single-pass analysis | Complete each analysis phase fully before starting the next |

**Reversible summarization:** before you summarize or drop large output, persist the full original to a scratch file and cite its path — detail stays recoverable. Only summarize (or delegate to a subagent for context savings) when the estimated tokens saved exceed the overhead; scale compression intensity up as the window fills.

## CLAUDE.md as Stable Prefix

CLAUDE.md content is prepended to every conversation. Identical content across sessions enables KV cache hits (provider-side optimization).

**Avoid in CLAUDE.md:**
- Timestamps or dates that change per session
- Counters or metrics that update frequently
- Dynamic content that varies between conversations

**Keep in CLAUDE.md:**
- Stable conventions, rules, preferences
- Static file maps and architecture descriptions
- Permanent workflow instructions

## Compaction-Friendly Patterns

When context pressure builds, Claude Code compacts (summarizes) earlier conversation turns. Structure your work to survive compaction gracefully.

| Pattern | Why |
|---------|-----|
| Write findings to files | Files persist; conversation memory doesn't |
| Append-only progress notes | Each step is independently meaningful |
| HANDOFF.md before pressure | Capture full state before compaction degrades it |
| Small, frequent commits | Git log preserves decision history |

## Cross-References

- **skill:code-agent-meta-patterns** — CLAUDE.md design, context management
- **skill:session-handoff** — handoff file creation before context pressure
- **reference:llm-application-patterns** — token reduction in LLM applications
