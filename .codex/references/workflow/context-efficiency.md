# Context Efficiency Best Practices

Actionable patterns for minimizing token waste and maximizing context window effectiveness in Codex workflows.

## U-Shaped Attention

LLMs attend most strongly to content at the **beginning** and **end** of the context window. Content in the middle gets lower attention weight ("lost in the middle" effect).

| Placement | Content Type |
|-----------|-------------|
| Beginning | Critical rules, constraints, "never do X" instructions |
| Middle | Reference material, examples, supporting details |
| End | Current task, most recent instructions, action items |

**Implications for AGENTS.md**: Put #1 rules and behavioral defaults at top. Put communication style and knowledge base structure (reference material) toward the middle. Current task context naturally lands at the end via conversation.

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

## Data Cleaning

External content (web pages, logs, API responses) carries significant bloat. Clean before injecting into context.

| Source | Strip | Keep |
|--------|-------|------|
| Web pages | Nav, ads, sidebars, footers, scripts | Article body, code blocks, headings |
| Logs | Repetitive entries, stack frame noise | First occurrence, root cause lines |
| API responses | Metadata, pagination, null fields | Relevant data fields |
| Documentation | Boilerplate headers, version badges | Content sections, examples |

**HTML to Markdown conversion reduces tokens 2-3x.** WebFetch does this automatically; when processing raw HTML, strip tags before reasoning.

## Context Budget

| Layer | Line Limit | Review Cadence |
|-------|-----------|---------------|
| AGENTS.md (global) | <150 lines | Quarterly |
| AGENTS.md (project) | <150 lines | Monthly |
| Skills | 150-300 lines | On modification |
| References | 200-400 lines | On modification |

**Over budget?** Extract detail into a reference file and link to it. Never inline >50 lines of reference material into AGENTS.md or skills.

## Subagent Context Isolation

Subagents get their own context window. Use this to protect the main thread from research bloat.

| Pattern | Benefit |
|---------|---------|
| Focused dispatch | Give subagent a specific question, not "explore everything about X" |
| Result summarization | Subagent returns summary, not raw findings |
| Type-appropriate agents | Use Explore for search, general-purpose for implementation |
| Single-turn design | Design tasks so subagent completes in one turn, avoiding context accumulation |

## Parallel Tool Calls

Each sequential tool call is a round trip. Batch independent operations to reduce turns.

```markdown
# Bad: 3 sequential turns
Read file A → Read file B → Read file C

# Good: 1 turn
Read file A + Read file B + Read file C (parallel)
```

**When to parallelize**: operations with no data dependencies between them.
**When NOT to**: one result informs the next call's parameters.

## AGENTS.md as Stable Prefix

AGENTS.md content is prepended to every conversation. Identical content across sessions improves reuse and keeps instruction overhead stable.

**Avoid in AGENTS.md:**
- Timestamps or dates that change per session
- Counters or metrics that update frequently
- Dynamic content that varies between conversations

**Keep in AGENTS.md:**
- Stable conventions, rules, preferences
- Static file maps and architecture descriptions
- Permanent workflow instructions

## Compaction-Friendly Patterns

When context pressure builds, Codex may compact earlier conversation turns. Structure your work to survive compaction gracefully.

| Pattern | Why |
|---------|-----|
| Write findings to files | Files persist; conversation memory doesn't |
| Append-only progress notes | Each step is independently meaningful |
| HANDOFF.md before pressure | Capture full state before compaction degrades it |
| Small, frequent commits | Git log preserves decision history |

## Cross-References

- **skill:code-agent-meta-patterns** — AGENTS.md design, context management, subagent orchestration
- **skill:session-handoff** — handoff file creation before context pressure
- **skill:multi-agent-development** — team coordination and context isolation
- **reference:llm-application-patterns** — token reduction in LLM applications
