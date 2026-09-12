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
| Tables | Highest (unmeasured here; the ratio depends on the content) | Comparisons, decision matrices, option lists |
| Code blocks | High | Commands, configurations, examples |
| Bullet points | Medium | Sequential steps, short items |
| Prose paragraphs | Lowest | Explanations, nuanced reasoning |

**Rule**: If information can be a table, make it a table. Reserve prose for explanations that require nuance.

**Counter-rule, and it is the one that gets violated:** density is measured *per unit of
information carried*. Structure the question did not ask for is a net cost even in table form,
because the scaffolding invents content to fill. A "summarize/compare X vs Y" prompt answered with
headed pro/con walls and "Pick A if / Pick B if" sections is the canonical case — Chisle measured
one such answer at 173% of a no-tool baseline, then rewrote the rule and re-measured it at 93%
(n=3). claude-token-efficient found the same thing from the other direction: "Words drop more
consistently than tokens. Markdown and structure tokens partly offset word savings."

So the two halves are: use the densest format for information you have already decided to include,
and do not let the format decide what to include.

### Answer at the Question's Altitude

A question that wants a verdict gets a verdict. A question that wants a matrix gets a matrix. Do not
manufacture headings, recaps, numbered scaffolding, or decorative tables the question did not ask
for. Two tight paragraphs beat five headed sections when the reader asked "which one".

### Output Rules Earn Their Place Or Cost You

An instruction file is re-sent on every request, so a rule that changes nothing is pure overhead.
claude-token-efficient measured **Claude** baselines at **0%** incidence of preamble, sycophancy,
"as an AI", and smart quotes, and concluded: "rules targeting those behaviors carry input cost
without changing output. Trim accordingly." Do not add such a rule to a Claude-facing config
without evidence the behavior still occurs.

Scope the finding to the model it was measured on. `.claude/CLAUDE.md` carries none of these rules;
`.codex/AGENTS.md` does, and that stays, because a measurement on Claude says nothing about what
another harness's model emits. Trimming it on this evidence would be exactly the unsourced
generalization the Evidence axis forbids.

The asymmetry within the same measurement is instructive: the em-dash rule *did* move its marker
(one model went 100% to 20% incidence), which is why that rule is worth its bytes. It was not
monotonic — one cell regressed from 0% to 40% — so treat it as directional.

**A terseness ruleset loses money on short interactions.** Four sources measured this
independently: ponytail at +26.2% and +38.7% cost on two reasoning models ("the ruleset is re-sent
as input every call and the baseline output is already terse, so the input and reasoning-token
overhead outweighs the lines saved"); Chisle losing its short-coding segment 70% to 62%;
claude-token-efficient finding short prompts "roughly cost-neutral"; and a third party measuring
173 of their own sessions and finding injection overhead roughly cancelling the savings. The rules
in this file pay off on long, tool-heavy, code-bearing work. On a one-line throwaway they do not.

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

**Tool priority.** Climb only when the rung below cannot answer it:

| Rung | Tool | Use when |
|---|---|---|
| 1 | Grep | you know the symbol or string |
| 2 | Glob | you know the path shape but not the contents |
| 3 | Read with `offset`/`limit` | you have located the file and need one range |
| 4 | `Explore` | the naming convention itself is unknown, or you want judgement too |

Do not search with Bash. `find`, `grep`, and `rg` in a Bash call return unbounded output straight into
context; the dedicated tools bound it. Bash search is for what the tools cannot do — `git grep` over
history, or `git log -S`.

## Data Cleaning

External content (web pages, logs, API responses) carries significant bloat. Clean before injecting into context.

| Source | Strip | Keep |
|--------|-------|------|
| Web pages | Nav, ads, sidebars, footers, scripts | Article body, code blocks, headings |
| Logs | Repetitive entries, stack frame noise | First occurrence, root cause lines |
| API responses | Metadata, pagination, null fields | Relevant data fields |
| Documentation | Boilerplate headers, version badges | Content sections, examples |

**HTML to Markdown conversion cuts tokens substantially** (unmeasured here; depends on markup density). WebFetch does this automatically; when processing raw HTML, strip tags before reasoning.

**Fetch-tool selection (cheapest first):** WebFetch for public/static pages (does HTML to Markdown for free); the agent-browser CLI when a page is JS-rendered or behind an auth wall; `pdftotext` for PDFs rather than the Read tool, which spends vision tokens on document pages.

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

**Shaping applies to command output, never to file reads.** A `Read` result's exact bytes feed the
`old_string` of a later `Edit`, so shaping one makes you edit against text you never saw. This is a
correctness rule, not an efficiency preference, and it has measured consequences: headroom observed
that lossy-compressing file reads caused agents to re-read the same file (`cat` then `cat -A` then
`cat -n`) to recover exact detail — turn inflation — and to lose the content outright when recovery
failed. Narrow the read at the source instead (`offset`/`limit`, `sed -n`), which shrinks what you
fetch rather than what you keep.

**Two indicators, not one, before you treat output as an error.** A single keyword false-positives
on ordinary output that merely mentions the word "error" — source code, a log schema, a help text.
Requiring two distinct signals before switching to error handling avoids preserving everything and
thereby preserving nothing.

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

Five rules make that one safe. Without them it permits a summarize whose persist silently failed —
a lossy compression that believes it is lossless.

**If the original cannot be persisted, do not compress it.** Ship it verbatim and say why. The
recovery path is what makes the compression reversible, so losing the path means losing the licence
to compress. squeez states the consequence plainly: "No stash means no recovery path, so
compressing would silently destroy the dropped lines ... Fail open: ship the verbatim original."
This is `hook-patterns`' fail-open rule for transforms, applied to your own summarizing.

**Round-trip before you trust a fold.** A transformation you call lossless carries its inverse and
gets checked: if the round trip does not reproduce the original, or the result is not actually
smaller, return the original unchanged. headroom does this per call rather than claiming it once in
a doc, which is the difference between a guarantee and a promise.

**The citation counts inside the gate, not after it.** A pointer, marker, or path is overhead that
exists only because you compressed, so it belongs in the arithmetic. squeez shipped and then fixed
exactly this bug: a call saving 25 tokens emitted a ~40-token marker and still reported a win.

**A pointer has to survive compaction.** Compaction can evict the content an in-conversation
reference points at, leaving the reader a pointer to nothing. A file path does not have this
problem, which is why the rule above says persist and cite a path rather than "refer back to the
earlier output".

**Declare loss per transformation, not per tool.** Two classes, and they get different treatment:
a *reformat* packs the same information denser and needs no recovery path (stripping ANSI codes,
collapsing an identical-line run to a counted marker, minifying JSON whitespace); an *offload*
drops bytes and therefore requires one. Sorting a transformation into the wrong class is how a
lossy step gets described as safe. Note the limit of the framing: calling an offload
"information-preserving" redefines loss as *unrecoverable* rather than *changed*, and that holds
only while the store is alive and the reader actually retrieves.

**Some output is never a compression candidate.** Suspend compression, and resume after, when the
content carries a security warning, a confirmation prompt for an irreversible action, or a
multi-step sequence a fragment would put out of order. Suspend it when compressing would create
technical ambiguity, and when the user asks you to clarify or repeats a question — the repeat is
evidence the compressed form already failed. Chisle states the stopping condition as well as it can
be put: compress until the rules would delete the answer, and no further.

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
- **reference:code-efficiency-ladder** — the same economy applied to what gets built rather than what enters context
