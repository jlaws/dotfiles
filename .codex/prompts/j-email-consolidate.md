---
name: j-email-consolidate
description: "Consolidate a directory of per-email AI newsletter analysis reports into a single unified weekly digest. Use when merging emails/*.md per-email reports. Do NOT use for processing a single email (use /j-email-analysis)."
argument-hint: "<emails-dir> <output-path>"
---

Build a consolidated weekly digest from: $ARGUMENTS

Two arguments: `<emails-dir>` and `<output-path>`.

## Setup (always do first)

Load skill `output-completeness` -- the digest must account for every source with real signal; no silently dropped items, no truncated sections.

You will build the consolidated weekly digest **incrementally**: write a skeleton first, then extend it as you read each source file. Do NOT batch all reads before writing. The output file must grow monotonically and visibly as you work.

### Step 1 — Enumerate

Glob `<emails-dir>/*.md`. EXCLUDE any file whose name ends in `-summary.md`. Hold the resulting list; you will iterate through it in Step 3.

### Step 2 — Write the skeleton (your FIRST Write call)

Before reading any source files, Write `<output-path>` with this exact structure (date in the header is today's date inferred from the file basenames or the current date):

```markdown
# AI Newsletter Weekly Digest — <YYYY-MM-DD>

_Covering newsletters processed through <YYYY-MM-DD>_

## Summary
<!-- END SECTION:Summary -->

## Research Papers
<!-- END SECTION:Research Papers -->

## Models & Releases
<!-- END SECTION:Models & Releases -->

## Tools & Repos
<!-- END SECTION:Tools & Repos -->

## People & Events
<!-- END SECTION:People & Events -->

## Other Links
<!-- END SECTION:Other Links -->
```

The `<!-- END SECTION:<name> -->` lines are sentinels. They MUST remain unique anchors throughout Step 3 — never write content that duplicates a sentinel string.

### Step 3 — Iterate source files, one at a time

For each source file from Step 1:

1. **Read** the source file.
2. **Classify each item** in it into one of the six sections above. A single source file usually contributes to multiple sections.
3. **Insert each item** by calling Edit on `<output-path>`. Use `old_string = "<!-- END SECTION:<name> -->"` and `new_string = "<item-as-tight-markdown>\n<!-- END SECTION:<name> -->"`. The sentinel stays in place so subsequent Edits keep working.
4. **Merge-on-insert**: before inserting, check whether the item already exists in the summary (from a previous source file in this same iteration). If a near-duplicate is already there, either skip it or do a small Edit to enrich the existing entry — do NOT add a second copy. Prefer skipping over enriching when the new info adds nothing material.
5. **End-of-source self-check**: after the last Edit for this source, write one short text line in the form `Finished <source-basename>: inserted N, merged M, skipped K`. This is a forcing function for full enumeration, not output — it is not part of the digest.

**Critical — process one source per cycle.** Each source file gets its own complete Read → classify → Edit → self-check loop before you touch the next file. Do NOT issue multiple Read tool calls in parallel or in rapid succession across different source files, even for short scholar alerts, small product roundups, or files that look similar. The "let me batch-read several at once to speed up" shortcut is the exact failure mode this prompt is designed to prevent: it produces silent topic drops, especially in the last 20% of files. There is no time budget pressure that justifies it — process every source individually, even if there are 80+ of them.

You may interleave Read and Edit calls **within a single source's cycle**. Do not accumulate many items in memory before writing — Write/Edit as you go so the file grows visibly.

### Step 4 — Final cleanup pass

**Before starting cleanup**, scan the Glob output from Step 1 one last time. For each source file, recall what you inserted from it. If any source produced zero or near-zero insertions and is not a known low-signal type (sponsor-only mailer, off-topic non-AI content), revisit it now — re-Read it and add what you missed. Do not enter cleanup until every source has been honestly accounted for.

**Sources-touched audit.** Walk the Glob list from Step 1 one more time. For each source file, identify at least one URL or named entity from that source present in the output. If a source contributed zero entries AND is not in the known low-signal types below, re-Read it and add what you missed.

Known low-signal source types where zero contribution is acceptable:
- Pure sponsor mailers (no AI-research content beyond sponsor placements)
- "Top N AI tools" / indie product roundups with no funding or research context
- Off-topic newsletter sections (genetics, anthropology, hardware reviews, non-AI policy)

Do not use numeric section-count thresholds — bullet count is not a quality signal. The audit is per-source: did every source that had real signal contribute at least one item?

Once every source file has been read and its items inserted, do one Edit per section to:
- Tighten prose where adjacent items overlap.
- Remove the `<!-- END SECTION:<name> -->` sentinel for that section.

After the cleanup pass, the file must contain zero `<!-- END SECTION:...` strings.

---

### Content rules (apply throughout Steps 3 and 4)

- **Aggressive merging.** If two source files describe the same paper/repo/release/person/event, write ONE tight entry. Drop padding, hedging, repeated context, and near-duplicate descriptions. The goal is a scannable digest, not a concatenation.
- **Summary section structure.** The `## Summary` section must contain **two paragraphs**:
  1. **Macro week** — what dominated the week across all sources (named funding rounds, model releases, deal headlines, big policy moves). One dense paragraph.
  2. **Research/thematic synthesis** — explicitly name the cross-paper threads you saw across the week (e.g., "agent memory consolidation", "compute geopolitics", "inference efficiency", "self-improving agents"), citing 2-4 representative items per thread. This paragraph turns the rest of the digest into a navigable index, not just a list.
  Do NOT collapse to one paragraph even if the source set looks thin.
- **Terse bullets over prose.** One line per item where possible.
- **Required table formats:**
  - Research Papers: `| Paper | Authors | URL | Note |`
  - Tools & Repos: `| Name | Description |`
  These two sections MUST be rendered as markdown tables, not bullet lists. Models & Releases and Other Links remain bullets.
- **Group by theme within a section**, not by source. Reorganize freely.
- **People & Events must use these `###` subsections in this order** (omit a subsection only if it has zero entries):
  - `### Podcasts & long-form interviews`
  - `### Funding, IPOs, talent`
  - `### Conferences & workshops` (post-drop-rule filtering — only keep non-sponsor events with named speakers or technical content)
  - `### Cultural moments` — keep AI/robotics cultural appearances and demos (humanoid monk initiations, AI character symphony performances, Met Gala robot cameos, viral robot demos). These belong in the digest as personality and PR-side signal even when no model/product is announced. Still drop pure non-AI cultural content.
- **What to always drop** (no judgment needed):
  - Newsletter sponsor placements — anything tagged `(sponsor)`, `(sponsored)`, `(promoted)`, or recognizable sponsor units (Framer, Spinach AI, Granola, Welo Data, CData, Cursor (sponsor), Cal.com, PlayerZero, Unwrap, Archera, Fivetran, Viktor, Sauna AI, Plaid).
  - Newsletter housekeeping: "Superhuman Top N tools/prompts/course", "TLDR is hiring", course/seminar/luma-event promo, "request a demo" CTAs.
  - Indie product-roundup entries from "AI tools weekly" / "Top tools of the week" emails — short product names with no funding/release/research context (Vids.new, AIApply, Brik, Spellar, Hume, Gamma, Blaze, Filect, Hoogly, ContentPilots, Voqusa, TrafficClaw, Saydi, UpSynth, Adject, Demi, Streva, Clarus, Planana, APImage, Nudge, Kanwas, Bitgrain, GetThis, Open Vibe, Abstraction, etc.).
  - Off-topic content from podcasts/articles outside AI/ML (genetics, anthropology, hardware reviews, supersonic travel — keep only if a specific AI angle is named). Exception: AI/robotics cultural appearances (humanoid robots at public events, AI characters in performances, viral robot demos with no specific model release) are NOT off-topic — route them to `People & Events → Cultural moments`.
  - Background reference links cited as "see also / further reading" inside a larger summarized post (Wikipedia, GeeksForGeeks, textbook lecture notes, Towards Data Science explainers). Keep these only if they ARE the primary subject of a source file, not background reading for one.
- **What to always keep** (no judgment needed — these are the digest's whole point):
  - Named model/product/framework releases (GPT-X.Y, Gemini Z, Claude N, new open-source model, new CLI tool, new agent framework).
  - Named engineering/post-mortem posts ("How we built X", "X at scale", "How OpenAI built Y", "Inside the architecture of Z").
  - Named industry news headlines with specific actors and amounts ($XB deal, IPO, lawsuit, acquisition, regulatory action, large funding round, notable hire).
  - Named research benchmarks and named research papers — but see the **Downloaded papers** rule below.
  - Named events (announcements, RFCs, scheduled launches, major releases).
- **Downloaded papers — do not duplicate library content.** Per-email reports include a Research Papers table whose **PDF** column reads either `downloaded` or `failed: <reason>` (see `j-email-analysis` for the convention). When you encounter a Research Papers row:
  - If the PDF column reads `downloaded` → **omit the paper entirely from the digest**. Do not include its title, URL, or description in any section. Downloaded papers are already in `~/Documents/papers/`, `papers-db.json`, and `analysis/` via the ingest pipeline. The digest is a queue of items needing user attention, not a library catalog.
  - If the PDF column reads `failed: <reason>` (or any value other than `downloaded`) → keep the paper in the Research Papers section with its URL intact, and append `(needs manual download: <reason>)` in the description so the user knows to fetch it.
  - If a paper is only mentioned in prose (no Research Papers table row in this source, no clear PDF status) → **always keep the link**, even if the same title appears with `downloaded` status in some *other* source file. Prose mentions are independent signal that the paper is being discussed in the discourse, and they belong in Other Links (or wherever the prose context fits). The downloaded-paper omission rule applies ONLY to rows in a source's Research Papers table, never to free-text mentions.
- **Preserve every unique external link** (`[text](url)`) verbatim for items you decide to keep — do not strip, rewrite, or simplify URLs. Link preservation is the one place where you must NOT reduce.
- Do NOT include Skipped Links, Downloads, or admin metadata.
- **Do NOT link back to per-email report files** (no `[source](emails/...)` backlinks). The per-email files are ephemeral and will be deleted. The inline external URL on each item IS the citation.
- If an item has no external destination URL in the source, leave it without a backlink — do not invent one.

### Operational rules

- **Do not print** the consolidated report to stdout. Everything goes through Write/Edit on `<output-path>`.
- **Tools allowed**: Glob, Read, Write, Edit. Do not call any other tools.
- Do not stop until every source file from Step 1 has been read AND the Step 4 cleanup pass has removed every sentinel.
