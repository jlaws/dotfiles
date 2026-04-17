---
name: j-email-analysis
description: "Process AI research newsletter emails (TLDR AI, The Batch, Import AI, etc.) — extract links, categorize, download paper PDFs, and produce a structured summary. Use when processing research newsletters. Do NOT use for non-research emails."
argument-hint: "<email-body-file>"
---

Load skill `analysis-output-patterns` for output structure rules.

Before producing output, perform pre-processing:

1. **Parse arguments**: The argument is the file path to the email body (read the file to extract content).
2. **Load seen links**: First check if the email body file contains a `<!-- SEEN LINKS (already processed, skip these):` header block — if present, parse URLs from that block (these are pre-filtered by the caller and only contain links relevant to this email). Only if NO embedded header is found, fall back to reading `.seen-links` from the current working directory. All seen URLs will be skipped during processing.
3. **Extract all links** from the email body. De-duplicate within this run.
4. **Filter**: Remove any URLs present in the seen-links file. This applies to ALL URLs encountered during processing — including redirect/resolved URLs discovered when fetching. If a redirect target is already in seen-links, skip that link entirely and exclude it from the report.
5. **Exclude dangerous/administrative URLs** — silently discard, never navigate to or fetch:
   - Unsubscribe links (URLs containing `unsubscribe`, `opt-out`, `manage-subscription`, `email-preferences`)
   - Tracking pixels / beacon URLs
   - `mailto:` links
   - Generic email-client links (e.g., "view in browser")
6. **Categorize** each remaining link:
   - **Papers**: arxiv.org, openreview.net, direct PDF links, semanticscholar.org
   - **Repositories**: github.com, gitlab.com, huggingface.co (model/dataset/space repos)
   - **Products & Tools**: product launches, API announcements, SaaS tools
   - **Blog Posts**: everything else (articles, tutorials, news posts)
7. **Download paper PDFs** to `backlog/` (relative to cwd), naming: `{first-author}_{year}_{short-title}.pdf`. Create `backlog/` if it doesn't exist.
8. **Gather repo metadata**: stars, primary language, description (via GitHub API or web fetch).

---

## Email Analysis Methodology

Process AI research newsletter emails into structured, actionable summaries with downloaded papers and categorized links.

### Pipeline

| Step | Action | Details |
|------|--------|---------|
| 1. Parse | Extract email metadata | Subject, sender, date, newsletter name |
| 2. Extract | Collect all URLs | De-duplicate within run, filter against seen-links file |
| 3. Categorize | Classify each link | See categorization rules below |
| 4. Download | Fetch paper PDFs | Save to `backlog/` with standardized naming |
| 5. Summarize | Produce structured output | One section per category, tables for papers/repos |

### Categorization Rules

| Category | URL Patterns | Notes |
|----------|-------------|-------|
| Papers | `arxiv.org`, `openreview.net`, `*.pdf`, `semanticscholar.org` | Convert arxiv abs URLs to PDF URLs for download |
| Repositories | `github.com`, `gitlab.com`, `huggingface.co` (repos/models/datasets/spaces) | Fetch stars, language, description |
| Products & Tools | Product launches, API announcements, SaaS tools, platform updates | Identify by context (not URL pattern alone) |
| Blog Posts | Articles, tutorials, news, opinion pieces — everything else | Includes substacks, medium, org blogs |

### PDF Download Convention

- Directory: `backlog/` relative to cwd (create if missing)
- Naming: `{first-author-last-name}_{year}_{short-title}.pdf`
  - `short-title`: lowercase, hyphens, max 5 words (e.g., `attention-is-all-you-need`)
  - `year`: 4-digit publication year
  - Example: `vaswani_2017_attention-is-all-you-need.pdf`
- For arxiv: convert `arxiv.org/abs/XXXX.XXXXX` to `arxiv.org/pdf/XXXX.XXXXX.pdf`
- Skip download if file already exists in `backlog/`

### Output Template

```markdown
# Newsletter Analysis: {newsletter-name} — {date}

## Summary
{1-2 sentence high-level overview}

## Topic Overview

### {Topic 1 Name}
{2-3 sentences: what this topic covers, why it matters, which links below relate to it}

### {Topic 2 Name}
{2-3 sentences: what this topic covers, why it matters, which links below relate to it}

{...repeat for each major theme in the newsletter}

## Research Papers

| Paper | Authors | URL | PDF | Key Contribution |
|-------|---------|-----|-----|-----------------|
| Title | First Author et al. | https://arxiv.org/abs/... | [downloaded](backlog/filename.pdf) or failed | One-sentence practitioner-focused summary |

## Open Source Repositories

| Repository | URL | Stars | Language | Description |
|-----------|-----|-------|----------|-------------|
| name | https://github.com/... | N | Lang | One-sentence description |

## Products & Tools

| Name | URL | Description |
|------|-----|-------------|
| Name | https://example.com/... | One-sentence description |

## Blog Posts & Articles

| Title | URL | Source | Summary |
|-------|-----|--------|---------|
| Title | https://example.com/... | Blog/Org name | One-sentence summary |

## New Links
- https://example.com/link1
- https://example.com/link2
- https://example.com/resolved-redirect-url
{all unique URLs from this analysis, one per line — includes both original and resolved/redirect URLs discovered during fetching; these are appended to the seen-links file}

## Stats
- Total links processed: N
- Skipped (seen): N
- Papers downloaded: N/M
- Repos: N | Products: N | Blog posts: N
```

### Guidelines

- **Every item must include its source URL** — no entry should lack a clickable link
- **Omit empty sections** — if no repos found, skip that section entirely
- **Practitioner-focused** — descriptions should answer "why should I care?"
- **De-duplicate** — same URL appearing in multiple newsletter sections counts once
- **Download failures** — note in the PDF column (e.g., "failed: 403"), don't retry
- **Redirect tracking** — when fetching URLs (PDFs, repo metadata), record any redirect/resolved URLs. Check each against seen-links; if already seen, skip entirely. Both original and final resolved URLs count as "seen" and appear in the New Links section
- **Ambiguous categorization** — prefer the more specific category (paper > blog)

---

Apply the above methodology to produce the structured output for: $ARGUMENTS

After producing the structured output, perform these final steps:

9. **Extract title**: Get the email subject line or first heading from the email body. Slugify it (lowercase, hyphens, no special chars, max 8 words).
10. **Write report**: Save the final report to `emails/YYYY-MM-DD-{slugified-title}.md` (relative to cwd, using today's date). Create `emails/` directory if it doesn't exist.
11. **Append seen links**: Append all new unique URLs to `.seen-links` in the current working directory (one URL per line). Create the file if it doesn't exist. This includes:
    - All URLs extracted from the email (after de-duplication and exclusion filtering)
    - Any redirect/resolved URLs discovered during fetching (e.g., arxiv abs -> pdf, shortened URLs -> final destination)
    - Do NOT append excluded/dangerous URLs (unsubscribe, tracking, mailto, etc.)
12. **Confirm**: Print the file path of the written report and the number of URLs appended to the seen-links file.
