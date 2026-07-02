# Email Analysis Methodology

Process AI research newsletter emails into structured, actionable summaries with downloaded papers and categorized links.

## Pipeline

| Step | Action | Details |
|------|--------|---------|
| 1. Parse | Extract email metadata | Subject, sender, date, newsletter name |
| 2. Extract | Collect all URLs | De-duplicate within run, filter against seen-links file |
| 3. Categorize | Classify each link | See categorization rules below |
| 4. Download | Fetch paper PDFs | Save to `backlog/` with standardized naming |
| 5. Summarize | Produce structured output | One section per category, tables for papers/repos |

## Categorization Rules

| Category | URL Patterns | Notes |
|----------|-------------|-------|
| Papers | `arxiv.org`, `openreview.net`, `*.pdf`, `semanticscholar.org` | Convert arxiv abs URLs to PDF URLs for download |
| Repositories | `github.com`, `gitlab.com`, `huggingface.co` (repos/models/datasets/spaces) | Fetch stars, language, description |
| Products & Tools | Product launches, API announcements, SaaS tools, platform updates | Identify by context (not URL pattern alone) |
| Blog Posts | Articles, tutorials, news, opinion pieces — everything else | Includes substacks, medium, org blogs |

## PDF Download Convention

- Directory: `backlog/` relative to cwd (create if missing)
- Naming: `{first-author-last-name}_{year}_{short-title}.pdf`
  - `short-title`: lowercase, hyphens, max 5 words (e.g., `attention-is-all-you-need`)
  - `year`: 4-digit publication year
  - Example: `vaswani_2017_attention-is-all-you-need.pdf`
- For arxiv: convert `arxiv.org/abs/XXXX.XXXXX` to `arxiv.org/pdf/XXXX.XXXXX.pdf`
- Skip download if file already exists in `backlog/`

## Output Template

```markdown
# Newsletter Analysis: {newsletter-name} — {date}

**Source email:** {email subject line}

## Summary
{1-2 sentence high-level overview — every mention of a specific paper, repo, product, or article must include an inline `[text](url)` link to the external URL the email cited}

## Topic Overview

### {Topic 1 Name}
{2-3 sentences: what this topic covers, why it matters, with inline `[text](url)` links to the specific articles/papers/repos it covers, e.g. "covers [X's new release](https://...) and [the accompanying paper](https://arxiv.org/...)"}

### {Topic 2 Name}
{2-3 sentences with inline links as above}

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

## Guidelines

- **Every item must include its source URL** — no entry should lack a clickable link
- **Inline-link every prose claim that references a specific item** — every mention of a paper, repo, product, or article in `## Summary` and `## Topic Overview` MUST include an inline markdown `[text](url)` pointing to the external URL that the email cited. Prose without links is not acceptable when a specific item is being discussed.
- **No redundant links in table description cells** — tables (Research Papers, Open Source Repositories, Products & Tools, Blog Posts & Articles) already have a URL column. Do not embed a second link inside the same row's description/summary cell. Every *other* cell or prose sentence needs a link.
- **Omit empty sections** — if no repos found, skip that section entirely
- **Practitioner-focused** — descriptions should answer "why should I care?"
- **De-duplicate** — same URL appearing in multiple newsletter sections counts once
- **Download failures** — note in the PDF column (e.g., "failed: 403"), don't retry
- **Redirect tracking** — when fetching URLs (PDFs, repo metadata), record any redirect/resolved URLs. Check each against seen-links; if already seen, skip entirely. Both original and final resolved URLs count as "seen" and appear in the New Links section
- **Ambiguous categorization** — prefer the more specific category (paper > blog)
