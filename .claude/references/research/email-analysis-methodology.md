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

## Summary
{2-3 sentence overview of the newsletter's key themes}

## Research Papers

| Paper | Authors | PDF | Key Contribution |
|-------|---------|-----|-----------------|
| [Title](url) | First Author et al. | [downloaded](backlog/filename.pdf) or failed | One-sentence practitioner-focused summary |

## Open Source Repositories

| Repository | Stars | Language | Description |
|-----------|-------|----------|-------------|
| [name](url) | N | Lang | One-sentence description |

## Products & Tools

| Name | Link | Description |
|------|------|-------------|
| Name | [link](url) | One-sentence description |

## Blog Posts & Articles

| Title | Source | Summary |
|-------|--------|---------|
| [Title](url) | Blog/Org name | One-sentence summary |

## Stats
- Total links processed: N
- Skipped (seen): N
- Papers downloaded: N/M
- Repos: N | Products: N | Blog posts: N
```

## Guidelines

- **Omit empty sections** — if no repos found, skip that section entirely
- **Practitioner-focused** — descriptions should answer "why should I care?"
- **De-duplicate** — same URL appearing in multiple newsletter sections counts once
- **Download failures** — note in the PDF column (e.g., "failed: 403"), don't retry
- **Ambiguous categorization** — prefer the more specific category (paper > blog)
