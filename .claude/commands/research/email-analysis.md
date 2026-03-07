---
name: email-analysis
description: "Process AI research newsletter emails (TLDR AI, The Batch, Import AI, etc.) — extract links, categorize, download paper PDFs, and produce a structured summary. Use when processing research newsletters. Do NOT use for non-research emails."
argument-hint: "<email-body-file>"
---

Before invoking the methodology, perform pre-processing:

1. **Parse arguments**: The argument is the file path to the email body (read the file to extract content).
2. **Load seen links**: Read `.seen-links` from the current working directory (one URL per line). All URLs in this file will be skipped during processing. If the file doesn't exist, treat it as empty (no seen links).
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

Read references/research/email-analysis-methodology.md and apply its methodology to produce the structured output for: $ARGUMENTS

After producing the structured output, perform these final steps:

9. **Extract title**: Get the email subject line or first heading from the email body. Slugify it (lowercase, hyphens, no special chars, max 8 words).
10. **Write report**: Save the final report to `emails/YYYY-MM-DD-{slugified-title}.md` (relative to cwd, using today's date). Create `emails/` directory if it doesn't exist.
11. **Append seen links**: Append all new unique URLs to `.seen-links` in the current working directory (one URL per line). Create the file if it doesn't exist. This includes:
    - All URLs extracted from the email (after de-duplication and exclusion filtering)
    - Any redirect/resolved URLs discovered during fetching (e.g., arxiv abs -> pdf, shortened URLs -> final destination)
    - Do NOT append excluded/dangerous URLs (unsubscribe, tracking, mailto, etc.)
12. **Confirm**: Print the file path of the written report and the number of URLs appended to the seen-links file.
