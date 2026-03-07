---
name: email-analysis
description: "Process AI research newsletter emails (TLDR AI, The Batch, Import AI, etc.) — extract links, categorize, download paper PDFs, and produce a structured summary."
argument-hint: "<seen-links-file> <email-body-or-source...>"
---

Before invoking the methodology, perform pre-processing:

1. **Parse arguments**: First arg is path to a seen-links file (one URL per line, read-only — another system handles appending). Remaining args are the email content (pasted text, file path, or URL).
2. **Load seen links**: Read the seen-links file. All URLs in this file will be skipped during processing.
3. **Extract all links** from the email body. De-duplicate within this run.
4. **Filter**: Remove any URLs present in the seen-links file.
5. **Categorize** each remaining link:
   - **Papers**: arxiv.org, openreview.net, direct PDF links, semanticscholar.org
   - **Repositories**: github.com, gitlab.com, huggingface.co (model/dataset/space repos)
   - **Products & Tools**: product launches, API announcements, SaaS tools
   - **Blog Posts**: everything else (articles, tutorials, news posts)
6. **Download paper PDFs** to `backlog/` (relative to cwd), naming: `{first-author}_{year}_{short-title}.pdf`. Create `backlog/` if it doesn't exist.
7. **Gather repo metadata**: stars, primary language, description (via GitHub API or web fetch).

Read references/research/email-analysis-methodology.md and apply its methodology to produce the structured output for: $ARGUMENTS
