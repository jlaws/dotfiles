---
description: "Analyze an ML/AI research paper: structured summary, critical evaluation, and reviewer-style assessment using the three-pass method."
---

Before invoking the skill, perform resource discovery:

1. **Resolve the paper**: If given a title or arxiv ID (not a URL), search the web to find the canonical paper page (arxiv abs, conference proceedings, or publisher page).
2. **Fetch the paper content**: Retrieve the full paper text for analysis.
3. **Gather resources** — search for each of these:
   - PDF link (direct download URL)
   - Official code repository (check paper footer, abstract, author pages)
   - PapersWithCode page
   - Community implementations (GitHub search by paper title; note framework for each)
   - Video presentations or talks (conference talks, author walkthroughs)
   - Blog posts or explainers (author blog, Distill, popular ML blogs)
   - Supplementary materials (appendices, datasets, project pages)
   - BibTeX citation (from arxiv, Semantic Scholar, or publisher)

Pass all discovered resources when invoking the skill so they can populate the Resources & Links section.

Invoke the research:paper-analysis-methodology skill and use it to analyze: $ARGUMENTS
