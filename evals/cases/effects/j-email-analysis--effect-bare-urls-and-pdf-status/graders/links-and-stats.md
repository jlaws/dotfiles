---
type: llm
focus: { source: file, path: emails/report.md }
---

All claims are required.
PASS if all five newsletter URLs (two papers, the OpenGrader release, the blog post, the CodeFix-2k dataset) appear in the report as markdown links.
PASS if every paper, repo, tool, or article named in the Summary and Topic Overview prose has an inline markdown link.
FAIL if any table description or summary cell contains a second link in the same row as the row's URL column.
PASS if the Stats section reports five total links processed and zero of two papers downloaded.
