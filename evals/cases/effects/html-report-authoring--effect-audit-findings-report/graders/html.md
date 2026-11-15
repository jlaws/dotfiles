---
type: llm
focus: { source: file, path: report.html }
---

All claims are required.
PASS if the file loads no external resources: no http(s) src or href for scripts, stylesheets, fonts, or images, and no remote @import.
PASS if it has a table of contents of in-page links whose href targets match section ids that exist in the file.
PASS if the inline SVG has role="img" and an aria-labelledby that points to ids on a title and desc inside the SVG, with the title as the first child of the svg element.
PASS if all four findings appear with their severities distinguished visually (for example color-coded cards) and the critical finding comes first.
FAIL if the file is truncated or a placeholder stands in for any of the findings or the diagram.
