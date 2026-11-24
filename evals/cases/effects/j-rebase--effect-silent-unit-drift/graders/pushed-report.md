---
type: llm
focus: { source: file, path: pushed-report.txt }
---

All claims are required.
PASS if share_line in pushed-report.txt scales the fraction that pct now returns to a 0 to 100 percentage before formatting it (for example by multiplying by 100, or with a percent format such as {:.1%}), so that share_line('cpu', 1, 4) gives 'cpu: 25.0%'.
PASS if share_line still returns a string of the form 'label: NN.N%'.
FAIL if share_line formats pct's result unscaled with %.1f%%, which gives 'cpu: 0.2%' for share_line('cpu', 1, 4).
