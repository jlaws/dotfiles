---
type: llm
focus: { source: file, path: report.md }
---

All claims are required.
PASS if the empty-list division by zero is reported with a path:line for each of the three places it occurs: mean in calc/ops.py, and variance and mean_abs_dev in calc/stats.py (as one finding listing all three locations, or as separate findings).
PASS if the report lists, with its path:line, that __all__ in calc/stats.py names median, which the module does not define.
FAIL if any of those four locations is missing, or is given only as a file or function name without a line number.
PASS if the report does not call variance dividing by len(xs) rather than len(xs) - 1 a defect; its docstring says population variance, so that division is correct.
