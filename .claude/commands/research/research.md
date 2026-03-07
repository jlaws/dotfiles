---
name: research
description: "Research consultation — literature review, statistical analysis, and paper writing. Use when conducting research, reviewing literature, or writing academic papers. Do NOT use for simple factual questions (web search instead)."
argument-hint: "<question-or-task>"
---

Before invoking the subagent, gather diagnostic context:

1. **Identify research context** from $ARGUMENTS — paper review, literature search, statistical analysis, or writing assistance.
2. **Check for existing research artifacts** by searching for .bib files, LaTeX sources (.tex), data files, or analysis notebooks (.ipynb).
3. **Detect statistical tools** from project config — R, scipy, statsmodels, pandas, or similar.
4. **Get scope overview** of the target area (if $ARGUMENTS specifies a paper or topic, scope to that).

Use the research-analyst subagent to help with: $ARGUMENTS
