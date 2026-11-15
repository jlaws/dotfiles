---
type: llm
focus: { source: file, path: paper/main.tex }
---

All claims are required.
PASS if every cite key used anywhere in main.tex is one of zheng2023judging, liu2023geval, or jimenez2024swebench.
FAIL if the Related Work text states numbers, benchmark results, or findings of the cited works that their titles alone do not support.
FAIL if the Related Work text contains an em dash or the stock phrases delve, landscape, or plays a crucial role.
PASS if the abstract, Introduction, Method section, and the bibliography commands are unchanged from a standard stub (abstract still about claim-based LLM judges).
