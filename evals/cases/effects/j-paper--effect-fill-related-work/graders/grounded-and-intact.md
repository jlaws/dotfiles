---
type: llm
focus: { source: file, path: paper/main.tex }
---

All claims are required.
PASS if every cite key used anywhere in main.tex is one of zheng2023judging, liu2023geval, or jimenez2024swebench.
FAIL if the Related Work text states numbers, benchmark results, or findings of the cited works that their titles alone do not support.
FAIL if the Related Work text contains an em dash (the Unicode em dash character or the LaTeX --- ligature) or the stock phrases delve, landscape, or plays a crucial role.
PASS if these strings remain in main.tex unchanged (line breaks aside): the abstract sentence "We study LLM judges that grade each response against a list of task-specific claims instead of a single quality score, and measure their agreement with expert graders.", the Introduction sentence "Automated evaluation of coding agents increasingly relies on LLM judges.", the Method sentence "Each task carries two to five claims; the judge returns PASS or FAIL per claim.", and the lines \bibliographystyle{plainnat} and \bibliography{refs}.
