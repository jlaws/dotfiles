---
name: j-paper
description: "LaTeX research paper consultation -- templates, section editing, compile + log analysis, source research, citation integration, and submission readiness. Use when scaffolding a new paper, updating sections, debugging build errors, integrating cited sources, or preparing a venue submission. Do NOT use for general LaTeX syntax questions (read references/research/latex-paper-writing.md directly) or non-academic documents (use /j-docs)."
argument-hint: "<task: e.g., 'init neurips template', 'compile and fix errors', 'add citation for <url>', 'check page limit'>"
---

Task: $ARGUMENTS

If no arguments provided, ask what the user wants to do (init a new paper, edit a section, compile + diagnose, add a citation from a source, check submission readiness, or clean the bibliography).

## Setup (always do first)

Load skills, in order:

1. `verification-before-completion` -- required before claiming "compiles cleanly" or "no errors"
2. `output-completeness` -- for any generated section, template, or table (no truncation)

Read the canonical reference before responding: `.agents/references/research/latex-paper-writing.md` (venue specs, preamble template, gotchas).

## Diagnostic scans (run only those relevant to the task)

1. **Paper root**: `Glob` for `*.tex`. Identify the file containing `\documentclass`.
2. **Venue**: grep the preamble for `neurips_*`, `icml*`, `iclr*`, `aaai*`, `acl*`, `cvpr*`. Note the `[preprint]` vs `[final]` style flag.
3. **Bibliography**: `Glob` for `*.bib`. Identify `\bibliography{...}` or `\addbibresource{...}` directive in the root file.
4. **Project layout**: `Glob` for `figures/`, `sections/`, `tables/`, `appendix/`.
5. **Build tool**: check for `latexmkrc`, `.latexmkrc`, or a `Makefile` target.

## Route on intent

### a. Init / template scaffold

- Determine target venue from arguments. If ambiguous, ask.
- Pull the preamble + macros from `.agents/references/research/latex-paper-writing.md`.
- If the venue style file (e.g. `neurips_2024.sty`) is not present locally, tell the user the canonical filename to download from the venue site -- do NOT fabricate a URL.
- Scaffold: `main.tex`, `references.bib`, `sections/`, `figures/`, `latexmkrc`. Use `Write` only for new files; `Edit` for any pre-existing file.

### b. Section editing

- Read the target file before editing.
- Prefer `Edit` over rewriting the whole file.
- Preserve existing `\label{...}`, `\ref{...}`, `\cite{...}`, and custom macros.

### c. Compile + log analysis

- Run the build in one Bash call: `latexmk -pdf -interaction=nonstopmode <root>.tex`.
- On failure, parse the `.log` for: lines starting with `! `, `LaTeX Warning: Reference`, `LaTeX Warning: Citation`, `Overfull \hbox`, `Underfull \vbox`, `! Package` errors, and font-substitution warnings.
- Categorize each issue: missing package, undefined ref, missing citation, overfull box, font issue, math-mode error.
- Iteration cap: max 2 fix attempts on the same error class. If still failing, stop and report -- do not loop.
- Verify before claiming success: re-run `latexmk` and paste the tail of the output showing 0 errors and 0 undefined refs/citations.

### d. Source research + citation integration

- Accept inputs: URL, DOI, arXiv ID, plain title.
- Use `WebFetch` to retrieve the source page (arxiv.org/abs/..., DOI resolver, venue page). Extract title, authors, year, venue, abstract.
- Generate a deterministic BibTeX key: `firstauthorlast_year_keyword` (e.g. `vaswani_2017_attention`).
- Append the entry to the detected `.bib` file. Dedupe by DOI first, then normalized title.
- If `$ARGUMENTS` indicates where to cite, insert `\cite{key}` at that location (Edit). Otherwise report the key for the user to place.
- Never fabricate citation fields. If authors / abstract / venue cannot be extracted, say so explicitly and leave the field blank rather than guess.

### e. Submission readiness checks

Look up the target venue's requirements in `.agents/references/research/latex-paper-writing.md`, then verify:

| Check | How |
|-------|-----|
| Page count | Compile, then `pdfinfo <pdf>` (one Bash call) -- compare to venue limit |
| Anonymization | grep `\author{`, acknowledgments block, and self-citations like "in our prior work" / "we previously showed" |
| Style flag | grep for `[preprint]` vs `[final]` -- must match submission phase |
| Font embedding | `pdffonts <pdf>` (one Bash call) -- every font must show "yes" under "emb" |
| Abstract length | Word-count the abstract block against the venue limit (NeurIPS: 250) |
| Supplementary structure | Confirm appendix is in the right form for the venue (separate PDF vs appended) |

Output a checklist: PASS / FAIL / NEEDS-REVIEW per item.

### f. Bibliography hygiene

- Parse `.bib` entries. Use `bibtool` if installed; otherwise grep-based scan.
- Detect duplicates by DOI first, then by normalized title (lowercase, strip punctuation).
- Flag entries with missing DOI, year, or venue.
- Propose merges -- never silently delete entries.

## Communication contract

- Lead with the answer, fix, or generated content.
- For multi-step tasks, paste verification output (compile log tail, page count, font table) BEFORE declaring done.
- When a fact is unknown (style file URL, exact filename version for the current year, DOI), say so explicitly. Do not guess.
- When generating LaTeX, never truncate -- every section, every table cell, every preamble line must be complete.

---

### Cross-References

- **.agents/references/research/latex-paper-writing.md** -- venue specs, preamble template, gotchas
- **.agents/references/research/literature-review.md** -- when scope expands to surveying related work
- **.agents/references/research/paper-analysis-methodology.md** -- when reading cited papers in depth
- **agent:research-analyst** -- for literature surveys or deep reading of cited papers, delegate (it loads `.agents/references/research/`); verify its output
- **skill:analysis-output-patterns** -- output structure rules
- **skill:verification-before-completion** -- required before claiming compile-clean / submission-ready
- **skill:output-completeness** -- required when generating templates, sections, or tables
