---
name: j-paper-analysis
description: "Analyze an ML/AI research paper: structured summary, critical evaluation, and reviewer-style assessment using the three-pass method. Use when reviewing an academic paper, arxiv preprint, or conference submission. Do NOT use for quick paper summaries (ask directly instead)."
argument-hint: "<paper-url-or-path>"
model: opus
---

Analyze the ML/AI research paper at: $ARGUMENTS

If no arguments provided, ask for the paper URL, arxiv ID, or local file path.

## Setup (always do first)

Load skills via the Skill tool, in order:

1. `analysis-output-patterns` -- output structure rules
2. `output-completeness` -- the full analysis must be emitted without truncated sections

## Fetch and gather resources

Before analyzing, resolve and fetch the paper, then gather its resources in parallel:

1. **Resolve the paper**: If given a title or arxiv ID (not a URL), search the web for the canonical paper page (arxiv abs, conference proceedings, or publisher page).
2. **Fetch the paper content**: Get the full paper text for analysis.
3. **Gather resources in parallel**: The searches below are independent. **Delegate them to `research-analyst` subagents via the Task tool and run them concurrently in a single message.** Give each subagent a self-contained brief (paper title, authors, arxiv ID) and ask it to return the exact links it finds, or "None found".

   | Resource group | Agent | Returns |
   |---|---|---|
   | Official code repo, PapersWithCode, community implementations (note framework per repo) | `research-analyst` | Repo URLs, PapersWithCode URL, community impl URLs |
   | Videos, talks, blog posts, explainers | `research-analyst` | Conference/author talk URLs, blog/Distill/explainer URLs |
   | Supplementary materials, BibTeX citation | `research-analyst` | Project page, appendix, and dataset URLs; BibTeX from arxiv, Semantic Scholar, or publisher |

**Integrate**: collect the subagent results, dedupe, and populate the Resources & Links section. A subagent reporting "found it" is not proof -- resolve each link yourself and confirm it actually loads before writing it down as resolved. Use "Not available" for expected resources such as official code, and "None found" for optional ones such as blogs or videos.

For small or well-known papers, run these searches inline instead of delegating.

---

## Paper Analysis Methodology

Analyze ML/AI research papers using S. Keshav's three-pass reading method. Produces structured, self-contained summaries that capture all key information.

### Three-Pass Method

#### Pass 1: Bird's-Eye View

Quick scan to understand what the paper is about:

1. Read the title, abstract, and introduction carefully
2. Read all section and sub-section headings (ignore body text)
3. Glance at mathematical content to identify theoretical foundations
4. Read the conclusions
5. Scan the references, noting which ones you recognize

After this pass, answer the **Five Cs**:

| C | Question |
|---|----------|
| **Category** | What type of paper? (empirical study, new architecture, theoretical analysis, benchmark, survey, system description, method/technique) |
| **Context** | What prior work does it build on? What theoretical bases are used? |
| **Correctness** | Do the assumptions appear valid? |
| **Contributions** | What are the main contributions claimed? |
| **Clarity** | Is it well written? Clear structure? |

#### Pass 2: Content Grasp

Read with greater care. Ignore proofs and dense math derivations for now.

1. **Figures & diagrams**: Examine every figure, table, and diagram carefully
   - Are axes labeled? Error bars present? Results statistically significant?
   - What do the architecture diagrams reveal about the approach?
2. **Key claims**: Note every major claim and its supporting evidence
3. **Method details**: Understand the proposed method at a high level
   - What is the input/output?
   - What are the key components?
   - How does training work?
4. **Experimental setup**: Datasets, baselines, metrics, hardware
5. **Computational cost**: Note parameter counts, FLOPs, GPU hours, memory requirements
6. **Results**: Main results tables, ablation studies, comparisons
   - Are confidence intervals or error bars reported? How many runs/seeds?
7. **Terminology**: Note unfamiliar terms, acronyms, or concepts
8. **Unread references**: Mark important cited papers for follow-up

#### Pass 3: Deep Understanding

Virtually re-implement the paper mentally. Challenge everything.

1. **Assumptions**: Identify and challenge every assumption
   - Are they stated explicitly or implicit?
   - Are they reasonable for the problem domain?
2. **Methodology critique**: Could this be done differently or better?
   - What are the hidden failings?
   - What design choices are not well justified?
3. **Mathematical rigor**: Verify key equations and derivations
4. **Experimental validity**: Scrutinize the evaluation
   - Are baselines fair and up-to-date?
   - Is the evaluation protocol standard for the field?
   - Could the results be explained by confounding factors?
5. **Reproducibility**: Could you reimplement this?
   - Are hyperparameters fully specified?
   - Is the data pipeline described?
   - Is code available?
6. **Statistical rigor**: Multiple seeds/runs? Confidence intervals? Significance tests?
7. **Comparison fairness**: Do baselines get equal compute, tuning, and data access?
8. **Failure modes**: Where would this approach break? Edge cases, distribution shifts, adversarial inputs?
9. **Ethical considerations**: Bias, fairness, environmental cost, dual-use potential
10. **Future work**: Note ideas for extensions, improvements, or follow-up experiments
11. **Strong and weak points**: Identify what works well and what doesn't

### References

- `.claude/references/ai-ml/eval-and-benchmarking.md` -- ML evaluation methodology and benchmarks

---

### Output

Write the analysis to a file named `{paper-short-title}.md`. Produce these sections, in order; each row lists the fields the section must cover -- write the content in whatever prose or table form fits the paper, not a form to fill in.

| Section | Required content |
|---|---|
| Header | Full title, authors, venue and year, DOI if available, one-sentence TL;DR of the core contribution and result |
| Resources & Links | Paper page, PDF, official code ("Not available" if none), PapersWithCode, community implementations (with framework noted), video/talk, blog/explainer, supplementary materials -- each a URL, or "Not available"/"None found"; BibTeX citation |
| Five Cs (first-pass) | Category, Context, Correctness, Contributions, Clarity |
| Problem Statement | The problem this paper addresses and why it matters |
| Motivation & Gap | The gap in existing work this paper fills |
| Proposed Method | High-level overview; architecture/algorithm components; key equations with what each computes; training/optimization (objective function, optimizer, schedule, key hyperparameters); computational cost (parameters, FLOPs, training cost/hardware, inference time, memory, how cost scales) |
| Experimental Setup | Datasets (size, task, split); baselines; metrics; hardware and training time; whether baselines get equal compute/tuning/data |
| Key Results | Main results by method and metric; ablation studies and their effect; statistical rigor (runs/seeds, variance reporting, significance tests) |
| Critical Analysis | Novelty vs. closest prior work; strengths and weaknesses with reasoning; limitations (acknowledged by authors vs. unacknowledged); failure modes and edge cases; ethical considerations and broader impact (omit entirely if genuinely N/A); missing references; reproducibility assessment (code/data availability, hyperparameters specified, implementation complexity, overall reproducibility) |
| Connections & Context | Papers this work builds on with the relationship; potential impact on the field |
| Future Work & Open Questions | Extensions, improvements, unresolved questions |
| Reviewer Assessment | Overall score 1-10 with 2-3 sentence justification (see calibration table below); confidence 1-5; recommendation (Accept / Weak Accept / Borderline / Weak Reject / Reject); questions for authors |
| Key Takeaways | 3-5 bullets |
| Glossary | Term/definition table for unfamiliar terms and acronyms |

### Score Calibration

| Score | Meaning |
|-------|---------|
| 1-3 | Serious flaws, not suitable for publication |
| 4-5 | Below average; significant weaknesses outweigh contributions |
| 6 | Marginally above acceptance threshold |
| 7 | Good paper; solid contribution with minor issues |
| 8 | Strong paper; clear contribution, well-executed |
| 9-10 | Exceptional; significant advance for the field |

### Confidence Calibration

| Score | Meaning |
|-------|---------|
| 1 | Low -- outside area of expertise |
| 2 | Willing to defend but not certain |
| 3 | Fairly confident |
| 4 | Confident -- checked key details |
| 5 | Very confident -- deeply familiar with area |

### Process Guidelines

- Read the full paper across all three passes before writing the summary
- Be precise -- use exact numbers from the paper
- Distinguish between what the paper claims and what the evidence supports
- For critical analysis, be honest and constructive -- identify real issues, not nitpicks
- The summary should be self-contained: someone reading it should understand the paper without reading the original
- **Score calibration:** 6-7 = good paper with solid contribution; 8+ = genuinely strong/exceptional; don't grade-inflate
- **Omit N/A sections** rather than filling them with "Not applicable" placeholders
- **Novelty assessment:** compare against the closest specific prior work, not the field in general
- **TL;DR:** draft after Pass 1, refine after Pass 3
- **Resource links:** require genuine search effort -- use "Not available" for expected resources (official code) and "None found" for optional ones (blog posts, videos)
