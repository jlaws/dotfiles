---
name: paper-analysis-methodology
description: Systematic methodology for analyzing ML/AI research papers using the three-pass reading method. Use when reviewing papers, extracting key insights, or producing structured summaries.
---

# Paper Analysis Methodology

Analyze ML/AI research papers using S. Keshav's three-pass reading method. Produces structured, self-contained summaries that capture all key information.

## Overview

This skill provides a systematic framework for reading and analyzing research papers. Use this skill when you need to:

- Analyze a research paper thoroughly
- Extract implementation-relevant details
- Produce a structured summary for future reference
- Evaluate paper quality and contributions
- Identify follow-up work and open questions

## Three-Pass Method

### Pass 1: Bird's-Eye View

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

### Resource Discovery

After Pass 1, search for supporting resources:

1. **Code**: Check paper footer/abstract for repo links, search GitHub by title, check PapersWithCode
2. **Community implementations**: Search GitHub for reimplementations; note framework (PyTorch, JAX, TF, etc.)
3. **Presentations**: Look for conference talks, author walkthroughs, or video explainers
4. **Blog posts**: Check author blogs, Distill, and popular ML blogs for write-ups
5. **Supplementary materials**: Project pages, appendices, datasets, interactive demos
6. **Citation**: Retrieve BibTeX from arxiv, Semantic Scholar, or the publisher

### Pass 2: Content Grasp

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

### Pass 3: Deep Understanding

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

## Output Template

Create a file named `{paper-short-title}.md`. Use this template:

`````markdown
# {Full Paper Title}

**Authors:** {Author list}
**Published:** {Venue, year}
**DOI:** {DOI if available}

> **TL;DR:** {One-sentence summary of the paper's core contribution and result.}

---

## Resources & Links

| Resource | Link |
|----------|------|
| Paper page | {arxiv abs, conference page, or publisher URL} |
| PDF | {direct PDF link} |
| Official code | {repository URL — or "Not available"} |
| PapersWithCode | {PapersWithCode URL — or "None found"} |
| Community implementations | {URLs with framework noted — or "None found"} |
| Video / Talk | {URL — or "None found"} |
| Blog post / Explainer | {URL — or "None found"} |
| Supplementary materials | {URL — or "None found"} |

### Citation

```bibtex
{BibTeX entry}
```

---

## Five Cs (First-Pass Assessment)

| Dimension | Assessment |
|-----------|------------|
| **Category** | {Paper type} |
| **Context** | {Key prior work and foundations} |
| **Correctness** | {Validity of assumptions} |
| **Contributions** | {Numbered list of contributions} |
| **Clarity** | {Writing quality assessment} |

---

## Problem Statement

{What problem does this paper address? Why does it matter?}

## Motivation & Gap

{What gap in existing work does this paper fill?}

---

## Proposed Method

### Overview
{High-level description in 3-5 sentences.}

### Architecture / Algorithm
- {Component 1}: {description}
- {Component N}: {description}

### Key Equations
1. {Equation name}: `{equation}` — {what it computes}

### Training / Optimization
- **Objective function:** {loss}
- **Optimizer:** {optimizer and hyperparameters}
- **Schedule:** {learning rate schedule}
- **Key hyperparameters:** {list with values}

### Computational Cost
- **Parameters:** {total parameter count}
- **FLOPs:** {training/inference FLOPs if reported}
- **Training cost:** {GPU hours, hardware, estimated cost}
- **Inference time:** {latency per sample/batch}
- **Memory:** {peak GPU memory}
- **Scalability notes:** {how cost scales with data/model size}

---

## Experimental Setup

### Datasets
| Dataset | Size | Task | Split |
|---------|------|------|-------|

### Baselines
{Comparison methods}

### Metrics
{Evaluation metrics}

### Hardware & Budget
- **Hardware:** {GPUs/TPUs, count, type}
- **Training time:** {wall-clock time}
- **Comparison fairness:** {Do baselines get equal compute/tuning/data?}

---

## Key Results

### Main Findings
| Method | {Metric 1} | {Metric 2} |
|--------|-----------|-----------|

### Ablation Studies
- {Component}: {effect on performance}

### Statistical Rigor
- **Runs/seeds:** {number of independent runs}
- **Variance reporting:** {std dev, CI, IQR — what's reported?}
- **Significance tests:** {statistical tests used, if any}

---

## Critical Analysis

### Novelty Assessment
- **What is genuinely new:** {novel contributions vs. incremental improvements}
- **Closest prior work:** {most similar existing method and key differences}

### Strengths
1. {Strength with reasoning}

### Weaknesses
1. {Weakness with reasoning}

### Limitations
- **Acknowledged by authors:** {limitations the authors explicitly discuss}
- **Unacknowledged:** {limitations not discussed but apparent from analysis}

### Failure Modes & Edge Cases
{Where would this approach break? Distribution shifts, adversarial inputs, scaling limits, etc.}

### Ethical Considerations & Broader Impact
{Bias, fairness, environmental cost, dual-use potential, societal implications. Omit this section entirely if genuinely N/A.}

### Missing References
{Important related work not cited by the paper.}

### Reproducibility Assessment
- **Code available:** {Yes/No — see Resources & Links}
- **Data available:** {Yes/No — public datasets vs. proprietary}
- **Hyperparameters specified:** {Yes/Partially/No}
- **Implementation complexity:** {Low/Medium/High — effort to reimplement}
- **Overall reproducibility:** {High/Medium/Low}

---

## Connections & Context

### Builds On
- [{Paper}]({url}): {relationship}

### Potential Impact
{How might this work influence the field?}

---

## Future Work & Open Questions
{Extensions, improvements, unresolved questions}

---

## Reviewer Assessment

### Overall Score

| Score | Meaning |
|-------|---------|
| 1-3 | Serious flaws, not suitable for publication |
| 4-5 | Below average; significant weaknesses outweigh contributions |
| 6 | Marginally above acceptance threshold |
| 7 | Good paper; solid contribution with minor issues |
| 8 | Strong paper; clear contribution, well-executed |
| 9-10 | Exceptional; significant advance for the field |

**Score: {X}/10**
**Justification:** {2-3 sentences explaining the score}

### Confidence

| Score | Meaning |
|-------|---------|
| 1 | Low — outside area of expertise |
| 2 | Willing to defend but not certain |
| 3 | Fairly confident |
| 4 | Confident — checked key details |
| 5 | Very confident — deeply familiar with area |

**Confidence: {X}/5**

### Recommendation
**{Accept / Weak Accept / Borderline / Weak Reject / Reject}**

### Questions for Authors
1. {Key question that would affect the assessment}

---

## Key Takeaways
- {3-5 bullet points}

---

## Glossary
| Term | Definition |
|------|------------|

*Analysis generated using the three-pass method (Keshav, 2016).*
`````

## Process Guidelines

- Read the full paper across all three passes before writing the summary
- Be precise — use exact numbers from the paper
- Distinguish between what the paper claims and what the evidence supports
- For critical analysis, be honest and constructive — identify real issues, not nitpicks
- The summary should be self-contained: someone reading it should understand the paper without reading the original
- **Score calibration:** 6-7 = good paper with solid contribution; 8+ = genuinely strong/exceptional; don't grade-inflate
- **Omit N/A sections** rather than filling them with "Not applicable" placeholders
- **Novelty assessment:** compare against the closest specific prior work, not the field in general
- **TL;DR:** draft after Pass 1, refine after Pass 3
- **Resource links:** require genuine search effort — use "Not available" for expected resources (official code) and "None found" for optional ones (blog posts, videos)
