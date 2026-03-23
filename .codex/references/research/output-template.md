# Output Template & Process Guidelines

Extracted from the paper-analysis-methodology skill for progressive disclosure.

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
