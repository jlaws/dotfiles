# Paper Analysis Methodology

Analyze ML/AI research papers using S. Keshav's three-pass reading method. Produces structured, self-contained summaries that capture all key information.

## Overview

Reference for a systematic framework for reading and analyzing research papers. Use when you need to:

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

See references/output-template.md for the complete paper analysis markdown template and process guidelines.

## References

- .cursor/references/ai-ml/eval-and-benchmarking.md — ML evaluation methodology and benchmarks
