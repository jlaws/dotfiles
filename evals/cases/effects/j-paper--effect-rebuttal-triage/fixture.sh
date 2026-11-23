set -euo pipefail
mkdir -p paper
cat > paper/main.tex <<'EOF'
\documentclass{article}
\title{Checkable-Claim Judges for Agent Evaluation}
\author{Anonymous}
\begin{document}
\maketitle

\section{Method}
Each task carries two to five claims, and the judge returns PASS or FAIL per claim. A response
passes when every claim passes. We compare the judge against two expert graders on 300 tasks.

\section{Results}
\begin{table}[h]
\centering
\begin{tabular}{lcc}
Judge & Agreement with experts & Cohen's $\kappa$ \\
\hline
Single quality score & 0.71 & 0.42 \\
Checkable claims (ours) & 0.88 & 0.74 \\
\end{tabular}
\caption{Agreement with expert graders on 300 tasks.}
\end{table}

\end{document}
EOF
cat > reviews.md <<'EOF'
# Reviews

## Reviewer 1

- R1.1: The results table has no lexical baseline. A BM25 retrieval baseline would show whether
  the judge adds anything over simple overlap with a reference answer.
- R1.2: The paper never says how many claims each task carries, so I cannot judge annotation cost.
- R1.3: "Checkable claims" is used before it is defined; define it in the introduction.

## Reviewer 2

- R2.1: Please compare against a BM25 baseline. Without it the gain over the single-score judge is
  hard to interpret.
- R2.2: Cohen's kappa cannot be computed for binary PASS/FAIL labels, so the kappa column is
  meaningless and should be removed.
EOF
