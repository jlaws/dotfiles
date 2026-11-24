set -euo pipefail
cp -R "$1/email-digest/." .
rm inbox/2026-11-01-newsletter.txt
cat > inbox/2026-11-15-newsletter.txt <<'TXT'
From: The Weekly Model Digest <digest@example.org>
Date: Sat, 15 Nov 2026 08:00:00 +0000
Subject: Weekly Model Digest #214: sparse attention revisited, eval drift, and a hosted leaderboard

Hi all,

This week's picks:

1. Paper: "Sparse Routing Attention for Long-Context Retrieval"
   A routing layer that sends each query to a small subset of key blocks, cutting attention cost
   at 128k tokens. Read it: https://arxiv.example.org/abs/2610.01234

2. Paper: "Measuring Eval Drift Across Model Releases"
   Tracks how the same benchmark scores move between model versions when prompts are held fixed.
   Read it: https://arxiv.example.org/abs/2610.05678

3. Tool release: OpenGrader 0.4 adds pairwise LLM judging with order shuffling.
   https://github.example.com/opengrader/opengrader/releases/tag/v0.4.0

4. Blog: "What we learned running 10,000 agent evals" from a mid-size lab.
   https://blog.example.net/posts/10k-agent-evals

5. Dataset: CodeFix-2k, a set of 2,000 real bug-fix commits with tests, released under CC-BY.
   https://data.example.com/codefix-2k

6. Product: EvalBoard opens a hosted leaderboard for private benchmark suites.
   https://click.example.net/ss/c/Q2xpY2s/evalboard

That's it for this week. Reply to tell us what to cover next.
TXT
cat > .seen-links <<'TXT'
https://arxiv.example.org/abs/2610.05678
https://blog.example.net/posts/10k-agent-evals
https://arxiv.example.org/abs/2609.11111
TXT
