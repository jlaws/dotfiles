set -euo pipefail
cp -R "$1/email-digest/." .
{
  cat inbox/2026-11-01-newsletter.txt
  printf '%s\n' '' '6. Product: EvalBoard opens a hosted leaderboard for private benchmark suites.'
  printf '%s\n' '   https://click.example.net/ss/c/Q2xpY2s/evalboard'
} > inbox/2026-11-15-newsletter.txt
cat > .seen-links <<'TXT'
https://arxiv.example.org/abs/2610.05678
https://blog.example.net/posts/10k-agent-evals
https://arxiv.example.org/abs/2609.11111
TXT
