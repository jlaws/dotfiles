set -euo pipefail
cp -R "$1/email-digest/." .
{
  printf '%s\n' '<!-- SEEN LINKS (already processed, skip these):'
  printf '%s\n' 'https://arxiv.example.org/abs/2610.01234'
  printf '%s\n' 'https://github.example.com/opengrader/opengrader/releases/tag/v0.4.0'
  printf '%s\n' '-->'
  cat inbox/2026-11-01-newsletter.txt
  printf '%s\n' '' '6. Product: Rubric Studio launches a hosted grader UI with shared rubrics.'
  printf '%s\n' '   https://click.example.net/ss/c/Zk9rU2xN/rubric-studio'
  printf '%s\n' '' 'View in browser: https://digest.example.org/view/212'
  printf '%s\n' 'Unsubscribe: https://digest.example.org/unsubscribe?u=77'
} > inbox/2026-11-08-newsletter.txt
