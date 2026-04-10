#!/bin/zsh
# Read JSON from stdin, extract prompt, append with timestamp
INPUT=$(cat)
PROMPT=$(echo "$INPUT" | jq -r '.prompt // empty')
[ -z "$PROMPT" ] && exit 0
# WARNING: Prompts may contain sensitive data (API keys, tokens, credentials).
# Redact common secret patterns before logging.
PROMPT=$(echo "$PROMPT" | sed -E \
  -e 's/(sk-[A-Za-z0-9_-]{10})[A-Za-z0-9_-]*/\1**REDACTED**/g' \
  -e 's/(Bearer )[A-Za-z0-9._-]+/\1**REDACTED**/g' \
  -e 's/([A-Za-z_]*SECRET[A-Za-z_]*=)[^ ]*/\1**REDACTED**/g' \
  -e 's/([A-Za-z_]*TOKEN[A-Za-z_]*=)[^ ]*/\1**REDACTED**/g' \
  -e 's/([A-Za-z_]*KEY[A-Za-z_]*=)[^ ]*/\1**REDACTED**/g')
LOGFILE="$HOME/.codex/PROMPT_LOG.md"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
# Use lockfile (mkdir is atomic) for concurrent safety
LOCKDIR="$LOGFILE.lock"
LOCK_RETRIES=0
while ! mkdir "$LOCKDIR" 2>/dev/null; do
  LOCK_RETRIES=$((LOCK_RETRIES + 1))
  if [ "$LOCK_RETRIES" -ge 50 ]; then
    rmdir "$LOCKDIR" 2>/dev/null  # stale lock -- force remove
    mkdir "$LOCKDIR" 2>/dev/null || true
    break
  fi
  sleep 0.1
done
trap 'rmdir "$LOCKDIR" 2>/dev/null' EXIT
echo -e "\n## $TIMESTAMP\n\n$PROMPT" >> "$LOGFILE"
