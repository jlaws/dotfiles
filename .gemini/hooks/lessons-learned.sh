#!/bin/zsh
# Gemini SessionEnd hook -- prompt agent to record lessons learned
# stdout MUST be JSON only; diagnostics go to stderr.
INPUT=$(cat)
MEMFILE="$HOME/.gemini/MEMORY.md"
# Use lockfile (mkdir is atomic) for concurrent safety
LOCKDIR="$MEMFILE.lock"
LOCK_RETRIES=0
while ! mkdir "$LOCKDIR" 2>/dev/null; do
  LOCK_RETRIES=$((LOCK_RETRIES + 1))
  if [ "$LOCK_RETRIES" -ge 50 ]; then
    rmdir "$LOCKDIR" 2>/dev/null
    mkdir "$LOCKDIR" 2>/dev/null || true
    break
  fi
  sleep 0.1
done
trap 'rmdir "$LOCKDIR" 2>/dev/null' EXIT
jq -n --arg msg "If any corrections were made to your understanding, mistakes identified, or lessons learned during this session, please append them to ~/.gemini/MEMORY.md under a dated heading. Keep entries concise (1-2 lines each). If no lessons worth recording, do nothing." \
  '{systemMessage: $msg}'
