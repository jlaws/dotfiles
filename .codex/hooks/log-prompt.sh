#!/bin/zsh
# Read JSON from stdin, extract prompt, append with timestamp
INPUT=$(cat)
PROMPT=$(echo "$INPUT" | jq -r '.prompt // empty')
[ -z "$PROMPT" ] && exit 0
LOGFILE="$HOME/.codex/PROMPT_LOG.md"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
# Use lockfile (mkdir is atomic) for concurrent safety
LOCKDIR="$LOGFILE.lock"
while ! mkdir "$LOCKDIR" 2>/dev/null; do sleep 0.1; done
trap 'rmdir "$LOCKDIR" 2>/dev/null' EXIT
echo -e "\n## $TIMESTAMP\n\n$PROMPT" >> "$LOGFILE"
