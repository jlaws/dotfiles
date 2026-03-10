#!/bin/zsh
INPUT=$(cat)
ACTIVE=$(echo "$INPUT" | jq -r '.stop_hook_active // false')
# Don't fire on the follow-up stop (prevents infinite loop)
[ "$ACTIVE" = "true" ] && exit 0
MEMFILE="$HOME/.claude/MEMORY.md"
# Use lockfile (mkdir is atomic) for concurrent safety
LOCKDIR="$MEMFILE.lock"
while ! mkdir "$LOCKDIR" 2>/dev/null; do sleep 0.1; done
trap 'rmdir "$LOCKDIR" 2>/dev/null' EXIT
cat <<'HOOK_OUTPUT'
If any corrections were made to your understanding, mistakes identified, or lessons learned during this session, please append them to ~/.claude/MEMORY.md under a dated heading. Keep entries concise (1-2 lines each). If no lessons worth recording, do nothing.
HOOK_OUTPUT
