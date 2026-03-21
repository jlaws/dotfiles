#!/bin/bash
INPUT=$(cat)
STATUS=$(echo "$INPUT" | jq -r '.status // "completed"')
MEMFILE="$HOME/.cursor/MEMORY.md"
LOCKDIR="$MEMFILE.lock"
while ! mkdir "$LOCKDIR" 2>/dev/null; do sleep 0.1; done
trap 'rmdir "$LOCKDIR" 2>/dev/null' EXIT
cat <<'EOF'
{"followup_message":"If any corrections were made to your understanding, mistakes identified, or lessons learned during this session, please append them to ~/.cursor/MEMORY.md under a dated heading. Keep entries concise (1-2 lines each). If no lessons worth recording, do nothing."}
EOF
