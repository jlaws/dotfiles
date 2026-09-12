#!/usr/bin/env bash
# UserPromptSubmit hook -- append the submitted prompt to a local log, secrets redacted.
#
# Shebang is bash because both harnesses invoke this as `bash <path>`; the previous `#!/bin/zsh`
# described an interpreter that never ran.
#
# Prompts routinely carry credentials the user pasted without thinking, so redaction runs before
# anything touches disk and the log is created 0600. The rules below are deliberately broad and
# case-insensitive: an earlier version anchored on uppercase SECRET/TOKEN/KEY with `=`, which
# missed every lowercase `api_key=`, every `password=`, every colon form, and every provider
# prefix. Redaction is best-effort -- treat this file as sensitive regardless.

set -uo pipefail

command -v jq >/dev/null 2>&1 || exit 0

INPUT=$(cat)
PROMPT=$(printf '%s' "$INPUT" | jq -r '.prompt // empty' 2>/dev/null)
[ -z "$PROMPT" ] && exit 0

PROMPT=$(printf '%s' "$PROMPT" | sed -E \
  -e 's/(sk-[A-Za-z0-9_-]{6})[A-Za-z0-9_-]*/\1**REDACTED**/g' \
  -e 's/(sk_(live|test)_[A-Za-z0-9]{4})[A-Za-z0-9]*/\1**REDACTED**/g' \
  -e 's/(gh[pousr]_[A-Za-z0-9]{4})[A-Za-z0-9]*/\1**REDACTED**/g' \
  -e 's/(github_pat_[A-Za-z0-9]{4})[A-Za-z0-9_]*/\1**REDACTED**/g' \
  -e 's/(xox[baprs]-)[A-Za-z0-9-]{6,}/\1**REDACTED**/g' \
  -e 's/(AKIA)[0-9A-Z]{16}/\1**REDACTED**/g' \
  -e 's/(eyJ[A-Za-z0-9_-]{4})[A-Za-z0-9._-]*/\1**REDACTED**/g' \
  -e 's/(-----BEGIN [A-Z ]*PRIVATE KEY-----).*/\1**REDACTED**/g' \
  -e 's/([Bb]earer |[Bb]asic )[A-Za-z0-9._=+\/-]+/\1**REDACTED**/g' \
  -e 's/([A-Za-z_-]*([Ss][Ee][Cc][Rr][Ee][Tt]|[Tt][Oo][Kk][Ee][Nn]|[Kk][Ee][Yy]|[Pp][Aa][Ss][Ss][Ww][Oo][Rr][Dd]|[Pp][Aa][Ss][Ss][Ww][Dd])[A-Za-z_-]*)[[:space:]]*[=:][[:space:]]*"?[^"[:space:]]+"?/\1=**REDACTED**/g' \
  -e 's|([a-zA-Z][a-zA-Z0-9+.-]*://[^:/[:space:]]+):[^@[:space:]]+@|\1:**REDACTED**@|g')

LOGFILE="${PROMPT_LOG_FILE:-$HOME/.claude/PROMPT_LOG.md}"
# The lock loop cannot succeed when the parent is missing, so it would burn its full retry budget
# -- 5.0s, exactly the hook timeout -- on a path error.
mkdir -p "$(dirname "$LOGFILE")" 2>/dev/null || exit 0

# Append-only with no rotation reached 110 MB on this machine in ~6 months. Roll at 50 MB and keep
# one generation; the log is a convenience, not an archive.
MAX_BYTES=${PROMPT_LOG_MAX_BYTES:-52428800}
if [ -f "$LOGFILE" ]; then
  SIZE=$(wc -c <"$LOGFILE" 2>/dev/null | tr -d ' ')
  if [ -n "$SIZE" ] && [ "$SIZE" -gt "$MAX_BYTES" ]; then
    mv -f "$LOGFILE" "$LOGFILE.1" 2>/dev/null || true
  fi
fi

TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# mkdir is atomic, so it serialises concurrent sessions.
LOCKDIR="$LOGFILE.lock"
trap 'rmdir "$LOCKDIR" 2>/dev/null' EXIT
LOCK_RETRIES=0
while ! mkdir "$LOCKDIR" 2>/dev/null; do
  LOCK_RETRIES=$((LOCK_RETRIES + 1))
  if [ "$LOCK_RETRIES" -ge 20 ]; then
    rmdir "$LOCKDIR" 2>/dev/null # stale lock -- force remove
    mkdir "$LOCKDIR" 2>/dev/null || true
    break
  fi
  sleep 0.1
done

# Create with restrictive permissions before the first write. Codex keeps auth.json and
# history.jsonl at 0600 in the same directory; a world-readable prompt log beside them is a
# mismatch, not a considered choice.
if [ ! -f "$LOGFILE" ]; then
  (umask 077 && : >"$LOGFILE") 2>/dev/null || true
fi
chmod 600 "$LOGFILE" 2>/dev/null || true

printf '\n## %s\n\n%s\n' "$TIMESTAMP" "$PROMPT" >>"$LOGFILE"
