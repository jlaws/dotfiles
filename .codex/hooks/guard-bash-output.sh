#!/usr/bin/env bash
# PreToolUse:Bash guard -- ADVISORY ONLY.
#
# Suggests a bounded form of a command that would otherwise dump unbounded output into the agent's
# context. It NEVER blocks and NEVER rewrites: the only thing it emits is an advisory message, and
# it always exits 0. Rewriting the command an agent asked for would put a lossy layer between that
# agent and its evidence, which contradicts the byte-for-byte rule in CLAUDE.md. This is the
# Fail-Open Principle from references/workflow/hook-patterns.md, applied.
#
# Usage: guard-bash-output.sh [--format claude|codex]
#   claude (default) -> {"systemMessage": "..."}
#   codex            -> {"hookSpecificOutput": {"hookEventName": "PreToolUse",
#                                               "additionalContext": "..."}}
#
# `-e` is deliberately absent: a pattern that does not match must never fail the hook.

set -uo pipefail

FORMAT="claude"
while [ $# -gt 0 ]; do
  case "$1" in
    --format) FORMAT="${2:-claude}"; shift 2 || shift ;;
    --format=*) FORMAT="${1#*=}"; shift ;;
    *) shift ;;
  esac
done

command -v jq >/dev/null 2>&1 || exit 0

INPUT="$(cat)"
CMD="$(printf '%s' "$INPUT" | jq -r '.tool_input.command // empty' 2>/dev/null)"
[ -z "$CMD" ] && exit 0

# A heredoc carries its own body. Matching rules against that body is noise, and file writes and
# report bodies legitimately use one.
case "$CMD" in *'<<'*) exit 0 ;; esac

# Leading whitespace off, so every rule can anchor at the start of the first command.
TRIM="${CMD#"${CMD%%[![:space:]]*}"}"
read -ra PARTS <<<"$TRIM"

# has_flag <regex> -- does the command carry a flag matching this?
has_flag() { [[ "$TRIM" =~ $1 ]]; }

# extra_args <n> -- count tokens beyond the first n (the runner and its subcommand).
extra_args() { echo $(( ${#PARTS[@]} > $1 ? ${#PARTS[@]} - $1 : 0 )); }

# piped_into_bounding_tool -- is output already narrowed by a downstream filter?
piped_into_bounding_tool() {
  [[ "$TRIM" =~ \|[[:space:]]*(grep|rg|head|tail|jq|wc|sed|awk|sort|uniq|cut) ]]
}

SUGGESTION=""

if [[ "$TRIM" =~ \&\& ]]; then
  SUGGESTION="Chained with '&&' -- run these as separate calls so a failure points at one thing (CLAUDE.md, Bash)."

elif [[ "$TRIM" =~ ^git[[:space:]]+log([[:space:]]|$) ]]; then
  has_flag '(--oneline|--stat|--shortstat|--numstat|-n[[:space:]]+[0-9]+|(^|[[:space:]])-[0-9]+([[:space:]]|$))' \
    || SUGGESTION="Unbounded 'git log' -- prefer 'git log --oneline -20', or add --stat."

elif [[ "$TRIM" =~ ^git[[:space:]]+diff([[:space:]]|$) ]]; then
  if ! has_flag '(--stat|--name-only|--numstat|--shortstat|--name-status)'; then
    scoped=0
    for tok in "${PARTS[@]:2}"; do
      [[ "$tok" == -* ]] || { scoped=1; break; }
    done
    [ "$scoped" -eq 1 ] || SUGGESTION="Unbounded 'git diff' -- run 'git diff --stat' first, then diff only the paths that matter."
  fi

elif [[ "$TRIM" =~ ^cat([[:space:]]|$) ]]; then
  piped_into_bounding_tool \
    || SUGGESTION="'cat' of a whole file -- prefer the Read tool, or 'sed -n \"START,ENDp\"' for one range."

elif [[ "$TRIM" =~ ^find([[:space:]]|$) ]]; then
  has_flag '(-maxdepth|-name|-iname|-path)' \
    || SUGGESTION="Unbounded 'find' -- prefer the Glob tool, or add -maxdepth and -name."

elif [[ "$TRIM" =~ ^(grep|rg)([[:space:]]|$) ]]; then
  if ! has_flag '((^|[[:space:]])-[a-zA-Z]*[lcm]([[:space:]]|$)|--files-with-matches|--count|--max-count)'; then
    piped_into_bounding_tool \
      || SUGGESTION="Unbounded '${PARTS[0]}' -- prefer the Grep tool, or add -l to list files only."
  fi

elif [[ "$TRIM" =~ ^ls([[:space:]]|$) ]] && has_flag '(^|[[:space:]])-[a-zA-Z]*R([[:space:]]|$)'; then
  SUGGESTION="Recursive 'ls -R' -- prefer 'tree -L 2', or the Glob tool for a targeted pattern."

elif [[ "$TRIM" =~ ^tree([[:space:]]|$) ]]; then
  has_flag '(^|[[:space:]])-L([[:space:]]|$)' \
    || SUGGESTION="Unbounded 'tree' -- add '-L 2' to cap the depth."

elif [[ "$TRIM" =~ ^pytest([[:space:]]|$) ]]; then
  [ "$(extra_args 1)" -gt 0 ] \
    || SUGGESTION="Whole-suite 'pytest' -- scope it to a file or '-k <name>' while iterating."

elif [[ "$TRIM" =~ ^cargo[[:space:]]+test([[:space:]]|$) ]]; then
  [ "$(extra_args 2)" -gt 0 ] \
    || SUGGESTION="Whole-suite 'cargo test' -- scope it to a module path while iterating."

elif [[ "$TRIM" =~ ^npm[[:space:]]+test([[:space:]]|$) ]]; then
  [ "$(extra_args 2)" -gt 0 ] \
    || SUGGESTION="Whole-suite 'npm test' -- scope it with '-- --grep <name>' while iterating."

elif [[ "$TRIM" =~ ^go[[:space:]]+test([[:space:]]|$) ]]; then
  if [ "$(extra_args 2)" -eq 0 ] || [[ "${PARTS[2]:-}" == "./..." ]]; then
    SUGGESTION="Whole-module 'go test' -- scope it to a package or '-run <name>' while iterating."
  fi
fi

[ -z "$SUGGESTION" ] && exit 0

if [ "$FORMAT" = "codex" ]; then
  jq -n --arg msg "$SUGGESTION" \
    '{hookSpecificOutput: {hookEventName: "PreToolUse", additionalContext: $msg}}'
else
  jq -n --arg msg "$SUGGESTION" '{systemMessage: $msg}'
fi
exit 0
