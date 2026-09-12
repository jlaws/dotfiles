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
# Precision is the whole game. A guard that nags about a bounded command trains the reader to
# ignore it, so a false positive costs more than a miss. Two consequences run through the code:
#
#   1. The command is split into statements and pipeline stages with an awk tokenizer that tracks
#      quote state, then rules match against ARGV TOKENS rather than the raw string. Matching the
#      flat string let a search pattern silence a rule -- `grep -rn "needle -l here" .` read as
#      carrying -l -- and let a quoted `|` read as a pipeline.
#   2. Rules run against every statement, not just the first. An earlier cut instead flagged `&&`
#      itself; that fired on nearly every command and, being first in the chain, shadowed all seven
#      output rules. Deleting it left `cd . &&` as a universal opt-out. Splitting fixes both.
#
# `-e` is deliberately absent: a pattern that does not match must never fail the hook.

set -uo pipefail

FORMAT="claude"
while [ $# -gt 0 ]; do
  case "$1" in
    --format) FORMAT="${2:-}"; shift 2 || shift ;;
    --format=*) FORMAT="${1#*=}"; shift ;;
    *) shift ;;
  esac
done
# An unknown format would hand the harness a field it does not read. Say nothing instead.
case "$FORMAT" in claude | codex) ;; *) exit 0 ;; esac

if ! command -v jq >/dev/null 2>&1; then
  # Fail open, but not invisibly: without this line a missing jq is byte-identical to "looks fine",
  # and the guard would sit inert forever with nothing to diagnose it by. Both harnesses surface
  # hook stderr.
  echo "guard-bash-output: jq not found; guard inert" >&2
  exit 0
fi

INPUT="$(cat)"
CMD="$(printf '%s' "$INPUT" | jq -r '.tool_input.command // empty' 2>/dev/null)"
[ -z "$CMD" ] && exit 0

# A heredoc's body is an operand, not a command, and file writes legitimately use one. Anchor on
# real heredoc syntax: `<<WORD`, `<< WORD`, `<<'WORD'`, `<<-WORD`. A bare `<<` substring test also
# matched `<<<` and a trailing `# <<`, which made a comment a one-token opt-out.
if printf '%s' "$CMD" | grep -Eq '<<-?[[:space:]]*['"'"'"A-Za-z_]'; then
  exit 0
fi

# Splits into statements (on && || ; newline) and pipeline stages (on |), respecting quotes and
# backslash escapes, then emits one line per stage: KIND SEP tok SEP tok ...
# KIND is H for the first stage of a statement, P for a piped continuation.
SPLIT_AWK='
function flushtok() { if (tok != "") { print "A\t" tok; tok = "" } }
BEGIN { SQ = sprintf("%c", 39); tok = ""; started = 0 }
{ all = all (NR > 1 ? "\n" : "") $0 }
END {
  n = length(all)
  q = ""
  print "S"
  for (i = 1; i <= n; i++) {
    c = substr(all, i, 1)
    if (q != "") {
      if (c == q) { q = "" }
      else if (c == "\\" && q == "\"") { i++; tok = tok substr(all, i, 1) }
      else { tok = tok c }
      continue
    }
    if (c == SQ || c == "\"") { q = c; continue }
    if (c == "\\") { i++; tok = tok substr(all, i, 1); continue }
    two = substr(all, i, 2)
    if (two == "&&" || two == "||") { flushtok(); print "S"; i++; continue }
    if (c == ";" || c == "\n") { flushtok(); print "S"; continue }
    if (c == "|") { flushtok(); print "P"; continue }
    if (c == " " || c == "\t") { flushtok(); continue }
    tok = tok c
  }
  flushtok()
}
'

SUGGESTION=""

# say <claude-text> <codex-text> -- the same bytes ship to both harnesses, but Codex has no Read,
# Grep, or Glob tool. Naming one there is advice the reader cannot act on, the same reason rung 4 of
# the search ladder was reworded for the shared tree.
say() {
  if [ "$FORMAT" = "codex" ]; then SUGGESTION="$2"; else SUGGESTION="$1"; fi
}

# tok_matches <regex> [start] -- does any token from index <start> match, anchored whole?
tok_matches() {
  local re="$1" start="${2:-1}" i
  for ((i = start; i < ${#T[@]}; i++)); do
    [[ "${T[$i]}" =~ ^($re)$ ]] && return 0
  done
  return 1
}

# has_operand <start> [exclude] -- is there a non-flag token, other than `--` or <exclude>?
has_operand() {
  local start="$1" exclude="${2:-}" i tok
  for ((i = start; i < ${#T[@]}; i++)); do
    tok="${T[$i]}"
    [ "$tok" = "--" ] && continue
    [ -n "$exclude" ] && [ "$tok" = "$exclude" ] && continue
    case "$tok" in -*) continue ;; esac
    return 0
  done
  return 1
}

# check_statement -- run the ladder against the tokens in T, set SUGGESTION on the first hit.
check_statement() {
  local first="${T[0]:-}" sub="${T[1]:-}"

  # Output redirected to a file costs no context.
  tok_matches '>|>>|1>|&>' && return 0

  case "$first" in
    git)
      case "$sub" in
        log)
          tok_matches '--oneline|--stat|--shortstat|--numstat|--name-only|--name-status|-[0-9]+|-n[0-9]*|--max-count(=[0-9]+)?' 2 \
            || say "Unbounded 'git log' -- prefer 'git log --oneline -20', or add --stat." \
                   "Unbounded 'git log' -- prefer 'git log --oneline -20', or add --stat."
          ;;
        diff)
          tok_matches '--stat|--name-only|--numstat|--shortstat|--name-status' 2 && return 0
          # A path scope bounds the diff; a revision does not. `git diff main...HEAD` is the
          # largest diff the guard will ever see and is exactly what /j-diff-review runs, so
          # treating any bare word as a scope inverted the rule on its most important case.
          # Existence on disk is the test that separates the two without parsing revspecs.
          local i tok saw_ddash=0
          for ((i = 2; i < ${#T[@]}; i++)); do
            tok="${T[$i]}"
            [ "$tok" = "--" ] && { saw_ddash=1; continue; }
            [ "$saw_ddash" -eq 1 ] && return 0
            [ -e "$tok" ] && return 0
          done
          say "Unbounded 'git diff' -- run 'git diff --stat' first, then diff only the paths that matter." \
              "Unbounded 'git diff' -- run 'git diff --stat' first, then diff only the paths that matter."
          ;;
      esac
      ;;
    cat)
      say "'cat' of a whole file -- prefer the Read tool, or 'sed -n \"START,ENDp\"' for one range." \
          "'cat' of a whole file -- read a bounded range with 'sed -n \"START,ENDp\"' instead."
      ;;
    find)
      tok_matches '-maxdepth|-name|-iname|-path|-newer' 1 \
        || say "Unbounded 'find' -- prefer the Glob tool, or add -maxdepth and -name." \
               "Unbounded 'find' -- add -maxdepth and -name to bound it."
      ;;
    grep | rg)
      tok_matches '-[a-zA-Z]*[lcm][a-zA-Z]*[0-9]*|--files-with-matches|--count|--max-count(=[0-9]+)?' 1 \
        || say "Unbounded '$first' -- prefer the Grep tool, or add -l to list files only." \
               "Unbounded '$first' -- add -l to list files only, or -c to count."
      ;;
    ls)
      tok_matches '-[a-zA-Z]*R[a-zA-Z]*|--recursive' 1 \
        && say "Recursive 'ls -R' -- prefer 'tree -L 2', or a targeted glob." \
               "Recursive 'ls -R' -- prefer 'tree -L 2', or a targeted glob."
      ;;
    tree)
      tok_matches '-L[0-9]*|--level(=[0-9]+)?' 1 \
        || say "Unbounded 'tree' -- add '-L 2' to cap the depth." \
               "Unbounded 'tree' -- add '-L 2' to cap the depth."
      ;;
    pytest)
      # A scope is a path or -k. `-v` is not a scope, so "has any argument" was the wrong predicate.
      tok_matches '-k' 1 || has_operand 1 \
        || say "Whole-suite 'pytest' -- scope it to a file or '-k <name>' while iterating." \
               "Whole-suite 'pytest' -- scope it to a file or '-k <name>' while iterating."
      ;;
    cargo)
      [ "$sub" = "test" ] || return 0
      has_operand 2 \
        || say "Whole-suite 'cargo test' -- scope it to a module path while iterating." \
               "Whole-suite 'cargo test' -- scope it to a module path while iterating."
      ;;
    npm)
      [ "$sub" = "test" ] || return 0
      has_operand 2 \
        || say "Whole-suite 'npm test' -- scope it with '-- --grep <name>' while iterating." \
               "Whole-suite 'npm test' -- scope it with '-- --grep <name>' while iterating."
      ;;
    go)
      [ "$sub" = "test" ] || return 0
      # ./... is the whole module, so it is not a scope.
      tok_matches '-run' 2 || has_operand 2 "./..." \
        || say "Whole-module 'go test' -- scope it to a package or '-run <name>' while iterating." \
               "Whole-module 'go test' -- scope it to a package or '-run <name>' while iterating."
      ;;
  esac
}

# Walk the tokenized stream. `S` opens a statement, `P` opens a piped stage, `A\t<tok>` is a token.
# A statement whose pipeline reaches a bounding tool is exempt. head, tail, and wc cap the line
# count outright; grep, rg, and jq select a subset. sed, awk, sort, uniq, and cut emit one line
# per input line, so they transform without narrowing. An earlier cut called all nine bounding,
# which read `cat big.log | sort` as safe.
T=()
STMT_OPEN=0
STMT_BOUNDED=0
IN_FIRST_STAGE=0
STAGE_HEAD_PENDING=0

flush_statement() {
  [ "$STMT_OPEN" -eq 0 ] && return 0
  [ "$STMT_BOUNDED" -eq 0 ] && [ "${#T[@]}" -gt 0 ] && check_statement
  T=()
  STMT_OPEN=0
  STMT_BOUNDED=0
}

while IFS= read -r line; do
  case "$line" in
    S)
      flush_statement
      [ -n "$SUGGESTION" ] && break
      STMT_OPEN=1
      IN_FIRST_STAGE=1
      STAGE_HEAD_PENDING=0
      ;;
    P)
      IN_FIRST_STAGE=0
      STAGE_HEAD_PENDING=1
      ;;
    A*)
      tok="${line#A	}"
      if [ "$IN_FIRST_STAGE" -eq 1 ]; then
        T[${#T[@]}]="$tok"
      elif [ "$STAGE_HEAD_PENDING" -eq 1 ]; then
        STAGE_HEAD_PENDING=0
        case "$tok" in head | tail | wc | grep | rg | jq) STMT_BOUNDED=1 ;; esac
      fi
      ;;
  esac
done < <(printf '%s' "$CMD" | awk "$SPLIT_AWK")

[ -z "$SUGGESTION" ] && flush_statement
[ -z "$SUGGESTION" ] && exit 0

if [ "$FORMAT" = "codex" ]; then
  jq -n --arg msg "$SUGGESTION" \
    '{hookSpecificOutput: {hookEventName: "PreToolUse", additionalContext: $msg}}'
else
  jq -n --arg msg "$SUGGESTION" '{systemMessage: $msg}'
fi
exit 0
