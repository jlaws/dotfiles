---
name: bash-defensive-patterns
description: Defensive Bash patterns for production scripts. Use when writing robust shell scripts, CI/CD pipelines, or system utilities.
---

# Bash Defensive Patterns

Non-obvious safety patterns for production Bash scripts.

## Style Guide

Source: Google Shell Style Guide. Only rules linters/formatters cannot enforce.

### Naming
- Functions: `lower_with_under()` — descriptive verb phrases (`check_deps`, `parse_args`)
- Local variables: `lower_with_under`
- Constants/env vars: `UPPER_WITH_UNDER`
- Source filenames: `lower_with_under.sh`
- Executables: no extension (or `.sh`); libraries: must have `.sh`

### Practices
- Only bash for scripts <100 lines; rewrite larger in Python/Go
- File header comment required: description of contents
- Function comments: description, globals used/modified, arguments, outputs, return values
- `main` function pattern: all functions at top, `main "$@"` at bottom
- Separate `local` declaration from command substitution: `local val; val=$(cmd)`
- No `eval`, no aliases in scripts (use functions)
- All error messages to STDERR

## Always Start With

```bash
#!/bin/bash
set -Eeuo pipefail
trap 'echo "Error on line $LINENO" >&2' ERR
trap 'rm -rf -- "$TMPDIR"' EXIT
```

- `-E`: ERR trap inherited by functions
- `-e`: exit on error
- `-u`: exit on undefined variable
- `-o pipefail`: pipe fails if any command fails

## Variable Safety

```bash
# Always quote variables
cp "$source" "$dest"

# Required variable with message
: "${REQUIRED_VAR:?REQUIRED_VAR is not set}"

# Default value for optional
: "${OPTIONAL_VAR:=default_value}"

# Test safely (:-} prevents -u from triggering)
if [[ -z "${VAR:-}" ]]; then echo "unset"; fi
```

## Safe Iteration

```bash
# NUL-delimited find (handles spaces, newlines in filenames)
while IFS= read -r -d '' file; do
    echo "Processing: $file"
done < <(find "$dir" -type f -print0)

# Read into array safely
mapfile -t lines < <(some_command)
```

## Script Directory Detection

```bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
```

## Argument Parsing Template

```bash
VERBOSE=false; DRY_RUN=false; OUTPUT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -v|--verbose) VERBOSE=true; shift ;;
        -d|--dry-run) DRY_RUN=true; shift ;;
        -o|--output)  OUTPUT="$2"; shift 2 ;;
        -h|--help)    usage 0 ;;
        --)           shift; break ;;
        *)            echo "Unknown: $1" >&2; usage 1 ;;
    esac
done
```

## Dry-Run Pattern

```bash
run_cmd() {
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY RUN] $*"; return 0
    fi
    "$@"
}

run_cmd cp "$source" "$dest"
```

## Dependency Checking

```bash
check_deps() {
    local -a missing=()
    for cmd in "$@"; do
        command -v "$cmd" &>/dev/null || missing+=("$cmd")
    done
    if [[ ${#missing[@]} -gt 0 ]]; then
        echo "Missing: ${missing[*]}" >&2; return 1
    fi
}
check_deps jq curl git
```

## Atomic File Writes

```bash
atomic_write() {
    local target="$1"
    local tmp; tmp=$(mktemp) || return 1
    cat > "$tmp"
    mv "$tmp" "$target"  # atomic on same filesystem
}
echo "content" | atomic_write /path/to/file
```

## Structured Logging

```bash
log_info()  { echo "[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $*" >&2; }
log_error() { echo "[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $*" >&2; }
log_debug() { [[ "${DEBUG:-0}" == "1" ]] && echo "[$(date +'%Y-%m-%d %H:%M:%S')] DEBUG: $*" >&2; }
```

## Key Gotchas

- **Use `[[ ]]` not `[ ]`** -- safer, supports `&&`, `||`, regex
- **Use `command -v` not `which`** -- POSIX-compliant, no path surprises
- **Use `printf` not `echo`** -- predictable across systems (echo -n, -e vary)
- **Use `local -r`** for function-scoped constants
- **Close channels from sender** -- use sentinel values for queues
- **Idempotent design** -- scripts should be safe to rerun (`mkdir -p`, check before create)
