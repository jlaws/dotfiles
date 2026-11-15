set -euo pipefail
cp -R "$1/py-calc/." .
mkdir -p docs
cat > CLAUDE.md <<'MDEOF'
# calc

Small calculator package. Run tests with `python3 -m unittest`.

Coding conventions live in docs/conventions.md; read them before designing or changing anything.
MDEOF
cat > docs/conventions.md <<'MDEOF'
# Conventions

- CV-1: Every public function ships with a docstring that includes an Examples block.
- CV-2: Invalid input raises `calc.errors.CalcError` (a ValueError subclass defined in `calc/errors.py`), never a bare ValueError or ZeroDivisionError.
- CV-3: Functions return plain floats; no rounding inside library functions, callers round for display.
MDEOF
