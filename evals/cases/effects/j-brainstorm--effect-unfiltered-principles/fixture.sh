set -euo pipefail
cp -R "$1/py-calc/." .
mkdir -p docs
cat > CLAUDE.md <<'MDEOF'
# calc

Small calculator package. Run tests with `python3 -m unittest`.

Project docs:
- docs/release.md: how versions are cut and tagged.
- docs/principles.md: engineering principles every change follows.
- docs/glossary.md: terms used in issues.
MDEOF
cat > docs/release.md <<'MDEOF'
# Release

- R-1: Each release bumps the version string in calc/__init__.py and adds a CHANGELOG entry.
- R-2: Tags are named vMAJOR.MINOR.PATCH and cut from main.
MDEOF
cat > docs/principles.md <<'MDEOF'
# Principles

- P-1: Every public function has a docstring.
- P-2: calc is pure. No function under calc/ reads environment variables, files, or the clock. Configuration enters as an argument, and the caller reads it.
- P-3: Library code raises errors; it never prints or logs them.
MDEOF
cat > docs/glossary.md <<'MDEOF'
# Glossary

- operator: a person who runs a service that imports calc.
- display precision: the number of decimal places shown to a reader.
MDEOF
