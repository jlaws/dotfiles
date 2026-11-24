set -euo pipefail
cp -R "$1/ts-todo/." .
cat > README.md <<'EOF'
# ts-todo

A tiny in-memory todo list.

## API

- `addTodo(list, title)`: returns a new list with a todo appended.
- `complete(list, id)`: returns a new list with that todo marked done.

Run the tests with `npm test`.
EOF
cat > CHANGELOG.md <<'EOF'
# Changelog

## Unreleased

## 0.1.0

- Add `addTodo` and `complete`.
EOF
git init -q -b main
git config user.name kbeval
git config user.email kbeval@example.invalid
git add -A
git commit -q -m "Initial commit"
git init -q --bare .remote.git
git remote add origin ./.remote.git
git push -q origin main
printf "%s\n" .remote.git >> .git/info/exclude
