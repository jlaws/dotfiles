set -euo pipefail
cp -R "$1/ts-todo/." .
git init -q -b main
git config user.name kbeval
git config user.email kbeval@example.invalid
printf "%s\n" scratchpad/ > .gitignore
git add -A
git commit -q -m "Initial commit"
git init -q --bare .remote.git
git remote add origin ./.remote.git
git push -q origin main
printf "%s\n" .remote.git >> .git/info/exclude
cat >> src/todo.js <<'EOF'

function indexOfId(list, id) {
  return list.findIndex((item) => item.id === id);
}

export function removeTodo(list, id) {
  const index = indexOfId(list, id);
  if (index === -1) {
    throw new Error(`todo ${id} not found`);
  }
  return [...list.slice(0, index), ...list.slice(index + 1)];
}

export function clearCompleted(list) {
  return list.filter((item) => !item.done);
}
EOF
cat > test/remove.test.js <<'EOF'
import { test } from "node:test";
import assert from "node:assert/strict";
import { removeTodo } from "../src/todo.js";

test("removeTodo drops the item with the given id", () => {
  const list = [{ id: 1, title: "a", done: false }, { id: 2, title: "b", done: false }];
  assert.deepEqual(removeTodo(list, 1), [{ id: 2, title: "b", done: false }]);
});

test("removeTodo throws for an unknown id", () => {
  assert.throws(() => removeTodo([], 7), /not found/);
});
EOF
