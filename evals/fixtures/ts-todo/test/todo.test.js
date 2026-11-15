import { test } from "node:test";
import assert from "node:assert/strict";
import { addTodo } from "../src/todo.js";

test("addTodo appends a trimmed, open item", () => {
  const list = addTodo([], "  buy milk ");
  assert.deepEqual(list, [{ id: 1, title: "buy milk", done: false }]);
});

test("addTodo rejects an empty title", () => {
  assert.throws(() => addTodo([], "   "), /title is required/);
});
