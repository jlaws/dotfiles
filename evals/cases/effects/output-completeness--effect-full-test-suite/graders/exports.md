---
type: regex
pattern: 'export function clearCompleted.*export function search|export function search.*export function clearCompleted'
flags: s
target: { source: file, path: src/todo.js }
---
