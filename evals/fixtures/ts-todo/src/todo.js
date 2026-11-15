// A tiny in-memory todo list.

export function addTodo(list, title) {
  if (!title || !title.trim()) {
    throw new Error("title is required");
  }
  return [...list, { id: list.length + 1, title: title.trim(), done: false }];
}

export function complete(list, id) {
  return list.map((item) => (item.id === id ? { ...item, done: true } : item));
}
