---
type: llm
focus: { source: file, path: src/todos.fixed.jsx }
---

All claims are required.
PASS if the zustand store created in the file holds only filter and selectedId (and their setters), with no todos, loading, or error fields and no fetch calls.
PASS if every zustand selector either selects a single field or is wrapped in useShallow (or a shallow equality function), with no bare object-literal selector.
PASS if useAddTodo is a useMutation whose onMutate calls cancelQueries on the todos query before writing the optimistic entry with setQueryData, and onError restores the previous data saved in onMutate.
FAIL if the TodoList component still calls fetch directly inside a useEffect.
