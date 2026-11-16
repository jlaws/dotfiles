set -euo pipefail
mkdir -p src
cat > src/store.js <<'JS'
import { create } from 'zustand';

export const useStore = create((set, get) => ({
  todos: [],
  loading: false,
  error: null,
  filter: 'all',
  selectedId: null,
  fetchTodos: async () => {
    set({ loading: true });
    try {
      const res = await fetch('/api/todos');
      set({ todos: await res.json(), loading: false });
    } catch (e) {
      set({ error: e.message, loading: false });
    }
  },
  addTodo: async (title) => {
    const res = await fetch('/api/todos', {
      method: 'POST',
      body: JSON.stringify({ title }),
    });
    const todo = await res.json();
    set({ todos: [...get().todos, todo] });
  },
  setFilter: (filter) => set({ filter }),
  select: (id) => set({ selectedId: id }),
}));
JS
cat > src/TodoList.jsx <<'JSX'
import { useEffect } from 'react';
import { useStore } from './store';

export function TodoList() {
  const { todos, loading, error, filter, fetchTodos, addTodo, setFilter } = useStore(
    (s) => ({
      todos: s.todos,
      loading: s.loading,
      error: s.error,
      filter: s.filter,
      fetchTodos: s.fetchTodos,
      addTodo: s.addTodo,
      setFilter: s.setFilter,
    })
  );

  useEffect(() => {
    fetchTodos();
  }, []);

  if (loading) return <p>Loading...</p>;
  if (error) return <p>{error}</p>;
  const shown = todos.filter((t) => filter === 'all' || t.status === filter);
  return (
    <div>
      <select value={filter} onChange={(e) => setFilter(e.target.value)}>
        <option value="all">All</option>
        <option value="open">Open</option>
        <option value="done">Done</option>
      </select>
      <button onClick={() => addTodo('New todo')}>Add</button>
      <ul>
        {shown.map((t) => (
          <li key={t.id}>{t.title}</li>
        ))}
      </ul>
    </div>
  );
}
JSX
