---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
description: "move server data to TanStack Query, keep only UI state in zustand with shallow-safe selectors, and make an optimistic add cancel in-flight queries and roll back on error; .claude/agents/frontend-engineer.md; hard because an optimistic write without cancelQueries is overwritten by a refetch already in flight"
---

Use the kb:frontend-engineer agent to refactor the state handling in ./src/store.js and ./src/TodoList.jsx (leave both unchanged) and save the result as one file, ./src/todos.fixed.jsx, that exports useUiStore, useTodos, useAddTodo, and the TodoList component. Requirements: the todo list, its loading state, and its error come from a server-state library (TanStack Query), not from the client store; the client store holds only UI state (the filter and the selected id); a component re-renders only when a store field it actually uses changes, so no selector may build a new object on each call unless it is wrapped in shallow comparison; adding a todo shows it in the list immediately and rolls back if the request fails, and a refetch already in flight must not overwrite the optimistic entry.
