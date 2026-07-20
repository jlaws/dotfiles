---
name: j-frontend
description: "Frontend consultation — frameworks, design systems, and web patterns. Use when building with React, Next.js, Svelte, Tailwind, or solving frontend challenges. Do NOT use for backend API design (use $cmd-j-arch instead)."
argument-hint: "<question-or-task>"
---

Before starting, gather diagnostic context:

1. **Detect frontend framework** from config files (next.config.js, svelte.config.js, vite.config.ts, angular.json) and package.json dependencies.
2. **Identify styling approach** by searching for Tailwind config, CSS modules, styled-components, or design tokens.
3. **Check component structure** for existing component library, storybook config, or design system setup.
4. **Get scope overview** of the target area (if $ARGUMENTS specifies a component, scope to that; otherwise scan for src/components/, pages/, app/, or similar directories).

For deep frontend guidance, delegate to the `frontend-engineer` agent, passing the diagnostic findings above and the request. It loads its skills (language-testing-patterns, output-completeness) and the `.agents/references/frontend/` library, then returns specific guidance. Verify its output before presenting.

Help with: $ARGUMENTS
