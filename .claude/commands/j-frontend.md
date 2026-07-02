---
name: j-frontend
description: "Frontend consultation — frameworks, design systems, and web patterns. Use when building with React, Next.js, Svelte, Tailwind, or solving frontend challenges. Do NOT use for backend API design (use /j-arch instead)."
argument-hint: "<question-or-task>"
---

Load skill `analysis-output-patterns` for output structure rules.
Load skill `output-completeness` when generating full components or multi-file UI code.

Before starting, gather diagnostic context:

1. **Detect frontend framework** from config files (next.config.js, svelte.config.js, vite.config.ts, angular.json) and package.json dependencies.
2. **Identify styling approach** by searching for Tailwind config, CSS modules, styled-components, or design tokens.
3. **Check component structure** for existing component library, storybook config, or design system setup.
4. **Get scope overview** of the target area (if $ARGUMENTS specifies a component, scope to that; otherwise scan for src/components/, pages/, app/, or similar directories).

Load relevant references based on the diagnostic context:
- **Frameworks**: `references/frontend/nextjs-app-router-patterns`, `svelte-patterns`, `react-state-management`, `react-native-architecture` -- framework-specific patterns and state
- **Design systems & styling**: `references/frontend/design-system-patterns`, `tailwind-design-system`, `premium-design-aesthetics`, `web-animation-patterns` -- tokens, component libraries, aesthetics, motion
- **UX quality**: `references/frontend/accessibility-testing`, `responsive-web-design`, `form-patterns`, `i18n-and-localization`, `design-audit` -- a11y, responsive layout, forms, localization, UI review
- **Data layer**: `references/frontend/graphql-client-patterns` -- client caching, queries, mutations

Help with: $ARGUMENTS
