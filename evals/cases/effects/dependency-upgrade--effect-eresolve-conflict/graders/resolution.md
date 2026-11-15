---
type: llm
focus: { source: file, path: resolution.md }
---

All claims are required.
PASS if it identifies that enzyme-adapter-react-16 peers on react ^16 while the project is on react 18, so the conflict is a dev dependency that has no React 18 support.
PASS if the recommended resolution removes or replaces the incompatible adapter (for example migrating tests to React Testing Library) or moves to a compatible release, rather than suppressing the check.
PASS if it says --legacy-peer-deps (or --force) is a workaround that hides the conflict and is not the recommended fix.
PASS if any override it suggests is documented with a comment on why it exists, and it says to commit the lockfile and use a frozen install (npm ci) in CI.
FAIL if it recommends --legacy-peer-deps or --force as the resolution.
