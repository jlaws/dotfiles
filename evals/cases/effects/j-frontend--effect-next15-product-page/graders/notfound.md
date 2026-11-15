---
type: llm
focus: { source: file, path: page.tsx }
---

All claims are required.
PASS if page.tsx calls notFound() from next/navigation when the API responds 404.
PASS if the page function is async and awaits params before reading id (not params.id read synchronously).
FAIL if the fetch uses cache: 'no-store' or force-cache together with the revalidate requirement, or the component is marked 'use client'.
