---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

/kb:j-frontend write a Next.js 15 App Router page for the route /products/[id] and save it to ./page.tsx. It is a server component that loads https://api.example.com/products/{id} with fetch and renders the product name and price. Requirements: the data may be at most 60 seconds stale; the fetch is tagged products so it can be revalidated on demand; the route param is read the way Next.js 15 requires; a missing product (404 from the API) shows the not-found page.
