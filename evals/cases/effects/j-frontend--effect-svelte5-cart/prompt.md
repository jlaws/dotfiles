---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

/kb:j-frontend write a Svelte 5 component to ./Cart.svelte using TypeScript. It takes an items prop (array of {name: string, price: number, qty: number}) and an onCheckout callback prop, shows each line with a remove button, shows the total, and renders optional footer content passed in by the parent. Use current Svelte 5 idioms throughout: reactive state, derived total, props, event handling, and the way children content is passed.
