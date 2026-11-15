---
type: llm
focus: { source: file, path: Cart.svelte }
---

All claims are required.
PASS if the optional footer is a Snippet prop rendered with {@render ...} and not a <slot>.
PASS if the total is computed with $derived or $derived.by and button handlers use onclick attributes.
FAIL if any Svelte 4 construct (export let, $: labels, on:click, slots, createEventDispatcher) appears.
