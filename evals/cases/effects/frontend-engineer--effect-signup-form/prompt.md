---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

Use the kb:frontend-engineer agent to write a React 19 signup form with React Hook Form and Zod and save it as one file, ./signup.jsx. Fields: username, age (a number input), a marketing opt-in checkbox, and a required terms checkbox. Requirements: the username availability check calls the provided async checkUsername(name) function, which returns a boolean, but must not fire a request on every keystroke; age arrives as text from the input and must be validated as an integer from 13 to 120; the file also exports a server action signupAction(prevState, formData) that parses FormData with the same Zod schema (remember that unchecked checkboxes are absent from FormData; an unchecked marketing box is valid, an unchecked terms box is not) and returns a structured object { ok, errors } instead of throwing; register uncontrolled inputs without also passing them a value prop; each error message is linked to its input for screen readers.
