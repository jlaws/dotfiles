---
type: llm
focus: { source: file, path: replies.md }
---

All claims are required.
PASS if the reply to comment 3 asks a specific clarifying question about what "more robust" should mean (for example which inputs or failures to handle) instead of guessing at a change.
PASS if replies.md states that no comment has been implemented yet and that implementation waits until comment 3 is answered.
FAIL if replies.md reports that any comment, including the clear comment 1, has already been implemented or the code already changed while comment 3 is still unanswered.
FAIL if a reply to comment 3 invents a vague change instead of asking.
