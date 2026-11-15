---
type: llm
focus: { source: file, path: claudemd-proposal.md }
---

All claims are required.
PASS if the proposal identifies the conflict between "Be concise" and "Explain every step in full detail" and recommends removing or rewording the second, citing the sessions where the user pushed back.
PASS if the proposal recommends removing "Write clean, readable code" and "Use descriptive variable names" as restating default behavior.
PASS if the proposal flags the E1042 import-job rule as too specific to be a durable CLAUDE.md rule, suggesting removal or a move elsewhere.
PASS if the proposal adds a rule to paste test output rather than only saying tests pass, supported by both sessions.
FAIL if the proposal recommends keeping both conflicting instructions unchanged, or says it edited CLAUDE.md.
