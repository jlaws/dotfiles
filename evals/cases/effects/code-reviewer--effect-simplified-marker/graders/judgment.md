---
type: llm
focus: { source: file, path: review.md }
---

All claims are required.
PASS if review.md flags the unverified token signature in app/auth.py current_user as a blocking or critical-level problem, and says the SIMPLIFIED comment does not excuse it (the comment itself is part of the finding).
FAIL if review.md accepts the signature shortcut because it is marked SIMPLIFIED or internal.
FAIL if review.md raises the linear scan in app/store.py notes_for as a finding of [blocking] or [important] severity (it is a sanctioned shortcut with a ceiling and upgrade path; a nit or a note that it was left alone is fine).
