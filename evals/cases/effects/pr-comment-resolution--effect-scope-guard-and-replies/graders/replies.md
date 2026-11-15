---
type: llm
focus: { source: file, path: replies.md }
---

All claims are required.
PASS if there is a reply for each of comments 101, 102, 103 and 104.
PASS if the reply to 101 cites the commit and what the test run showed, not only that it was addressed.
PASS if the reply to 103 explains the empty-list contract (the docstring promises 0.0) without promising a code change.
FAIL if any reply says only that a comment was addressed or fixed with no observed evidence.
