---
type: llm
focus: { source: file, path: review.md }
---

All claims are required.
PASS if the review reports that to_cents in billing/money.py truncates fractional amounts (int(amount * 100) turns 19.99 into 1998 cents) as a problem of this change, because refund_partial is the first caller that passes fractional amounts.
PASS if that to_cents finding is not tagged (pre-existing).
PASS if the review reports the logging of order.customer_email in refund_full and tags that finding (pre-existing), since no hunk adds or changes that line.
FAIL if the to_cents finding is tagged (pre-existing), or a finding lacks a file:line.
