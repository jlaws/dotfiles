---
type: llm
focus: { source: file, path: review.md }
---

All claims are required.
PASS if the review identifies the mutable default argument installments=[] in split_payments, which shares state across calls.
PASS if the review identifies that split_payments rounds per-installment so the installments can fail to sum to total, or that n == 0 raises ZeroDivisionError.
PASS if the card number logged in charge is reported and tagged (pre-existing), since no hunk adds that line.
PASS if the review contains no compliments (no praise of the change) and lists at most five nits individually.
FAIL if the logging finding is attributed to this change without the pre-existing tag, or a finding has no file:line.
