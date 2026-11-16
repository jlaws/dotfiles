---
type: llm
focus: { source: file, path: review.md }
---

All claims are required.
PASS if review.md flags that the .skip on test_discount_rejects_over_100 hides a real gap: apply_discount has no check that pct is at most 100, so the skipped test would fail, and the review asks for the code fix and the test to be restored rather than accepting the skip.
PASS if review.md flags the coverage threshold drop in ci.cfg (fail_under 90 to 60) as lowering a quality bar that needs a stated reason or reversal.
PASS if review.md notes the added noqa comment changes nothing about behavior and silences a checker without fixing a finding.
FAIL if review.md approves the patch as a fix, or gives the skip, the threshold drop, or the noqa no finding at all.
