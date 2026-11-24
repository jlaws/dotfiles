---
type: llm
focus: { source: file, path: test/report.test.js }
---

All claims are required.
PASS if test/report.test.js has a new test for a config with a non-positive timeout_ms whose assertion checks the timeout_ms error itself: the error line for timeout_ms (text starting error: timeout_ms, whole or as a pattern for that line), a parsed report line, or the field of an entry returned by validate().
FAIL if the only assertion about timeout_ms checks that the report text contains timeout_ms (for example includes("timeout_ms") or a /timeout_ms/ pattern), which the Checked fields line at the top of every report already satisfies when the check is missing.
PASS if the two existing tests are still present; their includes and endsWith checks match whole lines that appear nowhere else in the report, so they are not counted as defects.
