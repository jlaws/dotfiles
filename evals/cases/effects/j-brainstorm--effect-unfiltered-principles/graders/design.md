---
type: llm
focus: { source: file, path: design.md }
---

All claims are required.
PASS if, in the recommended approach, the operator's setting is read by the caller or by code outside calc/, and any rounding code under calc/ receives the precision as an argument.
PASS if design.md compares at least two approaches and recommends one.
FAIL if the recommended approach has any module under calc/ read the precision from an environment variable, a file, or other configuration (for example os.environ or os.getenv inside calc/).
