---
type: llm
focus: { source: file, path: .coveragerc }
---

All claims are required.
PASS if branch coverage is enabled (branch = True) and fail_under is set to 80.
PASS if the tests directory is not measured (source limited to the calc package, or tests omitted) and the generated calc/_rates.py is omitted.
PASS if the excluded lines cover `if TYPE_CHECKING:` blocks.
FAIL if calc/fmt.py, calc/ops.py, or calc/types.py is omitted from measurement; they are shipped code.
