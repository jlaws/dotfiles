---
type: llm
focus: { source: file, path: signup.jsx }
---

All claims are required.
PASS if age is converted from a string to a number before the integer and 13 to 120 checks (for example z.coerce.number().int().min(13).max(120)), so the FormData path does not fail on the string type.
PASS if signupAction reads the checkboxes so that a missing terms entry fails validation and a missing marketing entry is valid (for example formData.get('terms') === 'on' and a false default).
PASS if the username check is debounced or otherwise throttled instead of running on every keystroke.
FAIL if signupAction throws on invalid input instead of returning an errors object, or if any non-checkbox input gets both a register spread and a value prop (a checkbox's value attribute is fine).
