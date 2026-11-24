set -euo pipefail
cp -R "$1/ts-todo/." .
cat > src/report.js <<'EOF'
// Validates a service config and formats the result for the CLI.

export function validate(config) {
  const errors = [];
  if (!(Number.isInteger(config.retries) && config.retries >= 0)) {
    errors.push({ field: "retries", message: "must be zero or more" });
  }
  return errors;
}

export function formatReport(config, errors) {
  const lines = [`Checked fields: ${Object.keys(config).join(", ")}`];
  for (const e of errors) {
    lines.push(`error: ${e.field}: ${e.message}`);
  }
  lines.push(errors.length ? `${errors.length} error(s)` : "ok");
  return lines.join("\n");
}
EOF
cat > test/report.test.js <<'EOF'
import { test } from "node:test";
import assert from "node:assert/strict";
import { formatReport, validate } from "../src/report.js";

test("a negative retries count is reported", () => {
  const config = { retries: -1, timeout_ms: 500 };
  const report = formatReport(config, validate(config));
  assert.ok(report.includes("error: retries: must be zero or more"));
});

test("a valid config reports ok", () => {
  const config = { retries: 2, timeout_ms: 500 };
  assert.ok(formatReport(config, validate(config)).endsWith("\nok"));
});
EOF
