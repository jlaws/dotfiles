#!/usr/bin/env bash
# Sequential search for the test that creates unwanted files/state
# Requires bash, python3 (stdlib glob), and npm.
# Usage: ./find-polluter.sh <file_or_dir_to_check> <test_pattern>
# Example: ./find-polluter.sh 'tmp/unexpected-test-output' 'src/**/*.test.ts'

set -e

if [ $# -ne 2 ]; then
  echo "Usage: $0 <file_to_check> <test_pattern>"
  echo "Example: $0 'tmp/unexpected-test-output' 'src/**/*.test.ts'"
  exit 1
fi

POLLUTION_CHECK="$1"
TEST_PATTERN="$2"

echo "🔍 Searching for test that creates: $POLLUTION_CHECK"
echo "Test pattern: $TEST_PATTERN"
echo ""

# Expand recursive globs relative to the current directory, preserving filenames.
TEST_FILES=()
while IFS= read -r -d '' TEST_FILE; do
  TEST_FILES+=("$TEST_FILE")
done < <(python3 - "$TEST_PATTERN" <<'PY'
import glob
import os
import sys

for path in sorted(set(glob.glob(sys.argv[1], recursive=True))):
    if os.path.isfile(path):
        sys.stdout.buffer.write(os.fsencode(path) + b'\0')
PY
)
TOTAL=${#TEST_FILES[@]}

echo "Found $TOTAL test files"
echo ""

if [ "$TOTAL" -eq 0 ]; then
  echo "No test files matched: $TEST_PATTERN"
  exit 2
fi

COUNT=0
for TEST_FILE in "${TEST_FILES[@]}"; do
  COUNT=$((COUNT + 1))

  # An existing target prevents attributing its creation to a test.
  if [ -e "$POLLUTION_CHECK" ] || [ -L "$POLLUTION_CHECK" ]; then
    echo "Pollution already exists before test $COUNT/$TOTAL: $POLLUTION_CHECK"
    echo "Inconclusive: the target must be absent before running tests."
    exit 2
  fi

  echo "[$COUNT/$TOTAL] Testing: $TEST_FILE"

  # Run the test
  TEST_STATUS=0
  npm test "$TEST_FILE" > /dev/null 2>&1 || TEST_STATUS=$?

  # Check if pollution appeared
  if [ -e "$POLLUTION_CHECK" ] || [ -L "$POLLUTION_CHECK" ]; then
    echo ""
    echo "🎯 FOUND POLLUTER!"
    echo "   Test: $TEST_FILE"
    echo "   Created: $POLLUTION_CHECK"
    echo ""
    echo "Pollution details:"
    ls -la "$POLLUTION_CHECK"
    echo ""
    echo "To investigate:"
    echo "  npm test $TEST_FILE    # Run just this test"
    echo "  cat $TEST_FILE         # Review test code"
    exit 1
  fi

  if [ "$TEST_STATUS" -ne 0 ]; then
    echo "Test command failed (exit $TEST_STATUS): npm test $TEST_FILE"
    echo "Inconclusive: no pollution appeared, but the test command did not pass."
    exit 2
  fi
done

echo ""
echo "✅ No polluter found - all tests clean!"
exit 0
