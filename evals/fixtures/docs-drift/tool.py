"""Convert a CSV file to JSON lines."""

import argparse
import csv
import json
import sys


def main(argv=None):
    """Read CSV from --input and write JSON lines to --output (default: stdout)."""
    parser = argparse.ArgumentParser(description="Convert CSV to JSON lines.", allow_abbrev=False)
    parser.add_argument("--input", required=True, help="CSV file to read")
    parser.add_argument("--output", help="file to write; defaults to stdout")
    args = parser.parse_args(argv)
    out = open(args.output, "w") if args.output else sys.stdout
    with open(args.input, newline="") as handle:
        for row in csv.DictReader(handle):
            out.write(json.dumps(row) + "\n")
    if args.output:
        out.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
