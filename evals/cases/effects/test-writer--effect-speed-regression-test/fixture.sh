set -euo pipefail
mkdir -p textkit tests
cat > textkit/__init__.py <<'PY'
PY
cat > textkit/slug.py <<'PY'
"""Slug helpers."""
import re
import unicodedata

_NON_WORD = re.compile(r"[^a-z0-9]+")


def slugify(text):
    """Return a lowercase, hyphen-separated ASCII slug of text."""
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return _NON_WORD.sub("-", ascii_text.lower()).strip("-")


def _slugify_reference(text):
    """The previous implementation, kept as the speed baseline for slugify."""
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    out = ""
    pending_dash = False
    for ch in ascii_text.lower():
        if ch.isascii() and ch.isalnum():
            if pending_dash and out:
                out += "-"
            out += ch
            pending_dash = False
        else:
            pending_dash = True
    return out
PY
cat > tests/__init__.py <<'PY'
PY
cat > tests/test_slug.py <<'PY'
import unittest

from textkit.slug import _slugify_reference, slugify


class SlugifyTest(unittest.TestCase):
    def test_matches_reference(self):
        for text in ["Hello, World!", "  Crème brûlée  ", "a--b__c", "2024 Q3 Report"]:
            self.assertEqual(slugify(text), _slugify_reference(text))

    def test_lowercase_hyphenated(self):
        self.assertEqual(slugify("Hello, World!"), "hello-world")


if __name__ == "__main__":
    unittest.main()
PY
