"""Markdown section parsing shared by the suites that compare document structure.

`test_agent_config` uses it for reference-tree section parity; `test_adr` uses it for the ADR
section contract. One copy, because the fence state machine has edge cases -- an unterminated
block, a four-backtick fence -- and two copies means a fix lands in one of them.
"""

from __future__ import annotations

import re

# Captures the level too: a section demoted from `##` to `###` in one tree is a structural
# change, and matching on heading text alone would let it through. `[ \t]+` rather than `\s+`
# so an empty `## ` cannot swallow the next paragraph, and `#{1,6}` so a divergent H1 title or
# a deep H5 is not invisible. Trailing hashes are stripped: `## Foo ##` is the same section
# as `## Foo`.
HEADING = re.compile(r"^(#{1,6})[ \t]+(.*?)(?:[ \t]+#+)?[ \t]*$")
FENCE = ("```", "~~~")


def markdown_headings(text: str) -> list[tuple[str, str]]:
    """Headings outside fenced code blocks.

    A fenced block can hold a markdown *example* -- the ADR template inside
    architecture-decision-records.md, the sample CHANGELOG inside changelog-patterns.md. Those
    are body content the parity decision deliberately allows to differ, so counting them as
    structure would both couple the trees where they should be free and report a "section"
    divergence naming a section that does not exist.
    """
    out: list[tuple[str, str]] = []
    fence = ""
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(FENCE):
            token = stripped[:3]
            if not fence:
                fence = token
            elif token == fence:
                fence = ""
            continue
        if fence:
            continue
        match = HEADING.match(line)
        if match:
            out.append((match.group(1), match.group(2)))
    return out


def heading_texts(text: str) -> set[str]:
    """Section names only. The four existing-code-discipline copies nest at different depths --
    `##` in the references, `####` inside the two prompts that paste the body -- so level is not
    comparable across them, but "is this a heading at all" still is."""
    return {name for _level, name in markdown_headings(text)}


def has_unterminated_fence(text: str) -> bool:
    """Whether a fenced block is still open at end of file.

    `markdown_headings` skips everything inside a fence, so an unclosed one silently swallows
    every heading after it and any section check downstream reports a clean document. The parser
    cannot tell that case from a genuinely fence-free tail, so callers that care assert on this
    separately rather than getting a wrong answer quietly.
    """
    fence = ""
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(FENCE):
            token = stripped[:3]
            if not fence:
                fence = token
            elif token == fence:
                fence = ""
    return bool(fence)


def lines_outside_fences(text: str) -> list[tuple[int, str]]:
    """`(1-indexed line number, line)` for every line not inside a fenced code block.

    A fenced block in a document can quote anything -- an older revision of the document itself, a
    sample from another repo -- so a vocabulary ban that scanned it would fail on material the rule
    was never about.
    """
    out: list[tuple[int, str]] = []
    fence = ""
    for number, line in enumerate(text.splitlines(), 1):
        stripped = line.lstrip()
        if stripped.startswith(FENCE):
            token = stripped[:3]
            if not fence:
                fence = token
            elif token == fence:
                fence = ""
            continue
        if not fence:
            out.append((number, line))
    return out
