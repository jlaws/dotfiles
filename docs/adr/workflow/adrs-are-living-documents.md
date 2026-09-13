---
status: accepted
topic: workflow
created: 2026-09-13
updated: 2026-09-13
deciders: ["@jlaws"]
---
# ADRs are living documents, and dead ones are deleted

## Context

`references/architecture/architecture-decision-records.md` ran on the immutability model inherited
from the Nygard/MADR tradition. It stated "An Accepted ADR's **Decision is immutable**" and
"Superseded and deprecated ADRs stay in the log; they are immutable history, not deletions."

Two costs follow from that, both paid on every read:

- **Retired records stay on disk.** Every reversal adds a file and keeps the old one. A reader
  globbing `docs/adr/**/*.md` cannot tell a live decision from a dead one without opening each and
  checking `status`.
- **Every ADR carries an append-only `## Amendment Log`.** It grows monotonically with edits and
  duplicates what git already stores.

Both optimize for auditing how the system arrived here. The decision log's actual job in this repo is
the opposite: an agent planning a change needs the current reasoning in one read. History is the rare
query; "what do we do today" is the constant one.

## Decision Drivers

* **An agent must find current reasoning in one read** — no status-filtering pass, no archive to skip.
* **The log must not grow monotonically** — its size should track the number of live decisions, not
  elapsed time.
* **History must stay recoverable** — just not at a cost paid by every reader.

## Decision

ADRs are **living documents**. Edit the Decision in place when it changes, bump `updated`, and record
the previous approach as one dated row in `## Ruled Out`.

**Delete the file when the decision it records no longer exists.** There is no archive directory and
no retired status; a dead ADR is dead documentation and gets removed the way dead code does. Anything
still worth knowing moves into the successor's `## Ruled Out`.

Agents delete autonomously and name the deletion in their summary. Git plus PR review are the safety
net.

## Rationale

Git already stores the full amendment trail, addressable per file with `git log -p`. Storing it a
second time inside the record is a cost paid on every read for a benefit claimed on almost none.

The same argument disposes of the archive. A tombstone file answers "what did we used to think",
which `git log` answers better, while imposing "is this one still live?" on every reader of every
glob. Deleting is not information loss; it is moving the information to where its cost matches its
use.

What editing in place would genuinely lose is the reasoning behind a reversal, and `## Ruled Out`
catches exactly that — one dated row naming the idea and why it failed. That section is the reason
the rest of the history is safe to let go of.

## Consequences

**Gained**: `docs/adr/**/*.md` minus two scaffold files is the complete set of live decisions, every
one asserted to be currently true. Log size tracks live decisions, not elapsed time. Per-ADR heading
count drops from sixteen to nine.

**Accepted**: No in-file audit trail, so `git log -p` is the only amendment history — a force-pushed
history rewrite could lose it. Autonomous deletion means a dead ADR can disappear inside a larger
doc-sync diff, so PR review is doing real work here. And "the decision no longer exists" is a
judgment call rather than a mechanical test.

## Ruled Out

| Idea | Why ruled out | When |
|------|---------------|------|
| Archive directory for retired ADRs | A tombstone directory is dead docs with extra steps: it still has to be excluded from every glob, and it answers a question `git log` answers better. | 2026-09-13 |
| Bounded amendment log (keep the N most recent rows) | Still duplicates git, still grows to its cap, and the cap is arbitrary. Buys no property the `updated` field does not already give. | 2026-09-13 |
| Unbounded amendment log | The status quo being replaced. Grows monotonically inside the document whose job is to stay short. | 2026-09-13 |
| Dropping `## Implementation Notes` | Proposed during planning, then reversed on reading the content: all three existing ADRs used the section for enforcement pointers (owner file, pinning test, known gaps), not implementation detail. Renamed `## Enforcement` instead, which is what it was doing. | 2026-09-13 |
| Fresh ADR with a new `created` on full reversal | Preserves the literal reading of `created` as "when the decision was made", but reintroduces file churn on exactly the changes that matter most. `created` was redefined to mean "when the record was first written" instead. | 2026-09-13 |

## Enforcement

- Owner: `references/architecture/architecture-decision-records.md`, mirrored byte-for-byte in
  `.claude/` and `.agents/`.
- Pinned by `tests/test_adr.py`, which fails on a retired frontmatter key, a status outside
  `{proposed, accepted}`, a retired section heading, an `updated` earlier than `created`, a `topic`
  that disagrees with its directory, an option cited by number, and any drift between
  `docs/adr/template.md` and the spec's Standard ADR block.
- Consumers carrying the edit-in-place rule: `post-ship-doc-sync` (three trees) and `j-arch` (four
  trees).

## Reversal Conditions

Reverse if an external consumer starts depending on stable ADR file paths, so that deleting a record
breaks something outside this repo. Reverse the git-holds-the-history half if a compliance
requirement demands an in-repo audit trail independent of version control.

## Related
- [Reference trees share a section set, not a body](reference-tree-section-parity.md)
- [Thorough about quality dimensions, lazy about new scope](code-ladder-vs-completeness.md)
- [Structure preference is conditional](../context-efficiency/structure-is-conditional.md)
