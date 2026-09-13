# Architecture Decision Records

## Concepts

| Term | Meaning |
|------|---------|
| **Architectural Decision (AD)** | A justified design choice that addresses an architecturally significant requirement. |
| **Architecturally Significant Requirement (ASR)** | A requirement important enough that the decision it drives warrants a record. |
| **Architecture Decision Record (ADR)** | A document capturing a single AD and its rationale — context, options, decision, consequences. |
| **Decision Log** | The full collection of ADRs kept for a project (the `docs/adr/` set). |
| **ADR id** | An ADR's path under `docs/adr/` minus the extension — `data/postgres-primary-store`. There are no ADR numbers. |

### Architectural Significance Test

Write an ADR when a decision meets any of these — it is an ASR:
- Affects the system's structure or a component boundary.
- Cross-cutting: touches multiple components or teams.
- Hard or costly to reverse later.
- Involves a non-obvious trade-off worth explaining to future readers.

If none apply, skip the ADR (see the table below).

## ADR Lifecycle

```
Proposed -> Accepted -> Accepted (updated in place, as often as the decision moves)
                     -> Deleted  (the decision no longer exists)
Proposed -> Deleted
```

Status values:
- `status: proposed` — under discussion, not yet decided.
- `status: accepted` — the decision is in force. It stays `accepted` for the ADR's whole life; a
  revised decision is still a decision in force.

There is no `superseded`, `deprecated`, or `rejected` status, because there is no file left to carry
one. **A record whose decision no longer exists is deleted, not archived.** It is dead documentation,
and dead documentation costs every future reader the time to work out that it is dead. Git holds the
history. Anything still worth knowing — why the old approach failed, what it cost — moves into the
successor ADR's `## Ruled Out` table as one dated row.

## When to Write an ADR

| Write ADR | Skip ADR |
|-----------|----------|
| New framework adoption | Minor version upgrades |
| Database technology choice | Bug fixes |
| API design patterns | Implementation details |
| Security architecture | Routine maintenance |
| Integration patterns | Configuration changes |

## Frontmatter

Every ADR opens with a YAML frontmatter block. It is the machine-readable half of the record — status
and dates are read by globbing, not by parsing prose.

```yaml
---
status: accepted        # proposed | accepted
topic: data
created: 2026-03-01     # record first written; never changes
updated: 2026-09-13     # last substantive edit
deciders: ["@name"]
---
```

Three invariants:
- **`created` never changes. It records when the *record* was first written, not when the current
  decision was made.** Those are the same date until the first in-place reversal; after one they
  differ, and the record's date is the one worth keeping — it is when this question first became
  worth deciding. The newest `## Ruled Out` row dates the current decision.
- **`updated` is the date of the last substantive edit.** It is the timeliness signal — a reader
  weighing whether an ADR still describes the system starts here. A new ADR has `updated` equal to
  `created`. Fixing a typo is not substantive; changing what the record asserts is.
- **The path is authoritative for `topic`.** The field mirrors the containing directory so a reader
  holding only the frontmatter still knows the topic. On mismatch the directory wins; fix the field.

There are no `supersedes` or `superseded-by` fields. A superseded ADR is deleted, so there is no file
at either end of the link.

## Keeping an ADR Current

An ADR is a living document. It describes the decision **as it stands today**, not as it was first
written. Keeping it current is the job; a record that disagrees with the system is a defect in the
record.

| Situation | Action | Result |
|-----------|--------|--------|
| Clarification, corrected detail, or new consequence | Edit the text in place; bump `updated` | stays `accepted` |
| The decision changes | Edit `## Decision` and `## Rationale` in place, add the previous approach to `## Ruled Out` with its reason and date, bump `updated` | stays `accepted` |
| A finding invalidates an option, but the decision holds | Add the option to `## Ruled Out` with the finding and date, bump `updated` | stays `accepted` |
| The decision no longer exists — the component, constraint, or question is gone | **Delete the file.** Move anything still worth knowing into the successor's `## Ruled Out` | file removed |
| One decision splits into two, or two merge into one | Write the new ADRs, carry forward what still applies, delete the old file | file removed |

Editing in place is the default, not the exception. Three things make it safe:

- **`created` still records when the record was first written.** The decision's age is not lost by
  editing it.
- **`## Ruled Out` is where reversals go.** One dated row with the reason, not a narrative. It is the
  concise record of what was tried and why it was dropped, and it is the section that keeps an ADR
  honest about its own reversals.
- **Git holds the diff.** `git log -p docs/adr/<topic>/<slug>.md` is the full amendment trail, and it
  costs nothing to carry because it is not in the file.

**Name every deletion in the change summary.** Deleting an ADR needs no separate approval, so the
summary line is what puts it in front of a reviewer. A record that vanishes inside a large diff
with nothing pointing at it is the one loss this model can cause that the diff alone does not
make obvious.

**Never leave a stale ADR standing.** An ADR contradicted by shipped code is worse than no ADR: an
agent reads it as current and plans against it. Update it or delete it in the same change that made
it wrong.

## Templates

Recognized ADR formats include Nygard (the 2011 original), MADR (most widely adopted), the
Y-Statement (one-sentence), and ISO/IEC/IEEE 42010. The templates below cover the common ones.

### Standard ADR (MADR Format)

```markdown
---
status: proposed
topic: [topic]
created: YYYY-MM-DD
updated: YYYY-MM-DD
deciders: ["@name"]
---
# [Title]

## Context
[Why this decision exists. Constraints and requirements as they stand today, not as they once were.]

## Decision Drivers
* **Must have X** for Y reason
* **Should support Z** to reduce complexity

## Decision
We will use **[choice]**.

## Rationale
[Why this choice best fits the decision drivers.]

## Consequences
**Gained**: [benefit]
**Accepted**: [cost/risk]

## Ruled Out
| Idea | Why ruled out | When |
|------|---------------|------|
| [name the idea, never a number] | [reason, with the evidence if there was any] | YYYY-MM-DD |

## Enforcement
- Owner: `path/to/the/file/this/decision/governs`
- Pinned by: `tests/test_x.py::test_y`, or "nothing — convention only"

## Reversal Conditions
[What would have to become true for this decision to be revisited. Concrete enough that a future
reader can tell whether it has happened.]

## Related
- [title](../[topic]/[slug].md)
- [other document](../../path/to/doc.md)
```

### Lightweight ADR

```markdown
---
status: proposed
topic: [topic]
created: YYYY-MM-DD
updated: YYYY-MM-DD
deciders: ["@name"]
---
# [Title]

## Context
[1-2 paragraphs on the problem]

## Decision
[What we decided]

## Consequences
**Gained**: [benefits]
**Accepted**: [costs]
**Mitigations**: [how to address the costs]

## Ruled Out
| Idea | Why ruled out | When |
|------|---------------|------|
| [name the idea] | [reason] | YYYY-MM-DD |

## Reversal Conditions
[What would have to become true for this decision to be revisited.]

## Related
- [title](../[topic]/[slug].md)
```

### Y-Statement Format

```markdown
---
status: proposed
topic: [topic]
created: YYYY-MM-DD
updated: YYYY-MM-DD
deciders: ["@name"]
---
In the context of **[situation]**,
facing **[problem]**,
we decided for **[choice]**
and against **[alternatives]**,
to achieve **[goals]**,
accepting that **[tradeoff]**.
```

(Single-sentence format — no table sections, so it carries neither `## Ruled Out` nor
`## Reversal Conditions`. Use it only where the decision genuinely fits one sentence; revise the
sentence in place and bump `updated`.)

## Naming and Grouping

```
docs/adr/
  README.md            # conventions and this repo's topic list -- not an index
  template.md          # copy this to start an ADR
  data/
    postgres-primary-store.md
    redis-session-cache.md
  api/
    rest-versioning.md
  security/
    oidc-service-auth.md
```

- **Topic** is a single directory level naming the area the decision constrains. Pick names from the repo's own boundaries and record the list in `docs/adr/README.md`. There is no prescribed taxonomy.
- **Slug** is a kebab-case noun phrase naming the decision, unique within its topic. No number, no date, no author.
- **No sequence counter anywhere.** Ordering comes from `created`.
- **Adding an ADR creates exactly one file and edits none.** That is the point of the scheme: a shared counter or a hand-maintained index turns every concurrent ADR write into a merge conflict, because both writers claim the same next number or edit the same index lines.

Two cases still touch shared state, both rarer than writing an ADR:

- **A new topic** adds a line to `docs/adr/README.md`. Adding an ADR to an existing topic does not.
- **The same topic and slug** chosen twice is an add/add conflict. It means two writers recorded the same decision; merge the records rather than renaming one.

Two scaffold files sit beside the topic directories:

- `docs/adr/README.md` — the repo's topic list with one line on what each covers, the scope test, and the keep-it-current rule. It states explicitly that there is no index: discover ADRs with `docs/adr/**/*.md` and read frontmatter.
- `docs/adr/template.md` — the Standard ADR template above, verbatim, ready to copy.

Both scaffold files match `docs/adr/**/*.md` and neither is an ADR. Exclude `README.md` and `template.md` by name whenever that glob is used to enumerate decisions.

There is no archive directory. `docs/adr/**/*.md` minus those two files is the complete set of live
decisions, and every file in it is asserted to be currently true.

## Review Checklist

### Before Submission
- [ ] Context clearly explains the problem
- [ ] Every option that was weighed appears in `## Ruled Out` with its reason and date
- [ ] Trade-offs stated honestly, including the ones the decision accepts
- [ ] Consequences recorded as **Gained** and **Accepted**

### During Review
- [ ] At least 2 senior engineers reviewed
- [ ] Affected teams consulted
- [ ] Security and cost implications documented
- [ ] Reversibility assessed

### After Acceptance
- [ ] frontmatter status/created/updated set
- [ ] `## Related` links resolve, and `## Reversal Conditions` is concrete enough to check
- [ ] `## Enforcement` names a real owner and test, where the template carries that section
- [ ] Team notified
- [ ] Implementation tickets created

### Definition of Done

A decision is done when it has: **evidence** for the choice, the **criteria and alternatives** considered, **agreement** from the deciders, a written **ADR (documentation)**, and a **realization/review plan** (how it gets built and when it is revisited).

## Do's and Don'ts

- **Write early** - before implementation starts
- **Keep short** - 1-2 pages max
- **Be honest about trade-offs** - include real cons
- **Don't number ADRs** - concurrent writers collide on the next number
- **Don't maintain a hand-edited index** - glob `docs/adr/**/*.md` and read frontmatter
- **Keep accepted ADRs current** - edit the Decision in place and bump `updated`
- **Delete an ADR whose decision is gone** - move what still matters into the successor's `## Ruled Out`
- **Record reversals in `## Ruled Out`** - one dated row with the reason, not a narrative
- **Name ruled-out ideas, never number them** - a table row has no anchor, and "Option 2" stops resolving the moment the table is reordered
- **Don't hide failures** - the paths that did not work belong in `## Ruled Out`, not in a deleted draft
- **Don't be vague** - specific decisions, specific consequences

## Architecture Patterns Reference

Catalog of proven backend architecture patterns (Clean Architecture, Hexagonal Architecture, Domain-Driven Design), when not to reach for each, and their pitfalls. See `architecture-patterns.md`.
