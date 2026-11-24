---
status: accepted
topic: workflow
created: 2026-11-23
updated: 2026-11-24
---
# Skill descriptions are tuned against held-out triggers

## Context
The 2026-11-16 baseline found should-fire recall of about 60% on Sonnet and 72% on Opus. Several
skills fired in none of their should-fire runs. `evals/README.md` then recorded "Descriptions stay
unchanged", because description length did not predict firing. Length was the wrong axis: a
description can be short and still name the situations a user is in, or long and name none of them.
Each skill has one should-fire trigger case, so any rewrite tuned on it fits that one prompt.

## Decision Drivers
* **Must measure recall on prompts the tuner never saw**, or a rewrite fits its own test
* **Must not trade precision for recall**: near-miss prompts stay unfired
* **Must not change what the skill does** once it fires
* **Should keep the shared `.agents` descriptions** inside the Codex 64-character budget

## Decision
We will tune a `.claude` skill description that misses its should-fire cases **for content, not
length**, against held-out trigger cases. Each measured skill gains one tuning case (`tune-` slug;
the existing unprefixed cases also count as tuning cases) and two held-out cases (`held-` slug: one
should-fire, one near-miss) that a separate author writes and the tuner never reads. A new
description is adopted only when all of these hold:
- the held-out should case gains at least 4 of 12 runs across both models;
- the existing should case does not drop;
- every near-miss case stays at zero fires;
- the skill's effect cases show no regression;
- no untuned skill's trigger cases move on Sonnet: no near case fires, and no should case fires in
  2 or more fewer runs than the stored baseline. When one moves and the kept trace names the tuned
  skill that fired instead, that description is restored. A drop with no tuned skill firing in its
  place is recorded with the result and decided by the owner.

Skills reached only through an agent preload or a command, whose trigger prompts cannot exercise
them, are not tuned.

## Rationale
A gain of 4 of 12 runs separates a real change from run-to-run noise. With no real change, a second
Binomial(12, p) draw gains 4 or more over the first in at most 7.6% of cases (one-sided, at p =
0.5), and far less at the low base rates these skills start from. A separate author for the held-out
cases keeps the tuner from fitting them. The effect check guards against a description that leads
the model away from the body; `writing-skills` still bans a workflow summary in a description, and
review enforces it.

## Consequences
**Gained**: a description change carries a held-out before and after on both models.
**Accepted**: `.claude` descriptions diverge from the shared ones. Across many skills the chance
that some no-change description passes adds up, so each adoption is a single-skill bet, not a
family-wise guarantee. A skill whose held-out base is 9 of 12 or higher cannot gain 4 and is
restored by rule. Each measured skill carries three more trigger cases, which adds about 21 USD to a
full two-model baseline. The should-fire mean is not comparable with baselines taken before the new
cases existed. A case audit does not edit `held-` cases; it files findings on them. The written
report of the first full baseline after an adoption states each adopted description's held-out
should-fire count, read from that baseline's trigger rows; no command prints it as one figure.

## Ruled Out
| Idea | Why ruled out | When |
|------|---------------|------|
| Tune on the single existing case | Fits one prompt; says nothing about the next request | 2026-11-23 |
| Adopt on any gain | A one-run difference is noise at three runs per model | 2026-11-23 |
| Change the shared `.agents` descriptions | Codex holds them to 64 characters | 2026-11-23 |
| Add a `when_to_use` field | A second field for the job `description` already does | 2026-11-23 |
| Tune commands | Commands are typed, not chosen from the listing | 2026-11-23 |
| Mechanize the neighbour call | The call rests on a kept trace that shows which skill took the request, and counts alone cannot show that. The tool prints the neighbour lines and the owner decides | 2026-11-23 |

## Enforcement
- Owner: `evals/README.md` (case-authoring rules, Asset determination)
- Pinned by: `python3 -m evals.kbeval.hillclimb adopt` and `tests/test_evals.py::HillclimbAdoptTests`

## Reversal Conditions
- Held-out recall for an adopted description does not hold on the next full baseline.
- Claude Code changes how the skill listing is built, so descriptions are no longer what the model
  reads to choose a skill.

## Related
- [KB evals run on a staged plugin](kb-evals-run-on-a-staged-plugin.md)
- [evals/README.md](../../../evals/README.md)
