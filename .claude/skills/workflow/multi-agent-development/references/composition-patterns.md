# Team Composition Patterns

#### Review Team

- Lead gathers context (diff, changed files, branch info)
- Distributes full context to specialist reviewers (security, quality, testing, language)
- Each reviewer focuses on one perspective using Explore agents
- Lead merges findings, deduplicates, produces unified report

#### Research Team

- Lead defines scope, distributes query sets
- Each researcher explores different sources/angles in parallel
- Lead synthesizes findings, builds taxonomy or comparison matrix

#### Implementation Team

- Lead creates phased plan with file ownership boundaries
- Teammates implement independent modules in parallel (general-purpose agents)
- Reviewer teammate validates each module
- Lead integrates, runs full test suite

#### Adversarial Team

- Lead poses a question or problem
- Multiple agents investigate competing hypotheses independently
- Agents challenge each other's findings via messaging
- Lead synthesizes with confidence-weighted conclusions
