---
name: expert-consensus-building
description: "Orchestrate multi-expert panels for complex decisions through structured debate, weighted voting, and conflict resolution."
---

# Expert Consensus Building

## Overview

When complex decisions require multiple perspectives, orchestrate expert panels to debate options, surface trade-offs, and reach well-reasoned consensus. Use structured processes to handle disagreement productively.

## The Process

### 1. Panel Composition

Select experts based on the decision type:

| Decision Type | Panel Composition |
|--------------|-------------------|
| Architecture | system-architect, backend-architect, frontend-architect |
| Security | security-auditor, backend-security-coder, architect-review |
| Performance | performance-engineer, database-architect, backend-architect |
| New Feature | requirements-analyst, system-architect, test-automator |
| Tech Stack | tech-stack-researcher, system-architect, relevant-language-pro |
| Refactoring | refactoring-expert, architect-review, test-automator |

**Panel Size:** 3-5 experts optimal (odd number to avoid ties)

### 2. Structured Debate

**Round 1: Independent Analysis**
Each expert provides their assessment without seeing others:
- Recommended approach
- Key trade-offs identified
- Risks and concerns
- Confidence level (0.0-1.0)

**Round 2: Cross-Examination**
Experts respond to each other's positions:
- Points of agreement
- Points of disagreement with reasoning
- Questions for other experts

**Round 3: Position Refinement**
Experts may update their positions based on discussion:
- Updated recommendation (if changed)
- Remaining concerns
- Final confidence level

### 3. Weighted Voting

Not all expert opinions carry equal weight for every decision:

**Domain Relevance Weighting:**
```
Expert weight = base_expertise × domain_relevance
```

Example for database migration decision:
- database-architect: 1.0 × 1.0 = 1.0
- system-architect: 0.9 × 0.8 = 0.72
- frontend-architect: 0.9 × 0.3 = 0.27

**Confidence-Weighted Scoring:**
```
Score = Σ (expert_weight × confidence × vote)
```

### 4. Conflict Resolution

When experts disagree:

**Minor Disagreement (< 0.3 score difference):**
- Document both perspectives
- Go with majority recommendation
- Note minority concerns for monitoring

**Major Disagreement (> 0.3 score difference):**
1. Identify root cause of disagreement
2. Request additional evidence from each side
3. Look for synthesis (both could be partially right)
4. Escalate to user if unresolved

**Synthesis Patterns:**
- **Scope split**: "Use A for X, B for Y"
- **Phased approach**: "Start with A, migrate to B later"
- **Conditional**: "Use A if [condition], otherwise B"
- **Hybrid**: Combine elements from multiple approaches

### 5. Consensus Documentation

Record the decision with full context:

```markdown
## Decision: [Topic]

**Panel:** [List of experts consulted]

**Options Considered:**
1. [Option A] - Advocated by: [experts]
2. [Option B] - Advocated by: [experts]

**Consensus:** [Chosen approach]
**Confidence:** [Weighted score]

**Key Trade-offs:**
- [Pro 1] vs [Con 1]
- [Pro 2] vs [Con 2]

**Dissenting Views:**
- [Expert X] preferred [Option Y] because [reason]

**Monitoring:**
- [What to watch for that would trigger reconsideration]
```

## Panel Interaction Patterns

### Delphi Method (Anonymous)
- Experts don't see each other's responses initially
- Reduces anchoring bias
- Good for contentious decisions

### Round-Robin
- Each expert presents in turn
- Others respond before next presentation
- Ensures all voices heard equally

### Devil's Advocate
- Assign one expert to argue against majority
- Stress-tests the consensus
- Surfaces hidden assumptions

## Key Principles

- **Diverse perspectives**: Include experts who might disagree
- **Structured process**: Don't let loudest voice win
- **Weighted relevance**: Domain experts carry more weight
- **Document dissent**: Minority opinions are valuable signal
- **Revisit triggers**: Define what would reopen the decision
- **Synthesis over compromise**: Look for approaches that address multiple concerns
