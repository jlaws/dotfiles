# Hiring and Interviews

## Evaluation Rubrics

### Structured Scoring Template

Score each dimension 1-4. Avoid 5-point scales (the middle becomes a dumping ground).

| Score | Label | Meaning |
|-------|-------|---------|
| 1 | Does not meet | Significant gaps, could not perform at this level |
| 2 | Partially meets | Some capability shown, needs development |
| 3 | Meets | Competent at expected level for the role |
| 4 | Exceeds | Notably strong, above typical for this level |

### Coding Interview Rubric

| Dimension | 1 | 2 | 3 | 4 | Weight |
|-----------|---|---|---|---|--------|
| Problem solving | Can't break down problem | Needs significant hints | Solves with minor hints | Elegant solution, considers edge cases | 30% |
| Code quality | Unreadable, no structure | Works but messy | Clean, well-named, modular | Production-quality, defensive | 25% |
| Communication | Silent or confused | Explains when asked | Thinks aloud naturally | Drives conversation, checks assumptions | 20% |
| Testing mindset | No mention of tests | Tests when prompted | Identifies key test cases | Tests first, edge cases, error paths | 15% |
| Technical depth | Surface-level answers | Knows basics | Explains tradeoffs | Deep knowledge, references real experience | 10% |

### Calibration Sessions

Run calibration before each hiring cycle:

1. Interviewers independently score the same recorded interview (or written scenario)
2. Compare scores; discuss any dimension with >1 point spread
3. Align on what "3 - Meets" looks like for this specific role
4. Document calibrated examples in the rubric

Frequency: once per quarter or when adding new interviewers.

## Take-Home Assessment Design

### Take-Home Anti-Patterns
- "Build a full app from scratch" (too broad, biases toward free time)
- No time limit (candidates spend 20+ hours, creates inequity)
- Hidden evaluation criteria (candidates can't optimize for what matters)
- Requiring a specific framework/language without business justification

## Hiring Metrics

### Core Metrics

| Metric | Target | How to Measure |
|--------|--------|----------------|
| Time to hire | <30 days (eng) | Offer accepted date - req opened date |
| Pipeline velocity | <2 weeks per stage | Average days between stage transitions |
| Pass-through rate | Varies by stage | Candidates advancing / candidates entering stage |
| Offer acceptance rate | >80% | Offers accepted / offers extended |
| Interview-to-offer ratio | 3:1 to 5:1 | Final round interviews / offers made |
| Interviewer load | <4 hrs/week | Interview hours per interviewer per week |
| Candidate NPS | >60 | Post-interview survey (even for rejects) |

### Quality Metrics (Lagging)

| Metric | Timeframe | Signal |
|--------|-----------|--------|
| New hire performance rating | 6 months | Were our assessments predictive? |
| 90-day retention | 3 months | Did we set correct expectations? |
| 1-year retention | 12 months | Culture and role fit assessment quality |
| Time to productivity | 3 months | Onboarding effectiveness (related to hiring) |
| Regretted attrition | Ongoing | Are we losing people we wanted to keep? |

### Using Metrics

- Track pass-through rates by interviewer to detect outliers (too harsh or too lenient)
- If offer acceptance < 70%, investigate comp, speed, or candidate experience
- If time-to-hire > 45 days, audit where candidates stall

## Anti-Bias Practices

### Debrief Structure

```
1. Each interviewer shares:
   - Score per dimension (already submitted)
   - One strongest signal (positive)
   - One concern (if any)

2. Hiring manager synthesizes:
   - Areas of agreement
   - Areas of disagreement (discuss these)
   - Overall hire/no-hire recommendation

3. Decision:
   - Strong hire: ≥3 interviewers at 3+ average
   - Lean hire: mixed signals, discuss specific concerns
   - No hire: ≥2 interviewers below 2.5 average
```

## Gotchas

- **Over-indexing on algorithms**: LeetCode-style questions test a narrow skill. Use them as one signal, not the primary filter. Pair programming or take-homes test more relevant skills.
- **Culture fit bias**: "Culture fit" often means "similar to us." Replace with "values alignment" and define specific values with behavioral indicators.
- **Inconsistent evaluation**: Without rubrics, interviewers anchor on different things. Two interviewers can both say "strong hire" for completely different (even contradictory) reasons.
- **Speed vs. quality tradeoff**: Pressure to fill roles fast leads to lowered bars. Track quality metrics (new hire performance) alongside speed metrics.
- **Take-home inequity**: Candidates with families, multiple jobs, or disabilities may have less discretionary time. Always offer an alternative format (live session with same problem).
- **Interviewer burnout**: Senior engineers doing 6+ hours/week of interviews burn out and give worse evaluations. Cap at 4 hours/week maximum.
- **Feedback black holes**: Candidates who never hear back poison your employer brand. Send rejection emails within 5 business days of decision, always.
- **Panel homogeneity**: All-male, all-senior, all-same-background panels introduce systematic blind spots. Diverse panels catch different signals.
- **Moving the goalposts**: Changing evaluation criteria mid-pipeline because a favored candidate didn't score well. Lock rubrics before interviewing starts.
