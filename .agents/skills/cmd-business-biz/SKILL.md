---
name: cmd-business-biz
description: "Business strategy consultation — KPI definition, MVP planning, payment flow design, and team processes. Use when defining KPIs, planning MVPs, designing payment flows, or team processes. Do NOT use for technical architecture decisions (use /cmd-architecture-arch instead)."
disable-model-invocation: true
---

# Business Strategy Consultation

Before starting, gather diagnostic context:

1. **Detect project type** from config files (package.json, pyproject.toml, Gemfile) — web app, API, mobile, library, etc.
2. **Check analytics tooling** by searching for analytics/tracking integrations (Segment, Mixpanel, Amplitude, PostHog, GA config).
3. **Identify KPI definitions** by searching for dashboard configs, metrics definitions, or reporting modules.
4. **Get scope overview** of the target area (if the user specifies a domain, scope to that; otherwise scan for business-logic directories like billing/, analytics/, reports/).

Help with the business strategy topic specified by the user.
