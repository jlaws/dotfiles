---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
description: "normalize quarterly and yearly plans to a month for MRR and keep the executive page to 4 to 6 headline KPIs, each with a formula, an update frequency, and a strategic, tactical, or operational level; .claude/agents/business-analyst.md; hard because a model tends to overfill an executive page and can sum billed amounts without normalizing them to a month"
---

Use the kb:business-analyst agent to define the executive KPI page for our SaaS product from this month's data. Active subscriptions: 40 on the monthly plan at $30 per month, 10 on the quarterly plan billed $81 per quarter, and 5 on the yearly plan billed $360 per year. Marketing and sales spend this month was $9,000 and it brought in 20 new customers. Save the result to ./kpis.md. Compute MRR (normalizing quarterly and yearly plans to a month), ARR, and CAC, showing the arithmetic. Then list 4 to 6 headline KPIs for a single executive page (MRR, ARR, and CAC may be among them), and for each give its formula, update frequency, and the audience level it serves (strategic, tactical, or operational).
