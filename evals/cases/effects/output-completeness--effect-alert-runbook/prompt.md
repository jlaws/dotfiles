---
tags: [effects, effect, extended]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
description: "deliver every section in full with no deferral to earlier sections, while a TODO marking work explicitly put out of scope is correct; .claude/skills/output-completeness/SKILL.md; hard because the warning and critical alert pairs invite a same-as-above shortcut for the later procedure, and the escalation teams sit in the alert file rather than the prompt"
---

Write ./runbook.md with a response procedure for every alert defined in alerts/alerts.yaml.

Each procedure must work on its own for an on-call engineer paged at 3 a.m.: its own triage steps, the commands to run, and when and to whom to escalate, with no references to another procedure. An alert the file marks as out of scope gets only a one-line TODO naming its tracking issue.
