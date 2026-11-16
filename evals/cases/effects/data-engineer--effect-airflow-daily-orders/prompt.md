---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

Use the kb:data-engineer agent to fix the Airflow DAG in ./dags/daily_orders.py and save the corrected DAG to ./dags/daily_orders_fixed.py (leave the original unchanged). Requirements: running a given day twice leaves the same rows in daily_orders, with no duplicates; the day it loads is the run's logical date, never a hardcoded or current-time date; the scheduler parsing the file must not open a database connection or run a query; the file sensor must free its worker slot between checks; a failed day must not block later days from running; the sensor and the load task both have timeouts; the DAG has a fixed start_date.
