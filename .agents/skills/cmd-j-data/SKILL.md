---
name: cmd-j-data
description: "Use when invoking the j-data workflow."
disable-model-invocation: true
---

# Data Engineering Consultation

Before starting, gather diagnostic context:

1. **Detect data stack** from config files (dbt_project.yml, airflow.cfg, dagster workspace, prefect.yaml, spark configs).
2. **Identify DB connections** by searching for connection strings, database URLs, or ORM config (sqlalchemy, prisma, drizzle, knex).
3. **Check pipeline definitions** by searching for DAGs, workflows, ETL scripts, or migration files.
4. **Get scope overview** of the target area (if the user's provided input specifies a pipeline or dataset, scope to that; otherwise scan for data/, pipelines/, dags/, migrations/, etl/ directories).

For deep data-engineering guidance, delegate to the `data-engineer` agent, passing the diagnostic findings above and the request. It loads its skills (test-driven-development, analysis-output-patterns) and the `.agents/references/data/` library, then returns specific guidance. Verify its output before presenting.

Help with: the user's provided input
