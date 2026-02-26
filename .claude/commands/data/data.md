---
description: "Data engineering consultation — launches data-engineer subagent. Use when building data pipelines, optimizing queries, or designing data platforms."
---

Before invoking the subagent, gather diagnostic context:

1. **Detect data stack** from config files (dbt_project.yml, airflow.cfg, dagster workspace, prefect.yaml, spark configs).
2. **Identify DB connections** by searching for connection strings, database URLs, or ORM config (sqlalchemy, prisma, drizzle, knex).
3. **Check pipeline definitions** by searching for DAGs, workflows, ETL scripts, or migration files.
4. **Get scope overview** of the target area (if $ARGUMENTS specifies a pipeline or dataset, scope to that; otherwise scan for data/, pipelines/, dags/, migrations/, etl/ directories).

Use the data-engineer subagent to help with: $ARGUMENTS
