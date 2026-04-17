---
name: j-data
description: "Data engineering consultation. Use when building data pipelines, optimizing queries, or designing data platforms. Do NOT use for basic SQL questions (search references/data/ for quick help)."
argument-hint: "<question-or-task>"
---

Load skill `analysis-output-patterns` for output structure rules.

Before starting, gather diagnostic context:

1. **Detect data stack** from config files (dbt_project.yml, airflow.cfg, dagster workspace, prefect.yaml, spark configs).
2. **Identify DB connections** by searching for connection strings, database URLs, or ORM config (sqlalchemy, prisma, drizzle, knex).
3. **Check pipeline definitions** by searching for DAGs, workflows, ETL scripts, or migration files.
4. **Get scope overview** of the target area (if $ARGUMENTS specifies a pipeline or dataset, scope to that; otherwise scan for data/, pipelines/, dags/, migrations/, etl/ directories).

Help with: $ARGUMENTS
