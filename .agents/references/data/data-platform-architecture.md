# Data Platform Architecture

## Platform Component Selection

| Need | Component | Key Decision |
|------|-----------|-------------|
| Storage + table format | Lakehouse | Iceberg vs Delta vs Hudi (see below) |
| Data validation | Quality framework | Great Expectations vs dbt tests vs SodaCL |
| ML feature serving | Feature store | Feast vs Tecton vs custom (Redis + warehouse) |
| Data contracts | Schema governance | datacontract.com spec + SodaCL checks |
| File maintenance | Compaction / vacuum | Schedule based on write frequency |

## Lakehouse Storage

### Table Format Decision

| Criteria | Apache Iceberg | Delta Lake | Apache Hudi |
|----------|---------------|------------|-------------|
| Multi-engine support | Best (Spark, Trino, Flink, DuckDB) | Good (Spark-native, growing) | Good (Spark, Flink) |
| Schema evolution | Full (add, drop, rename, reorder) | Add/rename columns | Add columns |
| Hidden partitioning | Yes (no partition columns in queries) | No (explicit partition cols) | No |
| Partition evolution | Yes (change without rewrite) | No (requires rewrite) | No |
| Time travel | Snapshot-based, configurable | Version-based, 30-day default | Timeline-based |
| CDC / upserts | Merge-on-read or copy-on-write | MERGE INTO | Built-in (MoR native) |
| Best for | Multi-engine lakehouse | Databricks-centric orgs | CDC-heavy workloads |

### Iceberg Core Operations

Key points:
- **Hidden partitioning**: users write `timestamp`, Iceberg partitions by day automatically
- **Partition evolution**: change strategy without rewriting existing data
- **Schema evolution**: add, drop, rename, reorder columns -- no data rewrite needed

```python
# iceberg_setup.py -- PyIceberg catalog, schema, and table creation
from pyiceberg.catalog import load_catalog
from pyiceberg.schema import Schema
from pyiceberg.types import (
    NestedField,
    StringType,
    LongType,
    TimestampType,
    DoubleType,
)
from pyiceberg.partitioning import PartitionSpec, PartitionField
from pyiceberg.transforms import DayTransform, BucketTransform

catalog = load_catalog(
    "production",
    **{
        "type": "rest",
        "uri": "http://iceberg-rest:8181",
        "s3.endpoint": "http://minio:9000",
        "s3.access-key-id": "admin",
        "s3.secret-access-key": "password",
    },
)

schema = Schema(
    NestedField(1, "event_id", StringType(), required=True),
    NestedField(2, "user_id", StringType(), required=True),
    NestedField(3, "event_type", StringType(), required=True),
    NestedField(4, "timestamp", TimestampType(), required=True),
)
# Hidden partitioning: users write timestamp, Iceberg partitions by day
spec = PartitionSpec(
    PartitionField(source_id=4, field_id=1000, transform=DayTransform(), name="day")
)

table = catalog.create_table(
    "analytics.events",
    schema=schema,
    partition_spec=spec,
)


# --- Schema Evolution ---
table = catalog.load_table("analytics.events")

# Add columns (no rewrite needed)
with table.update_schema() as update:
    update.add_column("amount", DoubleType())
    update.add_column("region", StringType())

# Rename column
with table.update_schema() as update:
    update.rename_column("region", "geo_region")

# Partition evolution: change strategy without rewriting data
with table.update_spec() as update:
    update.add_field(
        "user_bucket",
        BucketTransform(16),
        source_column_name="user_id",
    )


# --- Time Travel ---
table = catalog.load_table("analytics.events")

for snapshot in table.metadata.snapshots:
    print(f"ID: {snapshot.snapshot_id}, Time: {snapshot.timestamp_ms}")

# Read at specific snapshot
df = table.scan(snapshot_id=123456789).to_pandas()

# Read as of timestamp
from datetime import datetime

snap = table.snapshot_as_of_timestamp(
    int(datetime(2025, 1, 15).timestamp() * 1000),
)
df = table.scan(snapshot_id=snap.snapshot_id).to_pandas()


# --- Compaction (via Spark) ---
# spark.sql("""CALL system.rewrite_data_files(
#     table => 'analytics.events', strategy => 'sort',
#     sort_order => 'user_id ASC, timestamp DESC',
#     options => map(
#         'target-file-size-bytes', '134217728',
#         'min-file-size-bytes', '67108864',
#         'max-file-size-bytes', '201326592'))""")
#
# spark.sql("""CALL system.expire_snapshots(
#     table => 'analytics.events',
#     older_than => TIMESTAMP '2025-01-01 00:00:00',
#     retain_last => 10)""")
#
# spark.sql("""CALL system.remove_orphan_files(
#     table => 'analytics.events',
#     older_than => TIMESTAMP '2025-01-01 00:00:00')""")
```

### Delta Lake Core Operations

Key points:
- **MERGE INTO**: full CDC with match/not-matched/not-matched-by-source
- **Z-Order**: multi-dimensional clustering for 2-4 columns (diminishing returns beyond)
- **Change Data Feed**: track insert/update/delete for downstream consumers

```python
# delta_operations.py -- Delta Lake PySpark operations
from pyspark.sql import SparkSession
from delta.tables import DeltaTable

spark = (
    SparkSession.builder.appName("delta-lakehouse")
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config(
        "spark.sql.catalog.spark_catalog",
        "org.apache.spark.sql.delta.catalog.DeltaCatalog",
    )
    .getOrCreate()
)

# --- Write initial table ---
df = spark.read.parquet("s3://raw/events/")
(df.write.format("delta").partitionBy("event_date").mode("overwrite").save("s3://lakehouse/events"))


# --- MERGE INTO (upsert) ---
target = DeltaTable.forPath(spark, "s3://lakehouse/customers")
source = spark.read.parquet("s3://staging/customers_update/")
(
    target.alias("t")
    .merge(source.alias("s"), "t.customer_id = s.customer_id")
    .whenMatchedUpdate(
        set={
            "name": "s.name",
            "email": "s.email",
            "updated_at": "s.updated_at",
        }
    )
    .whenNotMatchedInsert(
        values={
            "customer_id": "s.customer_id",
            "name": "s.name",
            "email": "s.email",
            "updated_at": "s.updated_at",
        }
    )
    .whenNotMatchedBySourceDelete()
    .execute()
)


# --- OPTIMIZE, VACUUM, Change Data Feed ---
dt = DeltaTable.forPath(spark, "s3://lakehouse/events")
dt.optimize().executeCompaction()
dt.optimize().executeZOrderBy("user_id", "event_date")
dt.vacuum(retentionHours=168)  # 7 days

# Enable change data feed
spark.sql("""ALTER TABLE delta.`s3://lakehouse/events`
    SET TBLPROPERTIES (delta.enableChangeDataFeed = true)""")

# Read change feed
changes = (
    spark.read.format("delta")
    .option("readChangeFeed", "true")
    .option("startingVersion", 10)
    .load("s3://lakehouse/events")
)
# _change_type: insert, update_preimage, update_postimage, delete
```

### Query Engine Integration

DuckDB + Iceberg and Trino + Iceberg patterns -- see `references/query-engines.md`.

### Compaction and Maintenance

| Task | Iceberg | Delta Lake |
|------|---------|------------|
| Compact small files | `rewrite_data_files` | `OPTIMIZE` / `executeCompaction` |
| Clustering | Sort via rewrite | Z-Order |
| Remove old versions | `expire_snapshots` | `VACUUM` (min 7 days retention) |
| Orphan cleanup | `remove_orphan_files` | Automatic with VACUUM |

### Lakehouse Gotchas

- **Small file problem** -- frequent writes create thousands of tiny files; schedule compaction
- **VACUUM too aggressive** -- retention below 7 days (Delta) risks breaking concurrent readers
- **Partition cardinality** -- >10K partitions degrades metadata; use bucket transforms
- **Schema evolution with Parquet** -- column renames work in Iceberg (name-based) but break raw Parquet (position-based)
- **Catalog lock contention** -- concurrent writers need atomic commits; use catalog-level locking
- **Cost of time travel** -- every snapshot retains file references; unbounded snapshots bloat metadata
- **Delta outside Databricks** -- UniForm improves compatibility but some features are Databricks-only
- **Partition evolution pitfall** -- old data keeps old layout; spanning queries may scan more files

## Data Quality

### Quality Dimensions

| Dimension | Example Check | Framework |
|-----------|---------------|-----------|
| Completeness | `not_be_null` | GE / dbt `not_null` |
| Uniqueness | `to_be_unique` | GE / dbt `unique` |
| Validity | `to_be_in_set` | GE / dbt `accepted_values` |
| Accuracy | Cross-reference validation | Custom / singular tests |
| Consistency | Column-pair comparisons | GE / dbt `expression_is_true` |
| Timeliness | Freshness check | dbt `recency` / SodaCL `freshness` |

### Framework Selection

| Criteria | Great Expectations | dbt Tests | SodaCL |
|----------|-------------------|-----------|--------|
| Integration | Standalone / Airflow | dbt-native | Standalone / Airflow |
| Test authoring | Python API | YAML + SQL | YAML (SodaCL DSL) |
| Data docs | Built-in HTML reports | None (use Elementary) | Soda Cloud |
| Best for | Complex validation logic | Transform-layer testing | Contract enforcement |

### Great Expectations Suite

Key pattern: build expectation suites covering schema, primary keys, categorical values, numeric ranges, and row counts.

```python
# great_expectations_suite.py -- Suite building and quality pipeline
from great_expectations.core import ExpectationSuite
from great_expectations.core.expectation_configuration import (
    ExpectationConfiguration,
)


def build_orders_suite() -> ExpectationSuite:
    suite = ExpectationSuite(expectation_suite_name="orders_suite")

    # Schema
    suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_table_columns_to_match_set",
            kwargs={
                "column_set": [
                    "order_id",
                    "customer_id",
                    "amount",
                    "status",
                    "created_at",
                ],
                "exact_match": False,
            },
        )
    )
    # Primary key
    suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={"column": "order_id"},
        )
    )
    suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_unique",
            kwargs={"column": "order_id"},
        )
    )
    # Categorical
    suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_in_set",
            kwargs={
                "column": "status",
                "value_set": [
                    "pending",
                    "processing",
                    "shipped",
                    "delivered",
                    "cancelled",
                ],
            },
        )
    )
    # Numeric ranges
    suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_between",
            kwargs={
                "column": "amount",
                "min_value": 0,
                "max_value": 100000,
                "strict_min": True,
            },
        )
    )
    # Row count sanity
    suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_table_row_count_to_be_between",
            kwargs={"min_value": 1000, "max_value": 10000000},
        )
    )
    return suite


# --- Checkpoint configuration (YAML equivalent) ---
# great_expectations/checkpoints/orders_checkpoint.yml
# name: orders_checkpoint
# config_version: 1.0
# validations:
#   - batch_request:
#       datasource_name: warehouse
#       data_asset_name: orders
#     expectation_suite_name: orders_suite
# action_list:
#   - name: store_validation_result
#     action: { class_name: StoreValidationResultAction }
#   - name: update_data_docs
#     action: { class_name: UpdateDataDocsAction }
#   - name: send_slack_notification
#     action:
#       class_name: SlackNotificationAction
#       slack_webhook: ${SLACK_WEBHOOK}
#       notify_on: failure
```

### dbt Data Tests

Built-in + custom generic and singular tests -- see `references/dbt-quality-tests.md`.

Key pattern: combine `unique`, `not_null`, `relationships`, `accepted_values` with custom `expression_is_true` and `recency` tests.

### Data Contracts

```yaml
apiVersion: datacontract.com/v1.0.0
kind: DataContract
metadata:
  name: orders
  version: 1.0.0
  owner: data-platform-team
schema:
  type: object
  properties:
    order_id: { type: string, format: uuid, required: true, unique: true }
    customer_id: { type: string, format: uuid, required: true, pii: true }
    total_amount: { type: number, minimum: 0, maximum: 100000 }
    status: { type: string, enum: [pending, processing, shipped, delivered, cancelled] }
quality:
  type: SodaCL
  specification:
    checks for orders:
      - row_count > 0
      - missing_count(order_id) = 0
      - duplicate_count(order_id) = 0
      - freshness(created_at) < 24h
sla:
  availability: 99.9%
  freshness: 1 hour
  latency: 5 minutes
```

### Automated Quality Pipeline

```python
# --- Automated Quality Pipeline ---
from dataclasses import dataclass
from typing import List, Dict, Any

import great_expectations as gx


@dataclass
class QualityResult:
    table: str
    passed: bool
    total_expectations: int
    failed_expectations: int
    details: List[Dict[str, Any]]


class DataQualityPipeline:
    def __init__(self, context: gx.DataContext):
        self.context = context

    def validate_table(self, table: str, suite: str) -> QualityResult:
        result = self.context.run_checkpoint(
            **{
                "name": f"{table}_validation",
                "config_version": 1.0,
                "class_name": "Checkpoint",
                "validations": [
                    {
                        "batch_request": {
                            "datasource_name": "warehouse",
                            "data_asset_name": table,
                        },
                        "expectation_suite_name": suite,
                    }
                ],
            }
        )
        validation_result = list(result.run_results.values())[0]
        results = validation_result.results
        failed = [r for r in results if not r.success]
        return QualityResult(
            table=table,
            passed=result.success,
            total_expectations=len(results),
            failed_expectations=len(failed),
            details=[
                {
                    "expectation": r.expectation_config.expectation_type,
                    "success": r.success,
                    "observed_value": r.result.get("observed_value"),
                }
                for r in results
            ],
        )

    def run_all(
        self,
        tables: Dict[str, str],
    ) -> Dict[str, QualityResult]:
        return {table: self.validate_table(table, suite) for table, suite in tables.items()}
```

## Feature Store

### Architecture Decision

| Criteria | Feast (OSS) | Tecton | Custom (Redis + Warehouse) |
|----------|------------|--------|---------------------------|
| Setup cost | Low | High (SaaS) | Medium-High |
| Online serving latency | <10ms (Redis) | <5ms | Depends on impl |
| Streaming features | Limited (push-based) | Native Spark/Flink | Build your own |
| Point-in-time joins | Built-in | Built-in | Must implement |
| Team size sweet spot | 2-15 | 15-100+ | 5-20 (eng-heavy) |

**Recommendation**: Feast for most teams -- covers 80% of use cases with minimal operational burden.

### Feast Patterns

Key concepts:
- **Entities**: natural grain keys (`user_id`, `product_id`)
- **BatchFeatureView**: offline-computed features materialized to online store
- **Point-in-time join**: prevents future data leaking into training examples
- **Materialization**: `feast materialize-incremental` syncs offline -> online

```python
# feast_patterns.py -- Feature definitions, serving, and point-in-time joins
from datetime import timedelta, datetime

from feast import Entity, FeatureStore, Field, BatchFeatureView
from feast.data_source import PushMode
from feast.types import Float32, Int64, String
from feast.infra.offline_stores.bigquery_source import BigQuerySource
import pandas as pd


# --- Feature Store Config (feature_store.yaml) ---
# project: my_ml_project
# registry: gs://my-bucket/feast/registry.pb
# provider: gcp
# online_store:
#   type: redis
#   connection_string: redis://10.0.0.5:6379
# offline_store:
#   type: bigquery
# entity_key_serialization_version: 2


# --- Entities ---
user = Entity(
    name="user_id",
    join_keys=["user_id"],
    description="Unique user identifier",
)

product = Entity(
    name="product_id",
    join_keys=["product_id"],
)


# --- Data Sources ---
user_stats_source = BigQuerySource(
    name="user_stats",
    table="ml_features.user_daily_stats",
    timestamp_field="event_date",
    created_timestamp_column="created_at",
)


# --- Feature Views ---
user_features = BatchFeatureView(
    name="user_features",
    entities=[user],
    ttl=timedelta(days=7),
    schema=[
        Field(name="order_count_30d", dtype=Int64),
        Field(name="avg_order_value_30d", dtype=Float32),
        Field(name="days_since_last_order", dtype=Int64),
        Field(name="lifetime_value", dtype=Float32),
        Field(name="preferred_category", dtype=String),
    ],
    source=user_stats_source,
    online=True,
    tags={"team": "recommendations", "version": "v2"},
)


# --- Point-in-Time Retrieval ---
store = FeatureStore(repo_path="feature_repo/")

entity_df = pd.DataFrame(
    {
        "user_id": [42, 99, 42, 17],
        "event_timestamp": pd.to_datetime(
            [
                "2024-03-15 10:00:00",
                "2024-03-15 14:00:00",
                "2024-03-10 08:00:00",  # same user, earlier time = different features
                "2024-03-12 12:00:00",
            ]
        ),
        "label": [1, 0, 1, 0],
    }
)

# Feast handles point-in-time join automatically
training_df = store.get_historical_features(
    entity_df=entity_df,
    features=[
        "user_features:order_count_30d",
        "user_features:avg_order_value_30d",
        "user_features:days_since_last_order",
    ],
).to_df()


# --- Online Serving ---
# Materialize: feast materialize-incremental $(date -u +"%Y-%m-%dT%H:%M:%S")
# Or: store.materialize_incremental(end_date=datetime.utcnow())

features = store.get_online_features(
    features=[
        "user_features:order_count_30d",
        "user_features:avg_order_value_30d",
        "user_features:lifetime_value",
    ],
    entity_rows=[{"user_id": 42}],
).to_dict()
# Returns: {"user_id": [42], "order_count_30d": [5], ...}


# --- Push-Based Streaming Features ---
store.push(
    push_source_name="user_realtime_stats",
    df=pd.DataFrame(
        {
            "user_id": [42],
            "session_duration_sec": [340],
            "pages_viewed": [12],
            "event_timestamp": [datetime.utcnow()],
        }
    ),
    to=PushMode.ONLINE,  # or ONLINE_AND_OFFLINE
)


# --- Detect Training-Serving Skew ---
# online = store.get_online_features(
#     features=feature_list, entity_rows=entities,
# ).to_df()
# offline = store.get_historical_features(
#     entity_df=entity_df_now, features=feature_list,
# ).to_df()
# Assert values match within tolerance
```

### Feature Store Gotchas

| Problem | Symptom | Fix |
|---------|---------|-----|
| Training-serving skew | Model degrades in prod | Define transforms once; compare online vs offline values |
| Time-travel bugs | Backfilled data pollutes history | Use business timestamp, not ingestion timestamp |
| Feature freshness | Stale predictions | Monitor feature age; alert when exceeding TTL |
| Entity key bloat | Slow lookups, high memory | One entity per natural grain; aggressive TTL on session-level |

### Entity Key Design

| Entity Pattern | Online Store Size | Lookup Speed | Use Case |
|---------------|------------------|-------------|----------|
| `user_id` | ~N users | Fast | User-level aggregates |
| `product_id` | ~N products | Fast | Product metadata/stats |
| `(user_id, product_id)` | ~N*M | Slow if M large | Interaction features |
| `session_id` | Unbounded | Degrades | Avoid; use TTL aggressively |
