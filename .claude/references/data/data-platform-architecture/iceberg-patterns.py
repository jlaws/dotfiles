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
