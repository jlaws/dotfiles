# Data Pipeline Architecture

## Architecture Patterns

| Pattern | Best For |
|---------|----------|
| **ETL** | Structured data, known schemas |
| **ELT** | Data lakes, schema-on-read |
| **Lambda** | Mixed latency requirements |
| **Kappa** | Real-time processing |
| **Lakehouse** | Modern unified platforms |

## Batch Ingestion

- Incremental loading with watermark columns
- Retry logic with exponential backoff
- Schema validation and dead letter queue
- Metadata tracking (`_extracted_at`, `_source`)

## Streaming Ingestion

- Kafka consumers with exactly-once semantics
- Manual offset commits within transactions
- Windowing for time-based aggregations
- Error handling and replay capability

## Storage Strategy

### Delta Lake
- ACID transactions, upsert with predicate-based matching
- Time travel, optimize (compact small files), Z-order clustering

### Apache Iceberg
- Partition/sort optimization, MERGE INTO for upserts
- Snapshot isolation, file compaction with binpack strategy

## Cost Optimization

- **Partitioning**: date/entity-based, keep >1GB per partition
- **File sizes**: 512MB-1GB for Parquet
- **Lifecycle**: hot (Standard) -> warm (IA) -> cold (Glacier)
- **Compute**: spot for batch, on-demand for streaming, serverless for adhoc
- **Query**: partition pruning, clustering, predicate pushdown

## Batch Pipeline Example

```python
ingester = BatchDataIngester(config={})

df = ingester.extract_from_database(
    connection_string='postgresql://host:5432/db',
    query='SELECT * FROM orders',
    watermark_column='updated_at',
    last_watermark=last_run_timestamp
)

schema = {'required_fields': ['id', 'user_id'], 'dtypes': {'id': 'int64'}}
df = ingester.validate_and_clean(df, schema)

dq = DataQualityFramework()
result = dq.validate_dataframe(df, suite_name='orders_suite', data_asset_name='orders')

delta_mgr = DeltaLakeManager(storage_path='s3://lake')
delta_mgr.create_or_update_table(df=df, table_name='orders', partition_columns=['order_date'], mode='append')

ingester.save_dead_letter_queue('s3://lake/dlq/orders')
```
