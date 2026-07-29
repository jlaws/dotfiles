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
