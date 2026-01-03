---
name: vector-database-patterns
description: Implement and optimize vector databases for semantic search. Use when building similarity search, implementing RAG retrieval, tuning HNSW parameters, or scaling vector infrastructure.
---

# Vector Database Patterns

Patterns for implementing and optimizing vector databases in production systems.

## When to Use This Skill

- Building semantic search systems
- Implementing RAG retrieval
- Creating recommendation engines
- Tuning HNSW parameters
- Implementing quantization
- Scaling to millions of vectors
- Optimizing search latency and recall

## Core Concepts

### 1. Distance Metrics

| Metric | Formula | Best For |
|--------|---------|----------|
| **Cosine** | 1 - (A·B)/(‖A‖‖B‖) | Normalized embeddings |
| **Euclidean (L2)** | √Σ(a-b)² | Raw embeddings |
| **Dot Product** | A·B | Magnitude matters |

### 2. Index Types

```
Data Size           Recommended Index
────────────────────────────────────────
< 10K vectors  →    Flat (exact search)
10K - 1M       →    HNSW
1M - 100M      →    HNSW + Quantization
> 100M         →    IVF + PQ or DiskANN
```

### 3. HNSW Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| **M** | 16 | Connections per node, ↑ = better recall, more memory |
| **efConstruction** | 100 | Build quality, ↑ = better index, slower build |
| **efSearch** | 50 | Search quality, ↑ = better recall, slower search |

### 4. Quantization Types

```
Full Precision (FP32): 4 bytes × dimensions
Half Precision (FP16): 2 bytes × dimensions
INT8 Scalar:           1 byte × dimensions
Product Quantization:  ~32-64 bytes total
Binary:                dimensions/8 bytes
```

---

## Vector Database Implementations

### Pinecone

```python
from pinecone import Pinecone, ServerlessSpec
from typing import List, Dict, Optional

class PineconeVectorStore:
    def __init__(self, api_key: str, index_name: str, dimension: int = 1536):
        self.pc = Pinecone(api_key=api_key)

        if index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        self.index = self.pc.Index(index_name)

    def upsert(self, vectors: List[Dict], namespace: str = "") -> int:
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            self.index.upsert(vectors=vectors[i:i + batch_size], namespace=namespace)
        return len(vectors)

    def search(self, query_vector: List[float], top_k: int = 10,
               namespace: str = "", filter: Optional[Dict] = None) -> List[Dict]:
        results = self.index.query(
            vector=query_vector, top_k=top_k, namespace=namespace,
            filter=filter, include_metadata=True
        )
        return [{"id": m.id, "score": m.score, "metadata": m.metadata}
                for m in results.matches]
```

### Qdrant

```python
from qdrant_client import QdrantClient
from qdrant_client.http import models

class QdrantVectorStore:
    def __init__(self, url: str, collection_name: str, vector_size: int = 1536):
        self.client = QdrantClient(url=url)
        self.collection_name = collection_name

        collections = self.client.get_collections().collections
        if collection_name not in [c.name for c in collections]:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size, distance=models.Distance.COSINE
                ),
                quantization_config=models.ScalarQuantization(
                    scalar=models.ScalarQuantizationConfig(
                        type=models.ScalarType.INT8, quantile=0.99, always_ram=True
                    )
                )
            )

    def search(self, query_vector: List[float], limit: int = 10,
               filter: Optional[models.Filter] = None) -> List[Dict]:
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector, limit=limit, query_filter=filter
        )
        return [{"id": r.id, "score": r.score, "payload": r.payload} for r in results]
```

### pgvector (PostgreSQL)

```python
import asyncpg
import numpy as np

class PgVectorStore:
    def __init__(self, connection_string: str):
        self.connection_string = connection_string

    async def init(self):
        self.pool = await asyncpg.create_pool(self.connection_string)
        async with self.pool.acquire() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    content TEXT,
                    metadata JSONB,
                    embedding vector(1536)
                )
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS documents_embedding_idx
                ON documents USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64)
            """)

    async def search(self, query_embedding: List[float], limit: int = 10) -> List[Dict]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT id, content, metadata,
                       1 - (embedding <=> $1::vector) as similarity
                FROM documents
                ORDER BY embedding <=> $1::vector
                LIMIT $2
            """, query_embedding, limit)
        return [dict(row) for row in rows]
```

---

## Performance Tuning

### HNSW Parameter Recommendations

```python
def recommend_hnsw_params(num_vectors: int, target_recall: float = 0.95) -> dict:
    if num_vectors < 100_000:
        m, ef_construction = 16, 100
    elif num_vectors < 1_000_000:
        m, ef_construction = 32, 200
    else:
        m, ef_construction = 48, 256

    ef_search = 256 if target_recall >= 0.99 else 128 if target_recall >= 0.95 else 64

    return {"M": m, "ef_construction": ef_construction, "ef_search": ef_search}
```

### Memory Estimation

```python
def estimate_memory_usage(num_vectors: int, dimensions: int,
                          quantization: str = "fp32", hnsw_m: int = 16) -> dict:
    bytes_per_dim = {"fp32": 4, "fp16": 2, "int8": 1, "pq": 0.05, "binary": 0.125}
    vector_bytes = num_vectors * dimensions * bytes_per_dim[quantization]
    index_bytes = num_vectors * hnsw_m * 2 * 4  # Graph edges
    total = vector_bytes + index_bytes

    return {
        "vector_storage_mb": vector_bytes / 1024 / 1024,
        "index_overhead_mb": index_bytes / 1024 / 1024,
        "total_gb": total / 1024 / 1024 / 1024
    }
```

### Quantization

```python
class VectorQuantizer:
    @staticmethod
    def scalar_quantize_int8(vectors: np.ndarray) -> tuple:
        min_val, max_val = vectors.min(), vectors.max()
        scale = 255.0 / (max_val - min_val)
        quantized = np.clip(np.round((vectors - min_val) * scale), 0, 255).astype(np.uint8)
        return quantized, {"min_val": min_val, "max_val": max_val, "scale": scale}

    @staticmethod
    def binary_quantize(vectors: np.ndarray) -> np.ndarray:
        binary = (vectors > 0).astype(np.uint8)
        n, dim = vectors.shape
        packed = np.zeros((n, (dim + 7) // 8), dtype=np.uint8)
        for i in range(dim):
            packed[:, i // 8] |= (binary[:, i] << (i % 8))
        return packed
```

### Optimized Collection Configuration (Qdrant)

```python
def create_optimized_collection(client, name: str, size: int,
                                optimize_for: str = "balanced"):
    configs = {
        "recall": {"m": 32, "ef": 256, "quantization": None},
        "speed": {"m": 16, "ef": 64, "quantization": "int8"},
        "balanced": {"m": 16, "ef": 128, "quantization": "int8"},
        "memory": {"m": 8, "ef": 64, "quantization": "pq"}
    }
    cfg = configs[optimize_for]

    client.create_collection(
        collection_name=name,
        vectors_config=models.VectorParams(size=size, distance=models.Distance.COSINE),
        hnsw_config=models.HnswConfigDiff(m=cfg["m"], ef_construct=cfg["ef"]),
        quantization_config=models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(type=models.ScalarType.INT8)
        ) if cfg["quantization"] == "int8" else None
    )
```

---

## Performance Monitoring

```python
from dataclasses import dataclass
import numpy as np
import time

@dataclass
class SearchMetrics:
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    recall: float
    qps: float

class VectorSearchMonitor:
    def measure_search(self, search_fn, query_vectors: np.ndarray,
                       k: int = 10, iterations: int = 100) -> SearchMetrics:
        latencies = []
        for _ in range(iterations):
            for query in query_vectors:
                start = time.perf_counter()
                search_fn(query, k=k)
                latencies.append((time.perf_counter() - start) * 1000)

        latencies = np.array(latencies)
        total_time = sum(latencies) / 1000

        return SearchMetrics(
            latency_p50_ms=np.percentile(latencies, 50),
            latency_p95_ms=np.percentile(latencies, 95),
            latency_p99_ms=np.percentile(latencies, 99),
            recall=0,  # Calculate with ground truth if available
            qps=(iterations * len(query_vectors)) / total_time
        )
```

---

## Best Practices

### Do's
- **Use appropriate index** - HNSW for most cases, flat for <10K vectors
- **Tune parameters** - ef_search for recall/speed tradeoff
- **Use quantization** - Significant memory savings with minimal recall loss
- **Monitor recall continuously** - Can degrade with data drift
- **Benchmark with real queries** - Synthetic may not represent production

### Don'ts
- **Don't over-optimize early** - Profile first, tune later
- **Don't ignore build time** - Index updates have cost
- **Don't skip evaluation** - Measure before optimizing
- **Don't forget warming** - Cold indexes are slow

## Resources

- [Pinecone Docs](https://docs.pinecone.io/)
- [Qdrant Docs](https://qdrant.tech/documentation/)
- [pgvector](https://github.com/pgvector/pgvector)
- [HNSW Paper](https://arxiv.org/abs/1603.09320)
- [ANN Benchmarks](https://ann-benchmarks.com/)
