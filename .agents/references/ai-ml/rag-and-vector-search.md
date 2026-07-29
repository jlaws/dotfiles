# RAG and Vector Search

## Embedding Model Selection

| Model | Dims | Params | Best For |
|-------|------|--------|----------|
| text-embedding-3-large | 3072 | -- | Highest quality API (OpenAI); Matryoshka dim reduction |
| text-embedding-3-small | 1536 | -- | Cost-effective default (OpenAI) |
| voyage-3 | 1024 | -- | Code, legal, finance (API) |
| gte-Qwen2-7B-instruct | 3584 | 7B | Best open-source overall (MTEB leader) |
| e5-mistral-7b-instruct | 4096 | 7B | Instruction-tuned, long context |
| bge-large-en-v1.5 | 1024 | 335M | Strong English, efficient |
| nomic-embed-text-v1.5 | 768 | 137M | Matryoshka, open weights, good quality/size |
| all-MiniLM-L6-v2 | 384 | 22M | Fast prototyping, lightweight |
| multilingual-e5-large | 1024 | 560M | 100+ languages (requires query/passage prefixes) |

**Decision rule**: API embeddings (text-embedding-3-large, voyage-3) for simplicity. Open-source when you need fine-tuning, data privacy, or cost control at scale. For sub-100ms latency, use MiniLM or distilled models.

### Matryoshka Embeddings
Models like text-embedding-3-large support dimension reduction: truncate vectors to 256/512/1024 dims with minimal quality loss. Reduces storage 3-12x. Test recall at target dimension before committing.

**Never mix embedding models** in the same index -- vectors from different models are incompatible.

## Bi-Encoder vs Cross-Encoder

| Aspect | Bi-Encoder | Cross-Encoder |
|--------|-----------|---------------|
| Speed | Fast (encode once, compare many) | Slow (re-encode each pair) |
| Accuracy | Good | Better (10-15% higher on STS) |
| Use case | Retrieval (first stage) | Reranking (second stage) |
| Scaling | O(n) encode + O(1) compare | O(n) per query |

Always use bi-encoder for retrieval, cross-encoder for reranking. See `ai-ml/rag-and-vector-search/embedding-fine-tuning.md` for the two-stage pipeline code.

## Embedding Fine-Tuning

### When to Fine-Tune
- Base model retrieval < 70% NDCG@10 on your domain
- Domain-specific vocabulary (legal, medical, code)
- Custom similarity definition (e.g., "similar" means same author, not same topic)

### Contrastive Loss Selection

| Loss | Requires | When |
|------|----------|------|
| InfoNCE / MultipleNegativesRankingLoss | Positive pairs + in-batch negatives | Default choice |
| Triplet loss | (anchor, positive, negative) | Explicit hard negatives available |
| Cosine similarity loss | Pairs + graded similarity (0-1) | Graded relevance labels |
| GISTEmbed | Guided in-batch negatives | Teacher model can filter false negatives |

### Key Levers
- **Hard negatives**: Single biggest quality lever. Mine with BM25 or cross-encoder. Easy negatives teach nothing after epoch 1.
- **Batch size**: Larger = more in-batch negatives = better signal. Use 64-256; gradient accumulation or GradCache for limited VRAM.
- **TSDAE pre-training**: Unsupervised domain adaptation when you have domain text but no labeled pairs. Fine-tune with labeled pairs after.
- **Evaluation**: Use MTEB (STS + retrieval tasks). Always compare against base model. Fine-tuning for retrieval can hurt STS and vice versa.

### Gotchas
- **Query/doc prefixes**: E5 and BGE models require `"query: "` / `"passage: "` prefixes. Missing = 5-15% retrieval drop. Check model card.
- **Normalization**: Always normalize before cosine similarity. `encode(..., normalize_embeddings=True)` or `F.normalize(embeddings, p=2, dim=1)`.
- **Catastrophic forgetting**: Low learning rate (2e-5), few epochs (1-3), evaluate on general benchmarks alongside domain benchmarks.
- **Index rebuild**: After fine-tuning, re-encode entire corpus and rebuild index. Old embeddings are incompatible.
- **Dims vs quality**: 768-dim BGE often outperforms 3072-dim models on specific domains after fine-tuning. Evaluate on your data.

Code examples for all patterns: `ai-ml/rag-and-vector-search/embedding-fine-tuning.md`

## Chunking Decisions

| Strategy | When |
|----------|------|
| Token-based (512-1000) | Default; predictable size |
| Semantic/header-based | Markdown/structured docs; preserves logical units |
| Recursive character | Unstructured text; LangChain default |
| Parent-child | Need small chunks for retrieval precision, large for LLM context |

- **Chunk size**: 500-1000 tokens default; smaller for precision, larger for context
- **Overlap**: 10-20% to avoid losing boundary context
- **Always test** chunk size impact on retrieval quality for your specific corpus

## Distance Metrics

| Metric | When |
|--------|------|
| Cosine | Default; works with normalized embeddings |
| Dot Product | When magnitude carries meaning |
| Euclidean (L2) | Raw/unnormalized embeddings |

## Index Selection by Scale

| Vector Count | Index Type | Notes |
|-------------|-----------|-------|
| < 10K | Flat (exact) | No approximation needed |
| 10K-1M | HNSW | Good recall/speed tradeoff |
| 1M-100M | HNSW + INT8 quantization | Reduces memory ~4x |
| > 100M | IVF + PQ or DiskANN | Trades recall for scale |

## HNSW Tuning

| Scale | M | efConstruction | efSearch (95% recall) | efSearch (99% recall) |
|-------|---|---------------|----------------------|----------------------|
| < 100K | 16 | 100 | 64 | 128 |
| < 1M | 32 | 200 | 128 | 256 |
| > 1M | 48 | 256 | 128 | 256 |

Higher M = better recall but more memory. Memory per vector: `dimensions * bytes_per_dim + M * 2 * 4 bytes`.

## Vector Database Selection

| DB | Strength | Best For |
|----|----------|----------|
| pgvector | Already using Postgres; hybrid FTS+vector | Small-medium scale, simplicity |
| Qdrant | Filtering, quantization, Rust perf | Production workloads needing metadata filters |
| Weaviate | GraphQL API, multi-modal, hybrid built-in | Multi-modal search, rapid prototyping |
| Pinecone | Fully managed, zero ops | Teams without infra capacity |
| Turbopuffer | S3-backed, cost-effective at scale | Large-scale with cold storage economics |
| Elasticsearch 8.x | Existing ES stack; native RRF | Hybrid search with mature text search |

## Retrieval Architecture

### Hybrid Search (Preferred for Production)
Combine dense (vector) + sparse (BM25/FTS) retrieval. Two fusion approaches:

- **RRF (Reciprocal Rank Fusion)**: Works well without tuning, robust default. Score = sum of `1/(k + rank)` across result lists, k=60.
- **Linear combination**: More control but requires tuning alpha. Normalize scores before combining.

### Reranking (Always Worth It)
- Retrieve 20-50 candidates with hybrid search
- Rerank with cross-encoder (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`)
- **Cohere Rerank API**: managed option, supports `rerank-english-v3.0` and multilingual
- **ColBERT / late-interaction**: token-level matching, better for long documents than bi-encoder reranking
- Return top 3-5 to LLM
- For diversity: use MMR (`lambda_mult=0.5` balances relevance vs diversity)

### pgvector + FTS Pattern
Store embeddings and `tsvector` in same table. Use CTE with RRF to combine vector similarity rank and text search rank in a single query.

## Advanced RAG Patterns

### GraphRAG
Build knowledge graph from documents, then traverse graph relationships during retrieval. Best for corpora with rich entity relationships (legal, biomedical, enterprise docs). Use with Neo4j or networkx.

### Contextual Retrieval (Anthropic Pattern)
Prepend chunk-specific context before embedding: "This chunk is from section X of document Y and discusses Z." Improves retrieval by 20-67% on benchmarks. Compute once at index time.

### Proposition-Based Chunking
Decompose documents into atomic propositions ("The Eiffel Tower is in Paris", "It was completed in 1889") instead of fixed-size chunks. Better precision for fact-lookup tasks. Higher indexing cost.

### Late Chunking
Embed the full document first (using long-context model), then pool token embeddings into chunks. Preserves cross-chunk context that gets lost with chunk-then-embed.

## RAG Pipeline Opinions

### Retrieval
- **Multi-query retrieval**: generate 3-5 query variations for better recall on ambiguous questions
- **Parent document retriever**: index small chunks, return parent context to LLM
- **Contextual compression**: extract only relevant portions from retrieved docs before sending to LLM
- **Metadata filtering**: always index source, timestamp, category; filter at query time to reduce noise

### Generation
- Always include citation markers ([1], [2]) in prompt template
- Ask for confidence score; instruct model to say "I don't have enough information" when context is insufficient
- Evaluate groundedness: NLI-based check that response is entailed by retrieved context

### Evaluation Metrics
- **Retrieval**: Precision@K, Recall@K, MRR, NDCG
- **Generation**: Groundedness (NLI), faithfulness, answer relevance
- Test retrieval and generation independently; don't just evaluate end-to-end

## Quantization Tradeoffs

| Type | Size Reduction | Recall Impact | When |
|------|---------------|---------------|------|
| FP16 | 2x | Negligible | Default for GPU |
| INT8 scalar | 4x | < 1% loss | Production default |
| Product Quantization | 16-32x | 2-5% loss | Memory-constrained, > 100M vectors |
| Binary | 32x | Significant | First-pass candidate filtering only |

## Memory Estimation
`total_bytes = num_vectors * (dimensions * bytes_per_dim + M * 2 * 4)`

Example: 1M vectors, 1536 dims, FP32, M=16 = ~6.1 GB vectors + ~128 MB index overhead.

## Cross-References

- **ai-ml:llm-application-patterns** -- prompt engineering, agent patterns, production deployment
- **ai-ml:llm-application-patterns#structured-output** -- extracting structured data from retrieved documents
