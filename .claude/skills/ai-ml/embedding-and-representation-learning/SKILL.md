---
name: embedding-and-representation-learning
description: "Use when fine-tuning embeddings, implementing contrastive learning, evaluating on MTEB, choosing bi-encoder vs cross-encoder, or building semantic search systems."
---

# Embedding and Representation Learning

## Model Selection

| Model | Dims | Params | Best For |
|-------|------|--------|----------|
| **text-embedding-3-large** | 3072 | -- | Highest quality API embedding (OpenAI) |
| **voyage-3** | 1024 | -- | Code, legal, finance (API) |
| **gte-Qwen2-7B-instruct** | 3584 | 7B | Best open-source overall (MTEB leader) |
| **bge-large-en-v1.5** | 1024 | 335M | Strong English, efficient |
| **all-MiniLM-L6-v2** | 384 | 22M | Fast prototyping, lightweight |
| **e5-mistral-7b-instruct** | 4096 | 7B | Instruction-tuned, long context |
| **nomic-embed-text-v1.5** | 768 | 137M | Matryoshka, open weights, good quality/size |
| **multilingual-e5-large** | 1024 | 560M | 100+ languages |

**Decision rule**: Use API embeddings (text-embedding-3-large, voyage-3) for simplicity. Use open-source when you need fine-tuning, data privacy, or cost control at scale. For sub-100ms latency, use MiniLM or distilled models.

## Contrastive Learning Losses

| Loss | Requires | When |
|------|----------|------|
| **InfoNCE / NT-Xent** | Positive pairs + in-batch negatives | Default choice, sentence-transformers `MultipleNegativesRankingLoss` |
| **Triplet loss** | (anchor, positive, negative) | When you have explicit negatives |
| **Cosine similarity loss** | Pairs + similarity labels (0-1) | When you have graded similarity |
| **Contrastive loss** | Pairs + binary labels | When you have same/different pairs |
| **GISTEmbed** | Guided in-batch negatives | When a teacher model can filter false negatives |

### MultipleNegativesRankingLoss (Default)

```python
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

model = SentenceTransformer("BAAI/bge-base-en-v1.5")

# Training data: (query, positive_passage) pairs
train_examples = [
    InputExample(texts=["How to reset password?", "Go to Settings > Security > Reset Password"]),
    InputExample(texts=["Return policy", "Items can be returned within 30 days of purchase"]),
    # ... more pairs
]

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=64)
train_loss = losses.MultipleNegativesRankingLoss(model)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    warmup_steps=100,
    output_path="./finetuned-model",
)
```

### Triplet Loss with Hard Negatives

```python
# (anchor, positive, negative) triplets
train_examples = [
    InputExample(texts=[
        "Python async tutorial",           # anchor
        "Guide to asyncio in Python 3",    # positive
        "Java concurrency with threads",   # hard negative (similar but wrong)
    ]),
]

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)
train_loss = losses.TripletLoss(model, distance_metric=losses.TripletDistanceMetric.COSINE, triplet_margin=0.2)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=5,
    output_path="./triplet-model",
)
```

## Hard Negative Mining

Hard negatives are the single biggest lever for embedding quality. Easy negatives (random documents) teach the model nothing after the first epoch.

### BM25 Hard Negatives

```python
# Use BM25 to find passages that lexically match but are semantically wrong
from rank_bm25 import BM25Okapi

corpus_tokenized = [doc.split() for doc in corpus]
bm25 = BM25Okapi(corpus_tokenized)

def mine_hard_negatives(query: str, positive_id: int, top_k: int = 10) -> list[str]:
    scores = bm25.get_scores(query.split())
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    negatives = []
    for idx, score in ranked:
        if idx != positive_id and len(negatives) < top_k:
            negatives.append(corpus[idx])
    return negatives
```

### Cross-Encoder Hard Negatives

```python
from sentence_transformers import CrossEncoder

cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def mine_with_cross_encoder(query: str, candidates: list[str], positive: str, n: int = 5) -> list[str]:
    """Find candidates that score high with cross-encoder but aren't the positive."""
    pairs = [(query, c) for c in candidates if c != positive]
    scores = cross_encoder.predict(pairs)
    ranked = sorted(zip(scores, [p[1] for p in pairs]), reverse=True)
    return [doc for _, doc in ranked[:n]]
```

### In-Batch Negatives
MultipleNegativesRankingLoss uses other positives in the batch as negatives. Larger batch sizes = more negatives = better training signal. Use batch size 64-256 when possible.

## Matryoshka Representation Learning

Train embeddings that work at multiple dimensions by slicing. Useful for trading accuracy vs speed/storage at inference time.

```python
from sentence_transformers import SentenceTransformer, losses

model = SentenceTransformer("BAAI/bge-base-en-v1.5")

# Matryoshka loss wraps another loss
base_loss = losses.MultipleNegativesRankingLoss(model)
matryoshka_loss = losses.MatryoshkaLoss(
    model,
    loss=base_loss,
    matryoshka_dims=[256, 128, 64],  # Train at these truncated dims
    matryoshka_weights=[1, 1, 1],    # Equal weight per dim
)

model.fit(
    train_objectives=[(train_dataloader, matryoshka_loss)],
    epochs=3,
    output_path="./matryoshka-model",
)

# At inference: truncate embeddings to desired dimension
embeddings = model.encode(texts)
embeddings_256d = embeddings[:, :256]  # Use first 256 dims
```

## MTEB Evaluation

MTEB (Massive Text Embedding Benchmark) is the standard for evaluating embeddings across tasks.

```python
from mteb import MTEB

model = SentenceTransformer("./finetuned-model")

# Evaluate on specific tasks
evaluation = MTEB(tasks=["STS17", "ArguAna", "NFCorpus"])
results = evaluation.run(model, output_folder="./mteb_results")

# Key task categories:
# - STS: Semantic Textual Similarity (correlation with human judgments)
# - Retrieval: Recall@K, NDCG@10 on search benchmarks
# - Classification: Linear probe accuracy
# - Clustering: V-measure on cluster assignments
# - Reranking: MAP on reranking benchmarks
```

**Evaluation tips**:
- Always compare against the base model (before fine-tuning)
- Report retrieval metrics (NDCG@10) and STS (Spearman correlation) at minimum
- Fine-tuning for retrieval can hurt STS and vice versa; check both
- Use domain-specific benchmarks if available (e.g., BEIR for retrieval)

## Bi-Encoder vs Cross-Encoder

| Aspect | Bi-Encoder | Cross-Encoder |
|--------|-----------|---------------|
| **Speed** | Fast (encode once, compare many) | Slow (re-encode each pair) |
| **Accuracy** | Good | Better (10-15% higher on STS) |
| **Use case** | Retrieval (first stage) | Reranking (second stage) |
| **Scaling** | O(n) encode + O(1) compare | O(n) per query |
| **Training data** | Positive pairs sufficient | Pairs with graded similarity |

### Two-Stage Pipeline

```python
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np

# Stage 1: Bi-encoder retrieval
bi_encoder = SentenceTransformer("BAAI/bge-base-en-v1.5")
corpus_embeddings = bi_encoder.encode(corpus, convert_to_numpy=True, normalize_embeddings=True)

query_embedding = bi_encoder.encode(query, convert_to_numpy=True, normalize_embeddings=True)
cosine_scores = query_embedding @ corpus_embeddings.T
top_k_indices = np.argsort(-cosine_scores)[:50]  # Top 50 candidates

# Stage 2: Cross-encoder reranking
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
pairs = [(query, corpus[i]) for i in top_k_indices]
rerank_scores = cross_encoder.predict(pairs)
reranked_indices = top_k_indices[np.argsort(-rerank_scores)][:5]  # Top 5 final

results = [corpus[i] for i in reranked_indices]
```

## Domain Adaptation Patterns

### When to Fine-Tune
- Base model retrieval < 70% NDCG@10 on your domain
- Domain-specific vocabulary (legal, medical, code)
- Custom similarity definition (e.g., "similar" means same author, not same topic)

### Minimal Fine-Tuning Recipe

```python
# 1. Collect 1K-10K (query, positive_doc) pairs from your domain
# 2. Mine hard negatives using BM25 or existing retrieval
# 3. Fine-tune with MultipleNegativesRankingLoss
# 4. Evaluate on held-out retrieval set

from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from torch.utils.data import DataLoader

model = SentenceTransformer("BAAI/bge-base-en-v1.5")

# Prepare data
train_examples = [InputExample(texts=[q, p]) for q, p in train_pairs]
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=64)

# Evaluator
dev_evaluator = evaluation.InformationRetrievalEvaluator(
    queries=dev_queries,        # dict[str, str]
    corpus=dev_corpus,          # dict[str, str]
    relevant_docs=dev_qrels,    # dict[str, set[str]]
    name="domain-eval",
)

train_loss = losses.MultipleNegativesRankingLoss(model)
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=dev_evaluator,
    epochs=3,
    evaluation_steps=500,
    warmup_steps=100,
    output_path="./domain-model",
    use_amp=True,  # Mixed precision for speed
)
```

### TSDAE Pre-Training (Unsupervised)

When you have domain text but no labeled pairs, use TSDAE (Transformer-based Sequential Denoising Auto-Encoder) for unsupervised domain adaptation before supervised fine-tuning.

```python
from sentence_transformers import losses

# Requires only unlabeled domain text
train_examples = [InputExample(texts=[doc, doc]) for doc in domain_docs]
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)

tsdae_loss = losses.DenoisingAutoEncoderLoss(
    model,
    decoder_name_or_path="BAAI/bge-base-en-v1.5",
    tie_encoder_decoder=True,
)

model.fit(
    train_objectives=[(train_dataloader, tsdae_loss)],
    epochs=1,
    output_path="./tsdae-pretrained",
)
# Then fine-tune with labeled pairs on top of this
```

## Gotchas

### Normalization Matters
Some models return unnormalized embeddings. Always normalize before cosine similarity. `sentence-transformers` `encode(..., normalize_embeddings=True)` handles this. For raw torch: `F.normalize(embeddings, p=2, dim=1)`.

### Query vs Document Prefixes
Models like E5 and BGE require prefixes: `"query: "` for queries, `"passage: "` for documents. Missing prefixes drop retrieval quality by 5-15%. Check model card.

### Batch Size and Memory
Large batch sizes help contrastive learning but eat GPU memory. Gradient accumulation or `GradCache` can simulate large batches with limited VRAM:
```python
# Effective batch size = batch_size * gradient_accumulation_steps
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    gradient_accumulation_steps=4,  # Simulates 4x batch size
)
```

### Embedding Dimension vs Quality
Higher dims don't always mean better. 768-dim BGE often outperforms 3072-dim models on specific domains after fine-tuning. Evaluate on your data, not just MTEB leaderboard.

### Catastrophic Forgetting
Fine-tuning too long on narrow domain data degrades general capability. Use low learning rate (2e-5), few epochs (1-3), and evaluate on general benchmarks alongside domain benchmarks.

### Index Rebuild After Fine-Tuning
After fine-tuning, you must re-encode your entire corpus and rebuild the vector index. Old embeddings are incompatible with the new model. Budget for this in production pipelines.
