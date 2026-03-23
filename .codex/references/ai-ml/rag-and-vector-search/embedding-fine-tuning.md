# Embedding Fine-Tuning Reference

Code examples for contrastive learning, hard negative mining, Matryoshka training, MTEB evaluation, and domain adaptation.

## MultipleNegativesRankingLoss (Default)

```python
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

model = SentenceTransformer("BAAI/bge-base-en-v1.5")

# Training data: (query, positive_passage) pairs
train_examples = [
    InputExample(texts=["How to reset password?", "Go to Settings > Security > Reset Password"]),
    InputExample(texts=["Return policy", "Items can be returned within 30 days of purchase"]),
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

## Triplet Loss with Hard Negatives

```python
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

## BM25 Hard Negative Mining

```python
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

## Cross-Encoder Hard Negative Mining

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

## Matryoshka Training

```python
from sentence_transformers import SentenceTransformer, losses

model = SentenceTransformer("BAAI/bge-base-en-v1.5")

base_loss = losses.MultipleNegativesRankingLoss(model)
matryoshka_loss = losses.MatryoshkaLoss(
    model,
    loss=base_loss,
    matryoshka_dims=[256, 128, 64],
    matryoshka_weights=[1, 1, 1],
)

model.fit(
    train_objectives=[(train_dataloader, matryoshka_loss)],
    epochs=3,
    output_path="./matryoshka-model",
)

# At inference: truncate embeddings to desired dimension
embeddings = model.encode(texts)
embeddings_256d = embeddings[:, :256]
```

## MTEB Evaluation

```python
from mteb import MTEB

model = SentenceTransformer("./finetuned-model")

evaluation = MTEB(tasks=["STS17", "ArguAna", "NFCorpus"])
results = evaluation.run(model, output_folder="./mteb_results")

# Key task categories:
# - STS: Semantic Textual Similarity (correlation with human judgments)
# - Retrieval: Recall@K, NDCG@10 on search benchmarks
# - Classification: Linear probe accuracy
# - Clustering: V-measure on cluster assignments
# - Reranking: MAP on reranking benchmarks
```

## Domain Adaptation Recipe

```python
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from torch.utils.data import DataLoader

model = SentenceTransformer("BAAI/bge-base-en-v1.5")

# 1. Collect 1K-10K (query, positive_doc) pairs from your domain
# 2. Mine hard negatives using BM25 or existing retrieval
train_examples = [InputExample(texts=[q, p]) for q, p in train_pairs]
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=64)

# Evaluator
dev_evaluator = evaluation.InformationRetrievalEvaluator(
    queries=dev_queries,
    corpus=dev_corpus,
    relevant_docs=dev_qrels,
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
    use_amp=True,
)
```

## TSDAE Unsupervised Pre-Training

When you have domain text but no labeled pairs:

```python
from sentence_transformers import losses

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

## Two-Stage Retrieval Pipeline

```python
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np

# Stage 1: Bi-encoder retrieval
bi_encoder = SentenceTransformer("BAAI/bge-base-en-v1.5")
corpus_embeddings = bi_encoder.encode(corpus, convert_to_numpy=True, normalize_embeddings=True)

query_embedding = bi_encoder.encode(query, convert_to_numpy=True, normalize_embeddings=True)
cosine_scores = query_embedding @ corpus_embeddings.T
top_k_indices = np.argsort(-cosine_scores)[:50]

# Stage 2: Cross-encoder reranking
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
pairs = [(query, corpus[i]) for i in top_k_indices]
rerank_scores = cross_encoder.predict(pairs)
reranked_indices = top_k_indices[np.argsort(-rerank_scores)][:5]

results = [corpus[i] for i in reranked_indices]
```
