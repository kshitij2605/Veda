# Distance Metrics Research for Veda

> Research compiled April 2026. Covers standard metrics, latest developments, competitor analysis, and recommendations.

## Key Finding

**No revolutionary new distance metrics have emerged in 2024-2026.** The core set (cosine, L2, dot product) remains universal. Innovation is happening in how metrics are *approximated* (RaBitQ, PQ) and how *indexes adapt* (Quake), not in the metrics themselves.

**Critical insight:** The metric must match the training objective of the embedding model. Every major embedding model in 2025-2026 (Gemini Embedding 2, Amazon Nova, CLIP, jina-v4) produces normalized vectors and recommends cosine similarity.

## Competitor Support Matrix

| Metric | Qdrant | Milvus | Weaviate | Pinecone | pgvector | ChromaDB |
|---|---|---|---|---|---|---|
| Cosine | Yes | Yes | Yes | Yes | Yes | Yes |
| Euclidean (L2) | Yes | Yes | Yes | Yes | Yes | Yes |
| Dot Product | Yes | Yes | Yes | Yes | Yes | Yes |
| Manhattan (L1) | Yes | No | Yes | No | Yes | No |
| Hamming | No | Yes | Yes | No | Yes | No |
| Jaccard | No | Yes | No | No | Yes | No |
| BM25 | No | Yes | No | No | No | Yes |

## Recommendation for Veda

### Tier 1 — MUST HAVE (Launch)
1. **Cosine Similarity** — universal default for all embedding models
2. **Euclidean (L2)** — standard for raw/unnormalized embeddings
3. **Inner Product (Dot Product)** — required for MIPS, recommendation systems

### Tier 2 — SHOULD HAVE (V1.1)
4. **Hamming Distance** — binary vectors, extremely fast (XOR + popcount)
5. **Manhattan (L1)** — more robust to outliers, better high-dim discrimination
6. **Jaccard Similarity** — set/sparse data

### Tier 3 — NICE TO HAVE (Future)
7. **BM25 scoring** — hybrid sparse/full-text search
8. **MinHash Jaccard** — approximate Jaccard for large sets
9. **MaxSim scoring** — late-interaction / multi-vector retrieval (ColBERT-style)

### DO NOT SUPPORT
- Angular Distance (identical rankings to cosine)
- Mahalanobis (too expensive, no precedent)
- Wasserstein/EMD (wrong abstraction)
- Bregman/KL divergences (research-stage)
- Learned/Neural metrics (breaks ANN index structures)

## Architectural Notes
- Metric-per-collection (different fields can use different metrics)
- Transparent quantization (users specify cosine/L2/dot, engine applies RaBitQ/PQ internally)
- Normalized vector detection (auto-optimize to dot product)
- SIMD optimization for all Tier 1+2 metrics (AVX-512, NEON)
