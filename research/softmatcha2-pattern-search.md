# SoftMatcha 2 — Pattern Search for Veda

**Source:** https://softmatcha.github.io/v2/
**Relevance:** V1 feature — suffix array-based fuzzy pattern matching

## What It Is

SoftMatcha 2 is an ultra-fast pattern matching system for searching massive text corpora. It searches over trillion-scale natural language corpora in under 0.3 seconds while handling semantic variations (substitution, insertion, deletion) and preserving word order.

## How It Works

- **Suffix array-based string matching** — classical data structure that scales well with corpus size
- **Fast exact lookup** via disk-aware design optimization
- **Dynamic corpus-aware pruning** to eliminate unnecessary search branches, preventing exponential blowup
- O(m log n) exact lookup (m = pattern length, n = corpus size)

## Why It Matters for Veda

**No OSS vector database offers suffix array-based pattern search.** This gives Veda a unique differentiator:

| Search Type | Technology | When to Use |
|------------|-----------|-------------|
| Semantic search | ANN (HNSW/DiskANN) | "Find documents about internet platform liability" |
| Keyword relevance | BM25 sparse vectors | "Rank documents mentioning these terms" |
| Pattern search | Suffix arrays (SoftMatcha) | "Find exact/near-exact occurrences of 'section 230 of the communications decency act'" |

Users get all three in one system. Competitors require separate tools (Elasticsearch for text, Qdrant for vectors).

## Integration in Veda

- New index type: `suffix_array` on String fields
- New query type: `pattern` with fuzzy matching and word-order preservation
- Built at segment seal time alongside ANN indexes
- Disk-aware design for billion-token scale
- Exposed in Niriksha's Search Playground as a dedicated Pattern Search tab

## Performance Targets

- Sub-second search over trillion-token corpora
- Significantly lower latency than infini-gram
- Handles substitution, insertion, deletion variations
