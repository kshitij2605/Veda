# Python SDK

`veda-client` is the official Python client for Veda.

## Installation

```bash
pip install veda-client
```

For development:

```bash
pip install veda-client[dev]
```

## Quick start

```python
from veda_client import VedaClient, CollectionSchema, FieldDefinition

client = VedaClient("http://localhost:6340")

# Create a collection
schema = CollectionSchema(
    name="images",
    fields=[
        FieldDefinition(name="title", type="string"),
        FieldDefinition(name="embedding", type="vector", dimension=768, metric="cosine"),
    ],
)
client.create_collection(schema)

# Insert records
client.insert("images", [
    {"title": "sunset", "embedding": [0.1] * 768},
    {"title": "mountain", "embedding": [0.2] * 768},
])

# Search
results = client.search("images", field="embedding", vector=[0.15] * 768, limit=5)
for hit in results.results:
    print(f"{hit.id}: {hit.distance}")
```

## Authentication

```python
client = VedaClient("http://localhost:6340")
client.login("admin", "password")

# Or pass token directly
client = VedaClient("http://localhost:6340", token="your-jwt-token")
```

## Filtered search

```python
response = client.search(
    collection="products",
    field="embedding",
    vector=[0.15, 0.22, ...],
    limit=10,
    filter="category = 'electronics'",
)

for hit in response.results:
    print(f"{hit.id}: {hit.distance:.4f}")
```

## Multi-vector fusion search

```python
results = client.search_multi(
    "multimodal",
    queries=[
        {"field": "text_emb", "values": [0.1] * 384, "weight": 0.7},
        {"field": "image_emb", "values": [0.2] * 512, "weight": 0.3},
    ],
    fusion="rrf",
    limit=10,
)
```

## Auto-embed insert

```python
client.embed_insert(
    "docs",
    records=[{"text": "hello world"}],
    auto_embed={"text_emb": {"source_field": "text", "model": "all-MiniLM-L6-v2"}},
)
```

## Document processing

```python
# Process a single document (OCR -> chunk -> embed -> insert)
with open("report.pdf", "rb") as f:
    result = client.process_document("my-docs", f)
    print(f"Inserted {result['records_inserted']} records")
    for chunk in result['chunks']:
        print(f"  [{chunk['type']}] {chunk['id']}: {chunk['preview'][:60]}")
```

See the [Document pipeline](/docs/document-pipeline) for chunking strategies and OCR setup.
