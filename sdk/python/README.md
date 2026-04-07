# veda-client

Python client for the [Veda](https://github.com/kshitij2605/Veda) multimodal vector database.

## Installation

```bash
pip install veda-client
```

For development:

```bash
pip install veda-client[dev]
```

## Quick Start

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

## Multi-Vector Fusion Search

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

## Auto-Embed Insert

```python
client.embed_insert(
    "docs",
    records=[{"text": "hello world"}],
    auto_embed={"text_emb": {"source_field": "text", "model": "all-MiniLM-L6-v2"}},
)
```
