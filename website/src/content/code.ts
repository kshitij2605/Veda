export interface CodeTab {
  label: string
  code: string
}

export const CODE_TABS: CodeTab[] = [
  {
    label: 'Create collection',
    code: `curl -X POST localhost:6340/v1/collections \\
  -H 'content-type: application/json' \\
  -d '{
    "name": "products",
    "fields": [
      { "name": "text_embedding",  "type": "vector", "dimension": 1024,
        "metric": "cosine", "model": "BAAI/bge-m3" },
      { "name": "image_embedding", "type": "vector", "dimension": 768 },
      { "name": "title", "type": "string", "bm25": true },
      { "name": "price", "type": "float32", "filterable": true }
    ]
  }'`,
  },
  {
    label: 'Fusion search',
    code: `POST /v1/collections/products/search
{
  "query": {
    "multi_vector": [
      { "field": "text_embedding",  "values": [/* … */], "weight": 0.7 },
      { "field": "image_embedding", "values": [/* … */], "weight": 0.3 }
    ],
    "fusion": "rrf",
    "fusion_params": { "k": 60 }
  },
  "filter": "category = 'electronics' AND price < 500",
  "limit": 10
}`,
  },
  {
    label: 'Process a document',
    code: `# OCR -> chunk -> embed -> insert, in one call
curl -X POST \\
  localhost:6340/v1/collections/docs/process-document \\
  -F 'file=@whitepaper.pdf' \\
  -F 'chunker=markdown_recursive'

# -> searchable vectors. No external OCR or embedding tooling.`,
  },
]
