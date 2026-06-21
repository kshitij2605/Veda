# Document pipeline

Upload a document and Veda handles the entire pipeline — OCR, chunking, embedding, and insertion into your collection. Document parsing is powered by FlashParse, which converts PDFs, Office documents (DOCX, PPTX, XLSX), images, and scans into structured markdown — running layout analysis to extract tables, figures, and images, not just flat text.

> **Pipeline:** upload → FlashParse OCR → markdown + images → chunk (configurable strategy) → embed (optional) → insert into collection

## Setup

1. Copy `.env.example` to `.env` and configure.
2. Start Veda with OCR enabled:

```bash
# Text-only OCR (FlashParse + GLM-OCR, ~8GB VRAM)
docker compose --profile ocr up

# Full OCR with VLM image captioning (~48GB+ VRAM)
docker compose --profile ocr --profile ocr-vlm up
```

The `ocr` profile starts FlashParse and GLM-OCR for text, table, and formula extraction. Add `ocr-vlm` to also start the VLM service for image captioning and classification.

To use an external VLM instead, point these at any OpenAI-compatible API:

| Variable | Example |
|---|---|
| `VLM_ENDPOINT` | `http://your-server:8008/v1/chat/completions` |
| `VLM_API_KEY` | your API key (leave empty for local models) |
| `VLM_MODEL` | the model name to use |

## Supported formats

PDF, Office documents (DOCX, PPTX, XLSX), images (PNG, JPG, TIFF, BMP), and scans. GLM-OCR plus PP-DocLayoutV3 layout analysis recover tables (as clean markdown tables), figures, and extracted images — so a spreadsheet, a slide deck, or a scanned report all become searchable in one call.

## Chunking strategies

| Strategy | Description | Default size | Default overlap |
|---|---|---|---|
| `markdown_recursive` | Recursively splits markdown by structure (default) | 2000 chars | 300 |
| `heading_1` | Splits on H1 headings | — | — |
| `heading_2` | Splits on H2 headings | — | — |
| `heading_3` | Splits on H3 headings | — | — |
| `semantic` | Word-overlap Jaccard similarity; splits on topic change | 2000 chars | 300 |
| `sentence` | Splits on sentence boundaries (CJK punctuation supported) | 2000 chars | 300 |
| `fixed_size` | Fixed character count chunks | 2000 chars | 300 |
| `page` | Splits on page separators | — | — |
| `none` | No chunking; entire document as one chunk | — | — |

## Process a document

```bash
# 1. Create a collection for documents
curl -X POST http://localhost:6340/v1/collections \
  -H "Content-Type: application/json" \
  -d '{
  "name": "my-docs",
  "fields": [
    {"name": "content", "type": "string", "filterable": true},
    {"name": "content_embedding", "type": "vector", "dimension": 768, "metric": "cosine"},
    {"name": "source_file", "type": "string", "filterable": true},
    {"name": "chunk_index", "type": "int32"},
    {"name": "image_blob", "type": "blob"}
  ]
}'

# 2. Process a document (end-to-end: OCR -> chunk -> embed -> insert)
# Default (markdown-recursive chunking)
curl -X POST http://localhost:6340/v1/collections/my-docs/process-document \
  -F "file=@report.pdf"

# Semantic chunking
curl -X POST http://localhost:6340/v1/collections/my-docs/process-document \
  -F "file=@report.pdf" \
  -F "chunking_strategy=semantic"

# Split by H2 headings with custom chunk size
curl -X POST http://localhost:6340/v1/collections/my-docs/process-document \
  -F "file=@report.pdf" \
  -F "chunking_strategy=heading_2" \
  -F "chunk_size=3000"

# 3. Search the processed content
curl -X POST http://localhost:6340/v1/collections/my-docs/search -d '{
  "query": {"vector": {"field": "content_embedding", "values": [...]}},
  "limit": 5
}'
```

## Options

| Option | Description |
|---|---|
| `chunking_strategy=<strategy>` | Chunking strategy (default: `markdown_recursive`) |
| `chunk_size=<chars>` | Maximum chunk size in characters (default: 2000) |
| `chunk_overlap=<chars>` | Overlap between chunks in characters (default: 300) |
| `skip_captions=true` | Skip VLM-based image captioning (faster processing) |
| `dpi=300` | DPI for PDF rasterization (default: automatic) |

```bash
curl -X POST http://localhost:6340/v1/collections/my-docs/process-document \
  -F "file=@report.pdf" \
  -F "skip_captions=true" \
  -F "dpi=300"
```

## Batch processing

Process multiple documents in a single request; returns aggregate stats with per-file results:

```bash
curl -X POST http://localhost:6340/v1/collections/my-docs/process-documents \
  -F "files=@report1.pdf" \
  -F "files=@report2.pdf" \
  -F "files=@slides.pptx"
```

## From the Python SDK

```python
from veda_client import VedaClient

client = VedaClient("http://localhost:6340")

# Process a single document
with open("report.pdf", "rb") as f:
    result = client.process_document("my-docs", f)
    print(f"Inserted {result['records_inserted']} records")
    for chunk in result['chunks']:
        print(f"  [{chunk['type']}] {chunk['id']}: {chunk['preview'][:60]}")
```

> If FlashParse is not running, the endpoint returns a clear error with setup instructions.

## Reindexing with a different strategy

Already-ingested collections can be re-chunked (and optionally re-embedded) without downtime: `POST /v1/collections/{name}/reindex` accepts `{chunker, re_embed, ...}`, rebuilds into a sibling collection, and atomically swaps the alias at commit time. It returns `424 FAILED_DEPENDENCY` if `re_embed` is requested with no embedding endpoint configured.
