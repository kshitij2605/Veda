# Chunking Strategies for RAG Pipelines (2024-2026 Research)

*Research compiled: April 2026*

---

## 1. Common Chunking Strategies

| Strategy | How It Works | Pros | Cons |
|---|---|---|---|
| **Fixed-size** | Split every N tokens/characters | Simple, fast, predictable | Breaks mid-sentence, ignores structure |
| **Recursive character** | Split by hierarchy of separators (`\n\n` > `\n` > `. ` > ` `) | Respects natural boundaries, fast | Still rule-based, no semantic awareness |
| **Sentence-based** | Split on sentence boundaries | Clean semantic units | Individual sentences often lack context |
| **Paragraph-based** | Split on double newlines | Natural topic units | Paragraphs vary wildly in length |
| **Markdown/HTML-aware** | Split on headings, preserve structure | Respects document hierarchy | Only works for structured input |
| **Semantic** | Embed sentences, split where cosine similarity drops | Topic-coherent chunks | Expensive, variable chunk sizes |
| **Document-structure-aware** | Parse layout (tables, figures, sections) then chunk | Preserves complex elements | Requires layout detection pipeline |

## 2. Optimal Chunk Sizes and Overlap

### Benchmark-Validated Defaults

- **512 tokens with 10-20% overlap** is the consensus starting point across NVIDIA, Vectara, FloTorch, and PremAI benchmarks.
- Recursive character splitting at 512 tokens scored **69% accuracy** in the largest real-document benchmark (PremAI 2026).
- NVIDIA (2024): **15% overlap** outperformed 10% and 20% on FinanceBench with 1024-token chunks.

### Query-Type Sensitivity

| Query Type | Optimal Chunk Size |
|---|---|
| Factoid / lookup | 256-512 tokens |
| Analytical / reasoning | 1024+ tokens |
| Narrative / summarization | 512-1024 tokens |

### Key Papers

- **Vectara (NAACL 2025 Findings)**: Fixed 200-word chunks match or beat semantic chunking across retrieval and answer generation tasks. Computational overhead of semantic methods was not justified.
- **"The Chunking Paradigm" (ACL 2025)**: Recursive semantic approach combining structure-aware splitting with embedding-based merging.
- **PremAI (2026)**: Identified a "context cliff" around 2,500 tokens where response quality drops sharply.
- **Clinical Decision Support (MDPI Bioengineering, Nov 2025)**: Adaptive chunking with cosine similarity > 0.8 threshold and ~500-word caps performed well for medical RAG.

### Overlap Findings

- Skipping overlap entirely is a common mistake -- even 10% recovers boundary context.
- 10-20% is the sweet spot. Beyond 20%, index bloat outweighs retrieval gains.

## 3. Multimodal Chunking (Text + Images + Tables)

### The Core Problem

Traditional text-only chunking ignores visual elements (figures, charts, layout) which often contain critical information. Tables spanning multiple pages get broken. Images lose their surrounding context.

### Approaches

**A. Vision-Guided Chunking (Tripathi et al., 2025)**
- Uses LMMs (e.g., Gemini-2.5-Pro) to process PDF page batches while maintaining semantic coherence
- Cross-batch context preservation handles multi-page tables and figures
- Superior preservation of document structure vs. text-only methods
- Cost: requires LMM API calls per page batch

**B. Multi-Vector Retriever (LangChain)**
Three options for handling mixed modality:
1. Multimodal embeddings (CLIP, voyage-multimodal-3) for joint text+image embedding
2. LLM-generated text summaries of images, then embed summaries
3. Embed summaries for retrieval, pass raw images/tables to multimodal LLM for synthesis (best quality)

**C. ColPali -- Skip Chunking Entirely**
- Index entire pages as images using vision-language models
- Eliminates OCR, layout parsing, and chunking pipelines entirely
- Reduces failure points but requires vision-capable retrieval models

**D. Best Practice for OCR-to-Markdown Pipeline**
- Keep images anchored to their surrounding text paragraph (within the same chunk)
- Extract tables as images; generate text summaries for embedding; store raw table for LLM synthesis
- Use document structure (headings) as primary split boundaries
- Preserve heading hierarchy in chunk metadata

## 4. Semantic Chunking Deep Dive

### How It Works

1. Segment text into sentences
2. Embed each sentence
3. Compute cosine similarity between consecutive sentence embeddings
4. Where similarity drops below threshold, insert a chunk boundary
5. Result: variable-size chunks (some 200 tokens, others 1500) each semantically coherent

### Performance Reality

| Source | Semantic Recall | Recursive Recall | End-to-End Winner |
|---|---|---|---|
| Chroma eval | 91.9% | 85-89% | Semantic (recall only) |
| FloTorch eval | -- | -- | Recursive (+15 pts accuracy) |
| Vectara NAACL 2025 | -- | -- | Fixed-size (all tasks) |
| Superlinked HotpotQA | -- | -- | SentenceSplitter (with ColBERT v2) |

**Key insight**: Semantic chunking improves recall in isolation but often produces fragments too small (avg 43 tokens) for the LLM to reason over. The retrieval gain is negated by insufficient context at generation time.

### Performance Cost

- Requires embedding every sentence at indexing time (N embedding calls for N sentences)
- 10x+ processing time vs recursive splitting
- Embedding model choice matters as much as chunking strategy

### Advanced Variants

- **Max-Min Semantic Chunking** (Milvus): Embed all sentences upfront, use min-max similarity thresholds to find optimal boundaries
- **Growing Window Semantic Chunking** (ScienceDirect 2025): Dynamically increases window size, comparing cumulative chunk embedding against candidate next sentences

## 5. Production Systems

### LangChain Text Splitters
- **RecursiveCharacterTextSplitter**: Recommended default. Splits by hierarchy of separators.
- **MarkdownHeaderTextSplitter**: Splits on markdown headings, preserves heading metadata.
- **SemanticChunker**: Splits where sentence similarity drops below threshold.
- **TokenTextSplitter**: Encodes to tokens, splits, decodes back.
- **UnstructuredLoader**: Integration with Unstructured.io for complex document types.

### LlamaIndex Node Parsers
- **SentenceSplitter**: Like RecursiveCharacterTextSplitter but token-based sizing.
- **SentenceWindowNodeParser**: Splits into individual sentences, stores surrounding window in metadata (useful for reranking).
- **SemanticSplitterNodeParser**: Embedding-based adaptive breakpoints.
- **HierarchicalNodeParser**: Multi-level chunk hierarchy (e.g., 2048 > 512 > 128 tokens) with parent references.
- **UnstructuredElementNodeParser**: Separates text nodes from embedded objects (tables, images).

### Unstructured.io
- Extracts structured elements (text blocks, tables, images, headers) from 25+ file types
- Elements can be chunked independently with type-aware strategies
- Tables extracted as both raw HTML and text summaries

## 6. Best Practices Summary

### Chunk Sizes by Embedding Model

| Model | Max Tokens | Recommended Chunk Size |
|---|---|---|
| OpenAI text-embedding-ada-002 | 8,191 | 256-512 tokens |
| OpenAI text-embedding-3-small/large | 8,191 | 256-512 tokens |
| Cohere Embed v4 | ~128k | 256-512 tokens (for RAG) |
| Voyage AI models | 32,000 | 256-512 tokens |
| Jina embeddings v3 | 8,192 | 256-512 tokens |

**Key finding**: Chunk size matters more than embedding model choice. Even models with 128k context windows benefit from 256-512 token chunks for retrieval.

### Overlap
- 10-20% of chunk size (50-100 tokens for 512-token chunks)
- 15% performed best in NVIDIA's FinanceBench evaluation

### Metadata Preservation
- Always store: source document ID, chunk index, heading hierarchy, page number
- For multimodal: store image references, table boundaries, content type
- Include parent document title and section path for context

## 7. Late Chunking and Contextual Chunking

### Anthropic's Contextual Retrieval (Sept 2024)

- Each chunk is sent to an LLM along with the full document
- LLM prepends a short context summary to each chunk (e.g., "This chunk is from the Q3 earnings report, specifically the revenue breakdown section")
- Reduces retrieval failures by up to **67%** when combined with BM25 hybrid search
- **Cost**: Requires one LLM call per chunk at indexing time; expensive at scale

### Jina's Late Chunking (Sept 2024)

- Encode the entire document through a long-context embedding model first
- Then split the token embedding sequence into chunks and mean-pool each
- Each chunk embedding inherits contextual information from the full document
- **Cost**: One embedding call per document (much cheaper than contextual retrieval)
- Requires long-context embedding models with mean pooling

### Head-to-Head

| Dimension | Contextual Retrieval | Late Chunking |
|---|---|---|
| Quality | Higher semantic coherence | Slightly lower |
| Cost | High (LLM per chunk) | Low (one embed per doc) |
| Complexity | Moderate | Low |
| Model requirement | Any LLM + any embedder | Long-context embedding model |

### Voyage Context-3 (July 2025)

Voyage AI's voyage-context-3 claims to outperform both:
- Beats Jina-v3 late chunking by **23.66%** on chunk-level retrieval
- Beats contextual retrieval by **6.76%** on chunk-level retrieval
- Achieves this natively in the embedding model without LLM calls

---

## Recommendations for Veda (OCR-to-Markdown Vector Database)

Given that Veda processes documents through OCR producing markdown with embedded images, here are specific recommendations in priority order:

### Chunking Strategy

**Use markdown-aware recursive splitting as the primary strategy.** Split on heading boundaries first (`# > ## > ###`), then paragraph breaks, then sentences. This naturally aligns with OCR-extracted document structure. Target **512 tokens with 15% overlap** (77 tokens). This outperforms semantic chunking in end-to-end benchmarks while being orders of magnitude cheaper.

### Multimodal Element Handling

Keep images within the chunk of their surrounding text -- never split an image from its caption or referring paragraph. Extract tables separately: store raw table content for LLM synthesis but generate a text summary for embedding and retrieval. Tag each chunk with content type metadata (text-only, text+image, table).

### Metadata

Every chunk must carry: document ID, chunk index, heading path (e.g., `["Chapter 3", "Revenue", "Q3 Results"]`), page number(s), content types present, and source coordinates. This heading hierarchy is critical for contextual retrieval and reranking.

### Advanced Features (Phase 2)

Implement **late chunking** support for users with long-context embedding models -- it is cheap, effective, and model-agnostic. Offer optional **contextual retrieval** as a premium indexing mode for users who want maximum recall and can afford LLM calls. Support **hierarchical chunking** (parent-child relationships at 2048/512/128 tokens) to enable recursive retrieval for complex queries.

### What NOT to Do

Do not default to semantic chunking -- it is expensive, produces inconsistent chunk sizes, and does not reliably improve end-to-end accuracy. Do not exceed 2,500 tokens per chunk (the "context cliff"). Do not strip metadata during chunking; it is as valuable as the content itself.

---

## Sources

- [NVIDIA: Finding the Best Chunking Strategy](https://developer.nvidia.com/blog/finding-the-best-chunking-strategy-for-accurate-ai-responses/)
- [PremAI: RAG Chunking Strategies 2026 Benchmark](https://blog.premai.io/rag-chunking-strategies-the-2026-benchmark-guide/)
- [Vectara NAACL 2025: Fixed vs Semantic Chunking](https://aclanthology.org/2025.icnlsp-1.15.pdf)
- [Vision-Guided Chunking (arXiv 2506.16035)](https://arxiv.org/abs/2506.16035)
- [Anthropic: Contextual Retrieval](https://www.datacamp.com/tutorial/contextual-retrieval-anthropic)
- [Jina: Late Chunking (arXiv 2409.04701)](https://arxiv.org/pdf/2409.04701)
- [Voyage AI: voyage-context-3](https://blog.voyageai.com/2025/07/23/voyage-context-3/)
- [LangChain: Semi-structured Multi-modal RAG](https://blog.langchain.com/semi-structured-multi-modal-rag/)
- [LlamaIndex: Node Parser Modules](https://developers.llamaindex.ai/python/framework/module_guides/loading/node_parsers/modules/)
- [Weaviate: Chunking Strategies for RAG](https://weaviate.io/blog/chunking-strategies-for-rag)
- [Firecrawl: Best Chunking Strategies for RAG 2025](https://www.firecrawl.dev/blog/best-chunking-strategies-rag)
- [Stack Overflow: Breaking Up Is Hard to Do](https://stackoverflow.blog/2024/12/27/breaking-up-is-hard-to-do-chunking-in-rag-applications/)
- [ColPali: Vision-Language Intelligence for RAG](https://towardsdatascience.com/bringing-vision-language-intelligence-to-rag-with-colpali/)
- [Milvus: Max-Min Semantic Chunking](https://milvus.io/blog/embedding-first-chunking-second-smarter-rag-retrieval-with-max-min-semantic-chunking.md)
- [ScienceDirect: Growing Window Semantic Chunking](https://www.sciencedirect.com/science/article/pii/S0950705125019343)
- [Pinecone: Choosing an Embedding Model](https://www.pinecone.io/learn/series/rag/embedding-models-rundown/)
- [Clinical Decision Support Chunking Study (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12649634/)
- [Late Chunking vs Contextual Retrieval](https://medium.com/kx-systems/late-chunking-vs-contextual-retrieval-the-math-behind-rags-context-problem-d5a26b9bbd38)
