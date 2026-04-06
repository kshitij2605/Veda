# Data Viewer Platform — Inspiration Analysis

**Source:** `/home/mac/test/data-viewer`
**Purpose:** Document analysis and chunk quality evaluation platform (FastAPI + React/TypeScript)

## Tech Stack
- **Backend:** FastAPI, SQLite (WAL mode), SQLAlchemy, JWT auth, SSE streaming
- **Frontend:** React 19 + TypeScript, Vite, Zustand, Framer Motion, pdfjs-dist
- **External:** LLM evaluation (Qwen3.5-122B), BGE-M3 embeddings

## Key Patterns to Adopt for MDB Dashboard

### 1. Resizable Split Panes
- Flexible layouts for comparing multiple data views side-by-side
- Apply to: image + UMAP projection, original media + similar results, query + results

### 2. Real-Time SSE Streaming
- Per-item updates without full-page refreshes
- Apply to: embedding computation progress, UMAP layout iterations, search results streaming

### 3. Image Gallery with Modal Navigation
- Categorized grid with modal viewer supporting prev/next
- Apply to: browsing multimodal dataset, viewing search results across modalities

### 4. Color-Coded Score Badges
- Red/yellow/green for multi-dimensional quality metrics
- Apply to: similarity scores, embedding quality, cross-modal alignment scores

### 5. Multi-Strategy Evaluation
- 5 strategies evaluated in parallel with scored metrics
- Apply to: comparing different index types, quantization levels, embedding models

### 6. LLM-as-a-Judge Pattern
- Structured rubrics, 0-10 scales, streaming results
- Apply to: evaluating search result relevance, embedding quality assessment

### 7. User Feedback Collection
- Emoji-based + comments for human validation
- Apply to: search result relevance feedback, active learning loops

### 8. Diff Tracking
- Original vs modified comparison view
- Apply to: comparing old vs new embeddings, re-indexed data

### 9. Design System
- Warm industrial aesthetic, DM Sans + JetBrains Mono typography
- Dark/light mode via CSS variables
- Framer Motion animations (120-400ms)

### 10. Architecture Pattern
- FastAPI serves both API + built frontend from dist/
- SQLite auto-created on first run
- Zustand stores for auth, documents, chunks with pagination/search
