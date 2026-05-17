use serde::{Deserialize, Serialize};

/// Chunking strategy configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChunkingStrategy {
    /// Fixed-size character splitting with overlap.
    FixedSize {
        chunk_size: usize,
        chunk_overlap: usize,
    },
    /// Recursive splitting on markdown structure: headings > paragraphs > sentences > chars.
    /// This is the DEFAULT and RECOMMENDED strategy for OCR/document output.
    MarkdownRecursive {
        max_chunk_size: usize,
        chunk_overlap: usize,
    },
    /// Split by sentences, combining up to max_chunk_size.
    Sentence {
        max_chunk_size: usize,
        chunk_overlap: usize,
    },
    /// Split by page separators (`\n---\n`) only.
    Page,
    /// Split by markdown headings at a specific depth level.
    /// Level 1 = split on # only, Level 2 = split on # and ##, Level 3 = split on #, ##, ###.
    MarkdownHeading {
        level: u8,             // 1, 2, or 3
        max_chunk_size: usize, // if a section exceeds this, sub-split by paragraphs
        chunk_overlap: usize,
    },
    /// Semantic chunking: split at points where embedding similarity drops.
    /// Computes a similarity score between consecutive segments and splits
    /// where similarity falls below a threshold.
    /// NOTE: Uses a lightweight text-based similarity heuristic (word overlap)
    /// since we can't call embedding models synchronously in the chunker.
    /// For true embedding-based semantic chunking, use the embed-insert pipeline.
    Semantic {
        max_chunk_size: usize,     // hard max per chunk (default 2000)
        similarity_threshold: f32, // 0.0-1.0, split when similarity drops below (default 0.5)
        window_size: usize,        // sentences per comparison window (default 3)
    },
    /// Hierarchical chunking: emits BOTH coarse (doc-section) chunks and
    /// fine (paragraph/sentence) chunks for the same document.
    ///
    /// Each fine chunk is tagged with `content_types = ["fine"]` and the
    /// corresponding coarse chunk's text is its parent. Coarse chunks are
    /// tagged `content_types = ["coarse"]`. Useful for retrieval pipelines
    /// that retrieve coarse-grained candidates first then refine.
    Hierarchical {
        coarse_size: usize, // characters per coarse chunk
        fine_size: usize,   // characters per fine chunk
        fine_overlap: usize,
    },
    /// Late chunking (Jina, Sept 2024) / voyage-context-3 (July 2025): the
    /// chunk boundaries are produced here, but actual *contextual*
    /// embedding happens in the embedding service:
    /// 1. Embed the FULL document through a long-context model.
    /// 2. For each chunk's `(start_offset, end_offset)`, mean-pool the
    ///    token embeddings within that window.
    /// 3. Each chunk vector inherits global document context.
    ///
    /// At the chunking layer we still emit MarkdownRecursive-style chunks
    /// (so downstream metadata works), and tag each chunk with
    /// `content_types = ["late_chunked"]` so the embedding service knows
    /// to apply the late-pooling path.
    ///
    /// The user is responsible for pointing `EMBEDDING_ENDPOINT` at a
    /// model that supports this (voyage-context-3, jina-embeddings-v3,
    /// or a custom long-context model + mean-pool helper).
    LateChunking {
        max_chunk_size: usize,
        chunk_overlap: usize,
    },
    /// Anthropic-style contextual retrieval (Sept 2024). For each chunk,
    /// call an LLM with the full document + chunk; the LLM emits a 1-2
    /// sentence context summary which is **prepended** to the chunk text
    /// before embedding. Reduces retrieval failures by ~49% solo, ~67%
    /// paired with reranking (Anthropic benchmark).
    ///
    /// Requires `CONTEXTUAL_LLM_ENDPOINT` env var pointing at an
    /// OpenAI-compatible chat-completions endpoint. Slow at ingest time
    /// (one LLM call per chunk); prompt caching is essential at scale.
    ContextualChunking {
        max_chunk_size: usize,
        chunk_overlap: usize,
    },
    /// No chunking — entire document is one chunk.
    None,
}

/// Time-window plan for audio / video chunking.
///
/// The chunker at this layer does not decode media; it produces a sequence of
/// `(start_ms, end_ms)` ranges, with optional overlap. The embed-insert path
/// is responsible for actually slicing the media bytes (via ffmpeg or the
/// embedding service's own segmentation) and embedding each slice.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TimeWindowPlan {
    pub window_ms: u64,
    pub overlap_ms: u64,
}

impl Default for TimeWindowPlan {
    /// 15-second overlapping windows with 3-second overlap — the de-facto
    /// default in recent multimodal RAG research for audio and video.
    fn default() -> Self {
        Self {
            window_ms: 15_000,
            overlap_ms: 3_000,
        }
    }
}

impl TimeWindowPlan {
    /// Emit `(start_ms, end_ms)` pairs covering `total_ms` with this plan.
    /// The final window may be shorter than `window_ms` to cover the tail.
    pub fn windows(&self, total_ms: u64) -> Vec<(u64, u64)> {
        if total_ms == 0 || self.window_ms == 0 {
            return Vec::new();
        }
        let step = self.window_ms.saturating_sub(self.overlap_ms).max(1);
        let mut out = Vec::new();
        let mut start = 0u64;
        while start < total_ms {
            let end = (start + self.window_ms).min(total_ms);
            out.push((start, end));
            if end == total_ms {
                break;
            }
            start += step;
        }
        out
    }

    /// Emit windows aligned to externally-detected scene boundaries.
    ///
    /// `scene_cuts_ms` is a sorted list of cut timestamps (boundaries
    /// between scenes). Scenes shorter than `window_ms / 2` are merged
    /// with the next scene; scenes longer than `window_ms * 3 / 2` are
    /// subdivided with a regular [`Self::windows`] plan inside the scene.
    ///
    /// Veda does **not** detect scenes itself — feed this output from
    /// PySceneDetect / ffmpeg-scenedetect / a VLM-based detector.
    pub fn scene_aligned_windows(
        &self,
        scene_cuts_ms: &[u64],
        total_ms: u64,
    ) -> Vec<(u64, u64)> {
        if total_ms == 0 {
            return Vec::new();
        }
        // Build raw scene spans from sorted cuts.
        let mut boundaries: Vec<u64> = Vec::with_capacity(scene_cuts_ms.len() + 2);
        boundaries.push(0);
        for &c in scene_cuts_ms {
            if c > 0 && c < total_ms {
                boundaries.push(c);
            }
        }
        boundaries.push(total_ms);
        boundaries.sort_unstable();
        boundaries.dedup();

        let min_len = self.window_ms / 2;
        let max_len = self.window_ms * 3 / 2;

        // Merge very-short scenes into the next.
        let mut merged: Vec<(u64, u64)> = Vec::new();
        let mut i = 0;
        while i + 1 < boundaries.len() {
            let start = boundaries[i];
            let mut end = boundaries[i + 1];
            while (end - start) < min_len && i + 2 < boundaries.len() {
                i += 1;
                end = boundaries[i + 1];
            }
            merged.push((start, end));
            i += 1;
        }

        // Subdivide long scenes.
        let mut out = Vec::new();
        for (start, end) in merged {
            let len = end - start;
            if len <= max_len {
                out.push((start, end));
            } else {
                // Plan inside the scene; offset by `start`.
                for (a, b) in self.windows(len) {
                    out.push((start + a, start + b));
                }
            }
        }
        out
    }
}

impl Default for ChunkingStrategy {
    fn default() -> Self {
        ChunkingStrategy::MarkdownRecursive {
            max_chunk_size: 2000,
            chunk_overlap: 300,
        }
    }
}

/// A chunk of text with metadata.
#[derive(Debug, Clone)]
pub struct TextChunk {
    pub text: String,
    pub index: usize,
    pub start_offset: usize,
    pub end_offset: usize,
    pub heading_path: Vec<String>,
    pub page_number: Option<usize>,
    pub content_types: Vec<String>,
}

/// Chunk a markdown document according to the given strategy.
pub fn chunk_markdown(text: &str, strategy: &ChunkingStrategy) -> Vec<TextChunk> {
    if text.is_empty() {
        return Vec::new();
    }
    match strategy {
        ChunkingStrategy::FixedSize {
            chunk_size,
            chunk_overlap,
        } => chunk_fixed_size(text, *chunk_size, *chunk_overlap),
        ChunkingStrategy::MarkdownRecursive {
            max_chunk_size,
            chunk_overlap,
        } => chunk_markdown_recursive(text, *max_chunk_size, *chunk_overlap),
        ChunkingStrategy::Sentence {
            max_chunk_size,
            chunk_overlap,
        } => chunk_by_sentence(text, *max_chunk_size, *chunk_overlap),
        ChunkingStrategy::MarkdownHeading {
            level,
            max_chunk_size,
            chunk_overlap,
        } => chunk_markdown_heading(text, *level, *max_chunk_size, *chunk_overlap),
        ChunkingStrategy::Semantic {
            max_chunk_size,
            similarity_threshold,
            window_size,
        } => chunk_semantic(text, *max_chunk_size, *similarity_threshold, *window_size),
        ChunkingStrategy::Page => chunk_by_page(text),
        ChunkingStrategy::Hierarchical {
            coarse_size,
            fine_size,
            fine_overlap,
        } => chunk_hierarchical(text, *coarse_size, *fine_size, *fine_overlap),
        ChunkingStrategy::LateChunking {
            max_chunk_size,
            chunk_overlap,
        } => {
            let mut chunks = chunk_markdown_recursive(text, *max_chunk_size, *chunk_overlap);
            for c in &mut chunks {
                if !c.content_types.iter().any(|t| t == "late_chunked") {
                    c.content_types.push("late_chunked".to_owned());
                }
            }
            chunks
        }
        ChunkingStrategy::ContextualChunking {
            max_chunk_size,
            chunk_overlap,
        } => {
            // Chunk normally; an external orchestrator should call the
            // LLM-context helper [`contextualize_chunks`] on the result
            // before passing chunks to the embedding service.
            let mut chunks = chunk_markdown_recursive(text, *max_chunk_size, *chunk_overlap);
            for c in &mut chunks {
                if !c.content_types.iter().any(|t| t == "contextual_pending") {
                    c.content_types.push("contextual_pending".to_owned());
                }
            }
            chunks
        }
        ChunkingStrategy::None => chunk_none(text),
    }
}

/// Anthropic-style contextual retrieval helper. For each chunk, prepend
/// a 1-2 sentence document-level context produced by an LLM.
///
/// Synchronous, sequential, slow: one LLM call per chunk. Production
/// callers should batch and cache the document-level prompt prefix.
///
/// Returns chunks with their `text` field rewritten as
/// `"<context>\n\n<original chunk text>"`. If `llm_endpoint` is empty or
/// the call fails, the chunk is returned unchanged.
pub fn contextualize_chunks(
    document: &str,
    chunks: Vec<TextChunk>,
    llm_endpoint: &str,
    llm_model: &str,
    llm_api_key: Option<&str>,
) -> Vec<TextChunk> {
    if llm_endpoint.is_empty() || chunks.is_empty() {
        return chunks;
    }
    let client = reqwest::blocking::Client::new();
    chunks
        .into_iter()
        .map(|mut chunk| {
            let prompt = format!(
                "<document>\n{document}\n</document>\n\n\
                Here is the chunk we want to situate within the whole document:\n\
                <chunk>\n{chunk_text}\n</chunk>\n\n\
                Provide a short (1-2 sentence) succinct context to situate this chunk \
                within the overall document for the purposes of improving search retrieval \
                of the chunk. Answer only with the succinct context and nothing else.",
                chunk_text = chunk.text
            );
            let body = serde_json::json!({
                "model": llm_model,
                "messages": [{ "role": "user", "content": prompt }],
                "max_tokens": 200,
                "temperature": 0.0,
            });
            let url = format!("{}/v1/chat/completions", llm_endpoint.trim_end_matches('/'));
            let mut req = client
                .post(&url)
                .json(&body)
                .timeout(std::time::Duration::from_secs(30));
            if let Some(key) = llm_api_key
                && !key.is_empty()
            {
                req = req.bearer_auth(key);
            }
            let context = req
                .send()
                .ok()
                .and_then(|r| r.json::<serde_json::Value>().ok())
                .and_then(|j| {
                    j["choices"][0]["message"]["content"]
                        .as_str()
                        .map(|s| s.trim().to_owned())
                });
            if let Some(ctx) = context
                && !ctx.is_empty()
            {
                chunk.text = format!("{ctx}\n\n{}", chunk.text);
                // Replace placeholder tag with completed marker.
                chunk
                    .content_types
                    .retain(|t| t != "contextual_pending");
                if !chunk.content_types.iter().any(|t| t == "contextualised") {
                    chunk.content_types.push("contextualised".to_owned());
                }
            }
            chunk
        })
        .collect()
}

// ── Hierarchical ──────────────────────────────────────────────────────────

/// Produce coarse + fine chunks for the same document. Coarse chunks come
/// first (each tagged `content_types = ["coarse"]`), followed by all fine
/// chunks (each tagged `content_types = ["fine"]`).
///
/// Indices are renumbered globally so that downstream consumers can treat
/// the output as a flat list. Use the `content_types` tag to filter.
fn chunk_hierarchical(
    text: &str,
    coarse_size: usize,
    fine_size: usize,
    fine_overlap: usize,
) -> Vec<TextChunk> {
    let coarse_chunks = chunk_fixed_size(text, coarse_size, 0);
    let fine_chunks = chunk_fixed_size(text, fine_size, fine_overlap);

    let mut out: Vec<TextChunk> =
        Vec::with_capacity(coarse_chunks.len() + fine_chunks.len());

    for mut c in coarse_chunks {
        c.index = out.len();
        c.content_types = vec!["coarse".to_owned()];
        out.push(c);
    }
    for mut c in fine_chunks {
        c.index = out.len();
        c.content_types = vec!["fine".to_owned()];
        out.push(c);
    }
    out
}

// ── Content-type detection ─────────────────────────────────────────────

fn detect_content_types(text: &str) -> Vec<String> {
    let mut types = Vec::new();
    let has_table = text.lines().any(|l| {
        let trimmed = l.trim();
        trimmed.starts_with('|') && trimmed.ends_with('|')
    });
    if has_table {
        types.push("table".to_owned());
    }
    if text.contains("```") {
        types.push("code".to_owned());
    }
    if text.contains("![") {
        types.push("image_ref".to_owned());
    }
    if types.is_empty() {
        types.push("text".to_owned());
    }
    types
}

// ── FixedSize ──────────────────────────────────────────────────────────

fn chunk_fixed_size(text: &str, chunk_size: usize, chunk_overlap: usize) -> Vec<TextChunk> {
    let step = chunk_size.saturating_sub(chunk_overlap).max(1);
    let mut chunks = Vec::new();
    let mut pos = 0;

    while pos < text.len() {
        let end = (pos + chunk_size).min(text.len());
        // Try to break at a word boundary.
        let actual_end = if end < text.len() {
            text[pos..end]
                .rfind(' ')
                .map(|i| pos + i)
                .unwrap_or(end)
                .max(pos + 1)
        } else {
            end
        };
        let slice = &text[pos..actual_end];
        chunks.push(TextChunk {
            text: slice.to_owned(),
            index: chunks.len(),
            start_offset: pos,
            end_offset: actual_end,
            heading_path: Vec::new(),
            page_number: None,
            content_types: detect_content_types(slice),
        });
        pos += step;
        if actual_end >= text.len() {
            break;
        }
    }
    chunks
}

// ── Sentence ───────────────────────────────────────────────────────────

fn split_sentences(text: &str) -> Vec<&str> {
    let mut sentences = Vec::new();
    let mut start = 0;
    let bytes = text.as_bytes();

    for i in 0..bytes.len() {
        let is_boundary = match bytes[i] {
            b'.' | b'?' | b'!' => {
                // Must be followed by space or end-of-text.
                i + 1 >= bytes.len() || bytes[i + 1] == b' ' || bytes[i + 1] == b'\n'
            }
            b'\n' => true,
            _ => false,
        };
        if is_boundary {
            let end = i + 1;
            let s = &text[start..end];
            if !s.trim().is_empty() {
                sentences.push(s);
            }
            start = end;
            // Skip trailing space after sentence-ending punctuation.
            if start < bytes.len() && bytes[start] == b' ' {
                start += 1;
            }
        }
    }
    if start < text.len() {
        let s = &text[start..];
        if !s.trim().is_empty() {
            sentences.push(s);
        }
    }
    sentences
}

fn chunk_by_sentence(text: &str, max_chunk_size: usize, chunk_overlap: usize) -> Vec<TextChunk> {
    let sentences = split_sentences(text);
    let mut chunks = Vec::new();
    let mut current = String::new();
    let mut start_offset = 0;

    for sent in &sentences {
        if !current.is_empty() && current.len() + sent.len() > max_chunk_size {
            let end_offset = start_offset + current.len();
            chunks.push(TextChunk {
                text: current.clone(),
                index: chunks.len(),
                start_offset,
                end_offset,
                heading_path: Vec::new(),
                page_number: None,
                content_types: detect_content_types(&current),
            });
            // Overlap: keep tail of current chunk.
            if chunk_overlap > 0 && current.len() > chunk_overlap {
                let keep_from = current.len() - chunk_overlap;
                start_offset = end_offset - chunk_overlap;
                current = current[keep_from..].to_owned();
            } else {
                start_offset = end_offset;
                current.clear();
            }
        }
        current.push_str(sent);
    }
    if !current.is_empty() {
        let end_offset = start_offset + current.len();
        chunks.push(TextChunk {
            text: current.clone(),
            index: chunks.len(),
            start_offset,
            end_offset,
            heading_path: Vec::new(),
            page_number: None,
            content_types: detect_content_types(&current),
        });
    }
    chunks
}

// ── Page ───────────────────────────────────────────────────────────────

fn chunk_by_page(text: &str) -> Vec<TextChunk> {
    let pages: Vec<&str> = text.split("\n---\n").collect();
    let mut chunks = Vec::new();
    let mut offset = 0;

    for (i, page) in pages.iter().enumerate() {
        let trimmed = page.trim();
        if !trimmed.is_empty() {
            chunks.push(TextChunk {
                text: trimmed.to_owned(),
                index: chunks.len(),
                start_offset: offset,
                end_offset: offset + page.len(),
                heading_path: Vec::new(),
                page_number: Some(i + 1),
                content_types: detect_content_types(trimmed),
            });
        }
        offset += page.len() + 5; // len of "\n---\n"
    }
    chunks
}

// ── None ───────────────────────────────────────────────────────────────

fn chunk_none(text: &str) -> Vec<TextChunk> {
    vec![TextChunk {
        text: text.to_owned(),
        index: 0,
        start_offset: 0,
        end_offset: text.len(),
        heading_path: Vec::new(),
        page_number: None,
        content_types: detect_content_types(text),
    }]
}

// ── MarkdownHeading ───────────────────────────────────────────────────

fn heading_separator(level: u8) -> Vec<&'static str> {
    match level {
        1 => vec!["\n# "],
        2 => vec!["\n# ", "\n## "],
        3 => vec!["\n# ", "\n## ", "\n### "],
        _ => vec!["\n# ", "\n## ", "\n### "],
    }
}

fn chunk_markdown_heading(
    text: &str,
    level: u8,
    max_chunk_size: usize,
    chunk_overlap: usize,
) -> Vec<TextChunk> {
    let separators = heading_separator(level);

    // Split text hierarchically by heading levels.
    let mut sections = vec![text.to_owned()];
    for sep in &separators {
        let mut new_sections = Vec::new();
        for section in &sections {
            let parts = split_keeping_separator(section, sep);
            new_sections.extend(parts);
        }
        sections = new_sections;
    }

    // For each section, if it exceeds max_chunk_size, sub-split by paragraphs then fixed-size.
    let mut raw_pieces: Vec<RawPiece> = Vec::new();
    let mut current_heading_path: Vec<String> = Vec::new();
    let mut page_number: usize = 1;

    for section in &sections {
        // Track headings.
        for line in section.lines().take(3) {
            let trimmed = line.trim();
            if trimmed.starts_with("### ") {
                if current_heading_path.len() > 2 {
                    current_heading_path.truncate(2);
                }
                current_heading_path.push(trimmed[4..].trim().to_owned());
            } else if trimmed.starts_with("## ") {
                current_heading_path.truncate(1);
                current_heading_path.push(trimmed[3..].trim().to_owned());
            } else if trimmed.starts_with("# ") {
                current_heading_path = vec![trimmed[2..].trim().to_owned()];
            }
        }

        if section.contains("\n---\n") {
            page_number += section.matches("\n---\n").count();
        }

        let trimmed = section.trim();
        if trimmed.is_empty() {
            continue;
        }

        if trimmed.len() <= max_chunk_size {
            raw_pieces.push(RawPiece {
                text: trimmed.to_owned(),
                heading_path: current_heading_path.clone(),
                page_number,
            });
        } else {
            // Sub-split by paragraphs.
            let paragraphs: Vec<&str> = trimmed.split("\n\n").collect();
            let mut current_text = String::new();
            for para in paragraphs {
                if !current_text.is_empty() && current_text.len() + para.len() + 2 > max_chunk_size {
                    raw_pieces.push(RawPiece {
                        text: current_text.clone(),
                        heading_path: current_heading_path.clone(),
                        page_number,
                    });
                    current_text.clear();
                }
                if !current_text.is_empty() {
                    current_text.push_str("\n\n");
                }
                current_text.push_str(para);
            }
            if !current_text.is_empty() {
                raw_pieces.push(RawPiece {
                    text: current_text,
                    heading_path: current_heading_path.clone(),
                    page_number,
                });
            }
        }
    }

    build_chunks_with_overlap(&raw_pieces, chunk_overlap, text)
}

// ── Semantic ──────────────────────────────────────────────────────────

/// Compute word-overlap similarity between two text segments (Jaccard-like).
/// Returns a value between 0.0 (no overlap) and 1.0 (identical word sets).
fn text_similarity(a: &str, b: &str) -> f32 {
    let words_a: std::collections::HashSet<&str> = a.split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
        .filter(|w| w.len() > 2) // skip short words like "a", "to", "is"
        .collect();
    let words_b: std::collections::HashSet<&str> = b.split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
        .filter(|w| w.len() > 2)
        .collect();

    if words_a.is_empty() || words_b.is_empty() {
        return 0.0;
    }

    let intersection = words_a.intersection(&words_b).count();
    let union = words_a.union(&words_b).count();

    if union == 0 { 0.0 } else { intersection as f32 / union as f32 }
}

fn chunk_semantic(
    text: &str,
    max_chunk_size: usize,
    similarity_threshold: f32,
    window_size: usize,
) -> Vec<TextChunk> {
    // Split into sentences first.
    let sentences = split_sentences(text);
    if sentences.is_empty() {
        return Vec::new();
    }

    // Compare consecutive windows of sentences. When similarity drops below
    // threshold, insert a split point.
    let window_size = window_size.max(1);
    let mut split_points: Vec<usize> = Vec::new(); // indices into sentences where splits occur

    for i in window_size..sentences.len() {
        // Window before: sentences[i-window_size..i]
        let before: String = sentences[i.saturating_sub(window_size)..i].join(" ");
        // Window after: sentences[i..i+window_size]
        let after_end = (i + window_size).min(sentences.len());
        let after: String = sentences[i..after_end].join(" ");

        let sim = text_similarity(&before, &after);
        if sim < similarity_threshold {
            // Check we don't split too close to previous split.
            let last_split = split_points.last().copied().unwrap_or(0);
            if i - last_split >= window_size {
                split_points.push(i);
            }
        }
    }

    // Build chunks from split points.
    let mut chunks = Vec::new();
    let mut start_idx = 0;
    let mut offset = 0;

    let mut all_points = split_points.clone();
    all_points.push(sentences.len()); // sentinel

    for &end_idx in &all_points {
        let chunk_text: String = sentences[start_idx..end_idx].join("");

        if chunk_text.trim().is_empty() {
            start_idx = end_idx;
            continue;
        }

        // If chunk exceeds max, sub-split with fixed-size.
        if chunk_text.len() > max_chunk_size {
            let sub_chunks = chunk_fixed_size(&chunk_text, max_chunk_size, max_chunk_size / 7);
            for mut sc in sub_chunks {
                sc.index = chunks.len();
                sc.start_offset += offset;
                sc.end_offset += offset;
                chunks.push(sc);
            }
        } else {
            chunks.push(TextChunk {
                text: chunk_text.clone(),
                index: chunks.len(),
                start_offset: offset,
                end_offset: offset + chunk_text.len(),
                heading_path: Vec::new(),
                page_number: None,
                content_types: detect_content_types(&chunk_text),
            });
        }

        offset += chunk_text.len();
        start_idx = end_idx;
    }

    chunks
}

// ── MarkdownRecursive ──────────────────────────────────────────────────

/// Separators ordered from coarsest to finest.
const SEPARATORS: &[&str] = &[
    "\n# ", "\n## ", "\n### ", "\n---\n", "\n\n", "\n", ". ", " ",
];

struct RecursiveCtx {
    max_chunk_size: usize,
    heading_path: Vec<String>,
    page_number: usize,
}

fn chunk_markdown_recursive(text: &str, max_chunk_size: usize, chunk_overlap: usize) -> Vec<TextChunk> {
    let mut ctx = RecursiveCtx {
        max_chunk_size,
        heading_path: Vec::new(),
        page_number: 1,
    };
    let raw_pieces = recursive_split(text, 0, &mut ctx);

    // Merge small adjacent pieces up to max_chunk_size.
    let merged = merge_small_pieces(&raw_pieces, max_chunk_size);

    // Build final chunks with overlap and metadata.
    build_chunks_with_overlap(&merged, chunk_overlap, text)
}

/// A raw piece from recursive splitting, carrying metadata.
#[derive(Clone)]
struct RawPiece {
    text: String,
    heading_path: Vec<String>,
    page_number: usize,
}

fn recursive_split(text: &str, sep_idx: usize, ctx: &mut RecursiveCtx) -> Vec<RawPiece> {
    if text.trim().is_empty() {
        return Vec::new();
    }
    if text.len() <= ctx.max_chunk_size || sep_idx >= SEPARATORS.len() {
        return vec![RawPiece {
            text: text.to_owned(),
            heading_path: ctx.heading_path.clone(),
            page_number: ctx.page_number,
        }];
    }

    let sep = SEPARATORS[sep_idx];
    let parts = split_keeping_separator(text, sep);

    if parts.len() <= 1 {
        // This separator doesn't split the text; try the next finer one.
        return recursive_split(text, sep_idx + 1, ctx);
    }

    let mut pieces = Vec::new();
    for part in &parts {
        // Update heading path for heading separators.
        update_heading_context(part, sep, ctx);

        if part.len() <= ctx.max_chunk_size {
            pieces.push(RawPiece {
                text: part.clone(),
                heading_path: ctx.heading_path.clone(),
                page_number: ctx.page_number,
            });
        } else {
            pieces.extend(recursive_split(part, sep_idx + 1, ctx));
        }
    }
    pieces
}

fn split_keeping_separator(text: &str, sep: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut remaining = text;

    // For heading separators that start with \n, the first part may not start with the separator.
    if let Some(idx) = remaining.find(sep) {
        if idx > 0 {
            parts.push(remaining[..idx].to_owned());
            remaining = &remaining[idx..];
        }
    }

    while !remaining.is_empty() {
        // remaining starts with sep (or is the first piece without it).
        let search_start = if remaining.starts_with(sep) { sep.len() } else { 1 };
        if let Some(next_idx) = remaining[search_start..].find(sep) {
            let split_at = search_start + next_idx;
            parts.push(remaining[..split_at].to_owned());
            remaining = &remaining[split_at..];
        } else {
            parts.push(remaining.to_owned());
            break;
        }
    }
    parts
}

fn update_heading_context(part: &str, sep: &str, ctx: &mut RecursiveCtx) {
    match sep {
        "\n# " => {
            let heading = extract_heading_text(part, "# ");
            if let Some(h) = heading {
                ctx.heading_path = vec![h];
            }
        }
        "\n## " => {
            let heading = extract_heading_text(part, "## ");
            if let Some(h) = heading {
                ctx.heading_path.truncate(1);
                ctx.heading_path.push(h);
            }
        }
        "\n### " => {
            let heading = extract_heading_text(part, "### ");
            if let Some(h) = heading {
                // Keep H1 and H2, replace H3+.
                if ctx.heading_path.len() > 2 {
                    ctx.heading_path.truncate(2);
                }
                ctx.heading_path.push(h);
            }
        }
        "\n---\n" => {
            ctx.page_number += 1;
        }
        _ => {}
    }
}

fn extract_heading_text(part: &str, prefix: &str) -> Option<String> {
    // The part may start with "\n# Heading\n..." or "# Heading\n..."
    for line in part.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with(prefix) {
            return Some(trimmed[prefix.len()..].trim().to_owned());
        }
        // Also handle "# Heading" at the very start.
        if trimmed.starts_with(&prefix[..prefix.len()]) {
            let after = trimmed.trim_start_matches('#').trim();
            if !after.is_empty() {
                return Some(after.to_owned());
            }
        }
    }
    None
}

fn merge_small_pieces(pieces: &[RawPiece], max_chunk_size: usize) -> Vec<RawPiece> {
    let mut merged = Vec::new();
    let mut current: Option<RawPiece> = None;

    for piece in pieces {
        if let Some(ref mut cur) = current {
            if cur.text.len() + piece.text.len() <= max_chunk_size {
                cur.text.push_str(&piece.text);
                continue;
            }
            merged.push(cur.clone());
        }
        current = Some(piece.clone());
    }
    if let Some(cur) = current {
        merged.push(cur);
    }
    merged
}

fn build_chunks_with_overlap(
    pieces: &[RawPiece],
    chunk_overlap: usize,
    _original: &str,
) -> Vec<TextChunk> {
    let mut chunks = Vec::new();
    let mut offset: usize = 0;

    for (i, piece) in pieces.iter().enumerate() {
        let mut text = String::new();

        // Prepend overlap from previous piece.
        if chunk_overlap > 0 && i > 0 {
            let prev = &pieces[i - 1].text;
            if prev.len() > chunk_overlap {
                text.push_str(&prev[prev.len() - chunk_overlap..]);
            } else {
                text.push_str(prev);
            }
        }

        let start_offset = if text.is_empty() { offset } else { offset.saturating_sub(text.len()) };
        text.push_str(&piece.text);

        chunks.push(TextChunk {
            text: text.clone(),
            index: i,
            start_offset,
            end_offset: start_offset + text.len(),
            heading_path: piece.heading_path.clone(),
            page_number: Some(piece.page_number),
            content_types: detect_content_types(&text),
        });
        offset += piece.text.len();
    }
    chunks
}
