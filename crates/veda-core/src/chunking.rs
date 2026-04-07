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
    /// No chunking — entire document is one chunk.
    None,
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
        ChunkingStrategy::Page => chunk_by_page(text),
        ChunkingStrategy::None => chunk_none(text),
    }
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
