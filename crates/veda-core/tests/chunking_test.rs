use veda_core::chunking::{chunk_markdown, ChunkingStrategy};

#[test]
fn default_strategy_is_markdown_recursive() {
    let strategy = ChunkingStrategy::default();
    match strategy {
        ChunkingStrategy::MarkdownRecursive {
            max_chunk_size,
            chunk_overlap,
        } => {
            assert_eq!(max_chunk_size, 2000);
            assert_eq!(chunk_overlap, 300);
        }
        _ => panic!("default should be MarkdownRecursive"),
    }
}

#[test]
fn empty_input_returns_empty() {
    let chunks = chunk_markdown("", &ChunkingStrategy::default());
    assert!(chunks.is_empty());

    let chunks = chunk_markdown("", &ChunkingStrategy::Page);
    assert!(chunks.is_empty());

    let chunks = chunk_markdown("", &ChunkingStrategy::None);
    assert!(chunks.is_empty());
}

#[test]
fn none_strategy_returns_single_chunk() {
    let text = "Hello world. This is a test.";
    let chunks = chunk_markdown(text, &ChunkingStrategy::None);
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].text, text);
    assert_eq!(chunks[0].index, 0);
}

#[test]
fn page_strategy_splits_on_separators() {
    let text = "Page one content\n---\nPage two content\n---\nPage three content";
    let chunks = chunk_markdown(text, &ChunkingStrategy::Page);
    assert_eq!(chunks.len(), 3);
    assert_eq!(chunks[0].text, "Page one content");
    assert_eq!(chunks[1].text, "Page two content");
    assert_eq!(chunks[2].text, "Page three content");
    assert_eq!(chunks[0].page_number, Some(1));
    assert_eq!(chunks[1].page_number, Some(2));
    assert_eq!(chunks[2].page_number, Some(3));
}

#[test]
fn fixed_size_chunks_are_approximately_equal() {
    let text = "a ".repeat(500); // 1000 chars
    let chunks = chunk_markdown(
        &text,
        &ChunkingStrategy::FixedSize {
            chunk_size: 200,
            chunk_overlap: 0,
        },
    );
    assert!(chunks.len() >= 4);
    for chunk in &chunks {
        assert!(chunk.text.len() <= 200);
    }
}

#[test]
fn fixed_size_overlap_is_correct() {
    let text = "word ".repeat(200); // 1000 chars
    let chunks = chunk_markdown(
        &text,
        &ChunkingStrategy::FixedSize {
            chunk_size: 200,
            chunk_overlap: 50,
        },
    );
    assert!(chunks.len() > 1);
    // Check overlap: end of chunk[i] should appear at start of chunk[i+1].
    for i in 0..chunks.len() - 1 {
        // Due to word-boundary breaking, we check that there's overlap in offsets.
        assert!(chunks[i + 1].start_offset < chunks[i].end_offset);
    }
}

#[test]
fn sentence_strategy_respects_sentence_boundaries() {
    let text = "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence.";
    let chunks = chunk_markdown(
        text,
        &ChunkingStrategy::Sentence {
            max_chunk_size: 40,
            chunk_overlap: 0,
        },
    );
    assert!(chunks.len() > 1);
    // Each chunk should end at a sentence boundary (ends with period or contains complete sentences).
    for chunk in &chunks {
        assert!(
            chunk.text.trim().ends_with('.') || chunk.text.trim().ends_with('?') || chunk.text.trim().ends_with('!'),
            "Chunk should end at sentence boundary: '{}'",
            chunk.text
        );
    }
}

#[test]
fn markdown_recursive_respects_heading_boundaries() {
    let text = "# Chapter 1\n\nSome intro text.\n\n## Section 1.1\n\nContent of section 1.1.\n\n## Section 1.2\n\nContent of section 1.2.\n\n# Chapter 2\n\nContent of chapter 2.";
    let chunks = chunk_markdown(
        text,
        &ChunkingStrategy::MarkdownRecursive {
            max_chunk_size: 80,
            chunk_overlap: 0,
        },
    );
    assert!(chunks.len() > 1);
    // All chunks should fit within max_chunk_size (with some tolerance for overlap and word boundaries).
    for chunk in &chunks {
        assert!(
            chunk.text.len() <= 88, // 10% tolerance
            "Chunk too large: {} chars: '{}'",
            chunk.text.len(),
            &chunk.text[..chunk.text.len().min(50)]
        );
    }
}

#[test]
fn markdown_recursive_overlap() {
    let text = "# Heading One\n\nParagraph one with enough text to make it substantial for testing overlap behavior in the chunking algorithm.\n\n# Heading Two\n\nParagraph two also with enough text to verify that overlap content is properly shared between chunks.";
    let chunks = chunk_markdown(
        text,
        &ChunkingStrategy::MarkdownRecursive {
            max_chunk_size: 120,
            chunk_overlap: 30,
        },
    );
    if chunks.len() > 1 {
        // Second chunk should contain overlap from first.
        let first_tail = &chunks[0].text[chunks[0].text.len().saturating_sub(30)..];
        assert!(
            chunks[1].text.contains(first_tail) || chunks[1].start_offset < chunks[0].end_offset,
            "Chunks should share overlap"
        );
    }
}

#[test]
fn markdown_recursive_size_limit() {
    // Generate a large document.
    let mut text = String::new();
    for i in 0..20 {
        text.push_str(&format!("## Section {}\n\n", i));
        text.push_str(&"Lorem ipsum dolor sit amet. ".repeat(20));
        text.push('\n');
    }
    let max = 2000;
    let chunks = chunk_markdown(
        &text,
        &ChunkingStrategy::MarkdownRecursive {
            max_chunk_size: max,
            chunk_overlap: 300,
        },
    );
    for chunk in &chunks {
        let tolerance = (max as f64 * 1.1) as usize;
        assert!(
            chunk.text.len() <= tolerance,
            "Chunk exceeds size limit: {} > {}",
            chunk.text.len(),
            tolerance
        );
    }
}

#[test]
fn content_type_detection_table() {
    let text = "Some text\n\n| Col1 | Col2 |\n| --- | --- |\n| a | b |\n\nMore text";
    let chunks = chunk_markdown(text, &ChunkingStrategy::None);
    assert!(chunks[0].content_types.contains(&"table".to_owned()));
}

#[test]
fn content_type_detection_code() {
    let text = "Some text\n\n```rust\nfn main() {}\n```\n\nMore text";
    let chunks = chunk_markdown(text, &ChunkingStrategy::None);
    assert!(chunks[0].content_types.contains(&"code".to_owned()));
}

#[test]
fn content_type_detection_image_ref() {
    let text = "Some text\n\n![alt text](image.png)\n\nMore text";
    let chunks = chunk_markdown(text, &ChunkingStrategy::None);
    assert!(chunks[0].content_types.contains(&"image_ref".to_owned()));
}

#[test]
fn heading_path_tracking() {
    let mut text = String::new();
    text.push_str("# Chapter 1\n\n");
    text.push_str("## Section 1.2\n\n");
    text.push_str(&"Content here. ".repeat(200)); // Force splitting
    text.push_str("\n\n# Chapter 2\n\n");
    text.push_str(&"More content. ".repeat(200));

    let chunks = chunk_markdown(
        &text,
        &ChunkingStrategy::MarkdownRecursive {
            max_chunk_size: 500,
            chunk_overlap: 0,
        },
    );
    // At least one chunk should have heading path populated.
    let has_headings = chunks.iter().any(|c| !c.heading_path.is_empty());
    assert!(has_headings, "Some chunks should have heading_path populated");
}

#[test]
fn page_number_tracking_in_markdown_recursive() {
    let text = "Page 1 content\n---\nPage 2 content\n---\nPage 3 content";
    let chunks = chunk_markdown(
        text,
        &ChunkingStrategy::MarkdownRecursive {
            max_chunk_size: 2000,
            chunk_overlap: 0,
        },
    );
    // Should have page numbers set.
    let page_numbers: Vec<Option<usize>> = chunks.iter().map(|c| c.page_number).collect();
    assert!(!page_numbers.is_empty());
    // First chunk should be page 1.
    assert_eq!(chunks[0].page_number, Some(1));
}
