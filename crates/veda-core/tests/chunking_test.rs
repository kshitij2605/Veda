use veda_core::chunking::{ChunkingStrategy, TimeWindowPlan, chunk_markdown};

#[test]
fn default_strategy_is_markdown_recursive() {
    let strategy = ChunkingStrategy::default();
    match strategy {
        ChunkingStrategy::MarkdownRecursive {
            max_chunk_size,
            chunk_overlap,
        } => {
            // Retuned 2026-05-19 per research/2026-05-19-sota-audit/07-chunking-ocr.md
            assert_eq!(max_chunk_size, 1500);
            assert_eq!(chunk_overlap, 0);
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
            chunk.text.trim().ends_with('.')
                || chunk.text.trim().ends_with('?')
                || chunk.text.trim().ends_with('!'),
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
    assert!(
        has_headings,
        "Some chunks should have heading_path populated"
    );
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

// ── MarkdownHeading Tests ─────────────────────────────────────────────

#[test]
fn markdown_heading_level_1_splits_on_h1() {
    let text = "# Chapter 1\n\nSome content about chapter one.\n\n# Chapter 2\n\nSome content about chapter two.";
    let chunks = chunk_markdown(
        text,
        &ChunkingStrategy::MarkdownHeading {
            level: 1,
            max_chunk_size: 5000,
            chunk_overlap: 0,
        },
    );
    assert!(
        chunks.len() >= 2,
        "Should split on H1 headings, got {} chunks",
        chunks.len()
    );
    assert!(chunks[0].text.contains("Chapter 1") || chunks[0].text.contains("chapter one"));
}

#[test]
fn markdown_heading_level_2_splits_on_h1_and_h2() {
    let text = "# Title\n\n## Section A\n\nContent A.\n\n## Section B\n\nContent B.";
    let chunks = chunk_markdown(
        text,
        &ChunkingStrategy::MarkdownHeading {
            level: 2,
            max_chunk_size: 5000,
            chunk_overlap: 0,
        },
    );
    assert!(
        chunks.len() >= 2,
        "Should split on H1/H2, got {} chunks",
        chunks.len()
    );
}

#[test]
fn markdown_heading_level_3_splits_on_all_headings() {
    let text =
        "# H1\n\n## H2\n\nContent.\n\n### H3a\n\nSub content a.\n\n### H3b\n\nSub content b.";
    let chunks = chunk_markdown(
        text,
        &ChunkingStrategy::MarkdownHeading {
            level: 3,
            max_chunk_size: 5000,
            chunk_overlap: 0,
        },
    );
    assert!(
        chunks.len() >= 3,
        "Should split on H1/H2/H3, got {} chunks",
        chunks.len()
    );
}

#[test]
fn markdown_heading_subsplits_large_sections() {
    let mut text = "# Big Section\n\n".to_string();
    for i in 0..50 {
        text.push_str(&format!(
            "Paragraph {} with enough content to make it sizeable. ",
            i
        ));
        text.push_str("This is additional text to bulk up the paragraph.\n\n");
    }
    let chunks = chunk_markdown(
        &text,
        &ChunkingStrategy::MarkdownHeading {
            level: 1,
            max_chunk_size: 500,
            chunk_overlap: 0,
        },
    );
    assert!(
        chunks.len() > 1,
        "Large section should be sub-split by paragraphs"
    );
    for chunk in &chunks {
        // Allow some tolerance (paragraphs may not split exactly at boundary).
        assert!(
            chunk.text.len() <= 600,
            "Chunk too large: {} chars",
            chunk.text.len()
        );
    }
}

#[test]
fn markdown_heading_preserves_heading_path() {
    let text = "# Chapter\n\n## Section\n\nContent under section.";
    let chunks = chunk_markdown(
        text,
        &ChunkingStrategy::MarkdownHeading {
            level: 2,
            max_chunk_size: 5000,
            chunk_overlap: 0,
        },
    );
    let has_path = chunks.iter().any(|c| !c.heading_path.is_empty());
    assert!(has_path, "Should track heading path");
}

// ── Semantic Chunking Tests ───────────────────────────────────────────

#[test]
fn semantic_splits_on_topic_change() {
    // Two clearly different topics with distinct vocabulary.
    let text = "The cat sat on the mat. The cat played with yarn. The cat slept by the fire. \
                The stock market crashed today. Investors lost billions. The economy is in recession. \
                Financial markets are volatile. Trading was halted on multiple exchanges.";
    let chunks = chunk_markdown(
        text,
        &ChunkingStrategy::Semantic {
            max_chunk_size: 2000,
            similarity_threshold: 0.3,
            window_size: 2,
        },
    );
    assert!(
        chunks.len() >= 2,
        "Should split between cats and finance topics, got {} chunks",
        chunks.len()
    );
}

#[test]
fn semantic_keeps_similar_content_together() {
    // All about the same topic — should produce fewer splits.
    let text = "Machine learning is a subset of artificial intelligence. \
                Deep learning uses neural networks with many layers. \
                Convolutional neural networks are used for image recognition. \
                Recurrent neural networks are used for sequential data. \
                Transformer models have revolutionized natural language processing.";
    let chunks = chunk_markdown(
        text,
        &ChunkingStrategy::Semantic {
            max_chunk_size: 2000,
            similarity_threshold: 0.3,
            window_size: 2,
        },
    );
    // Similar topic should have few or no splits.
    assert!(
        chunks.len() <= 3,
        "Similar content should stay together, got {} chunks",
        chunks.len()
    );
}

#[test]
fn semantic_respects_max_chunk_size() {
    let text = "Word. ".repeat(500);
    let chunks = chunk_markdown(
        &text,
        &ChunkingStrategy::Semantic {
            max_chunk_size: 200,
            similarity_threshold: 0.5,
            window_size: 2,
        },
    );
    for chunk in &chunks {
        // Allow some tolerance for sub-splitting.
        assert!(
            chunk.text.len() <= 250,
            "Chunk exceeds max: {} chars",
            chunk.text.len()
        );
    }
}

#[test]
fn semantic_empty_input() {
    let chunks = chunk_markdown(
        "",
        &ChunkingStrategy::Semantic {
            max_chunk_size: 2000,
            similarity_threshold: 0.5,
            window_size: 3,
        },
    );
    assert!(chunks.is_empty());
}

#[test]
fn semantic_single_sentence() {
    let chunks = chunk_markdown(
        "Just one sentence here.",
        &ChunkingStrategy::Semantic {
            max_chunk_size: 2000,
            similarity_threshold: 0.5,
            window_size: 3,
        },
    );
    assert_eq!(chunks.len(), 1);
}

#[test]
fn hierarchical_produces_coarse_and_fine() {
    let text = "Paragraph one. ".repeat(100);
    let chunks = chunk_markdown(
        &text,
        &ChunkingStrategy::Hierarchical {
            coarse_size: 400,
            fine_size: 100,
            fine_overlap: 0,
        },
    );

    let coarse: Vec<_> = chunks
        .iter()
        .filter(|c| c.content_types.iter().any(|t| t == "coarse"))
        .collect();
    let fine: Vec<_> = chunks
        .iter()
        .filter(|c| c.content_types.iter().any(|t| t == "fine"))
        .collect();

    assert!(!coarse.is_empty(), "expected coarse chunks");
    assert!(!fine.is_empty(), "expected fine chunks");
    assert!(
        fine.len() > coarse.len(),
        "fine chunks should outnumber coarse"
    );
    for w in chunks.windows(2) {
        assert!(w[0].index < w[1].index);
    }
}

#[test]
fn time_window_default_15s_overlap_3s() {
    let plan = TimeWindowPlan::default();
    // 30s clip → expect windows at [0,15], [12,27], [24,30].
    let windows = plan.windows(30_000);
    assert_eq!(
        windows,
        vec![(0, 15_000), (12_000, 27_000), (24_000, 30_000)]
    );
}

#[test]
fn time_window_handles_empty_and_short() {
    let plan = TimeWindowPlan::default();
    assert!(plan.windows(0).is_empty());
    // Shorter than one window: a single (0, total) range.
    assert_eq!(plan.windows(5_000), vec![(0, 5_000)]);
}

#[test]
fn time_window_no_overlap() {
    let plan = TimeWindowPlan {
        window_ms: 10_000,
        overlap_ms: 0,
    };
    let windows = plan.windows(25_000);
    assert_eq!(
        windows,
        vec![(0, 10_000), (10_000, 20_000), (20_000, 25_000)]
    );
}

#[test]
fn scene_aligned_windows_respects_cuts() {
    let plan = TimeWindowPlan {
        window_ms: 15_000,
        overlap_ms: 0,
    };
    // 60s video with cuts at 12s, 27s, 50s → scenes [0,12], [12,27], [27,50], [50,60]
    let out = plan.scene_aligned_windows(&[12_000, 27_000, 50_000], 60_000);
    // Each scene should map to at least one (start, end) range.
    assert!(out.contains(&(0, 12_000)));
    assert!(out.contains(&(12_000, 27_000)));
}

#[test]
fn scene_aligned_short_scenes_merge() {
    let plan = TimeWindowPlan {
        window_ms: 15_000,
        overlap_ms: 0,
    };
    // Cuts at 2s and 4s — both scenes are <7.5s (min_len = 7500), so they
    // merge with the next.
    let out = plan.scene_aligned_windows(&[2_000, 4_000], 30_000);
    // First emitted range should span from 0 well past 4_000 because of merging.
    assert!(out[0].0 == 0);
    assert!(out[0].1 > 4_000);
}

#[test]
fn scene_aligned_long_scenes_subdivide() {
    let plan = TimeWindowPlan {
        window_ms: 15_000,
        overlap_ms: 0,
    };
    // No cuts in a 60s video — the whole thing is one scene, longer than
    // 22.5s, so it should be subdivided into ~4 windows.
    let out = plan.scene_aligned_windows(&[], 60_000);
    assert!(
        out.len() >= 4,
        "expected subdivision, got {} windows",
        out.len()
    );
}

#[test]
fn late_chunking_matches_markdown_recursive_at_text_layer() {
    let text = "# Heading\n\nFirst paragraph. ".repeat(50);
    let mr = chunk_markdown(
        &text,
        &ChunkingStrategy::MarkdownRecursive {
            max_chunk_size: 200,
            chunk_overlap: 30,
        },
    );
    let lc = chunk_markdown(
        &text,
        &ChunkingStrategy::LateChunking {
            max_chunk_size: 200,
            chunk_overlap: 30,
        },
    );
    assert_eq!(mr.len(), lc.len());
    for (a, b) in mr.iter().zip(lc.iter()) {
        assert_eq!(a.start_offset, b.start_offset);
        assert_eq!(a.end_offset, b.end_offset);
    }
}
