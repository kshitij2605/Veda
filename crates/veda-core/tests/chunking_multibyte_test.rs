//! Regression for BUG-2 (2026-05-21 E2E campaign): the `FixedSize`
//! chunker called `text[pos..end]` with `end = pos + chunk_size`
//! computed in bytes. On multi-byte text (Japanese / Chinese / emoji)
//! this could land inside a UTF-8 sequence and panic with
//! `byte index N is not a char boundary; it is inside '空'`.
//!
//! Fix: snap `end` (and the next `pos`) down to a char boundary.

use veda_core::chunking::{ChunkingStrategy, chunk_markdown};

#[test]
fn fixed_size_chunker_handles_japanese_without_panic() {
    // Japanese text where every char is 3 bytes — placing
    // chunk_size at a value that doesn't divide cleanly by 3
    // is guaranteed to land in the middle of a char.
    let text: String = "東京都は日本の首都です。".repeat(50);
    let strategy = ChunkingStrategy::FixedSize {
        chunk_size: 100,
        chunk_overlap: 0,
    };
    // Must not panic.
    let chunks = chunk_markdown(&text, &strategy);
    assert!(!chunks.is_empty(), "should produce at least one chunk");
    for c in &chunks {
        // Every produced chunk must itself be valid UTF-8 (it
        // already is by Rust's type system since `text` is a String,
        // but we also check it's non-empty and round-trips).
        assert!(!c.text.is_empty());
        let _: &str = c.text.as_str();
    }
}

#[test]
fn fixed_size_chunker_handles_mixed_ascii_japanese_emoji() {
    let text = "Hello, 世界! 🌏 This is a mix. ".repeat(80);
    let strategy = ChunkingStrategy::FixedSize {
        chunk_size: 50,
        chunk_overlap: 5,
    };
    let chunks = chunk_markdown(&text, &strategy);
    assert!(!chunks.is_empty());
    // Reassemble and confirm characters survived intact (every
    // 4-byte emoji and 3-byte CJK appears whole somewhere).
    let joined: String = chunks.iter().map(|c| c.text.as_str()).collect();
    assert!(joined.contains("世界"));
    assert!(joined.contains("🌏"));
}

#[test]
fn fixed_size_chunker_handles_chunk_size_smaller_than_one_char() {
    // chunk_size = 2 < the 3-byte width of '空'. The chunker must
    // still produce ≥1 chunk per char, not infinite loop, not panic.
    let text = "空白空白";
    let strategy = ChunkingStrategy::FixedSize {
        chunk_size: 2,
        chunk_overlap: 0,
    };
    let chunks = chunk_markdown(text, &strategy);
    assert!(!chunks.is_empty());
    // Should still cover the full text.
    let joined: String = chunks.iter().map(|c| c.text.as_str()).collect();
    assert!(joined.contains("空"));
}

#[test]
fn sentence_chunker_overlap_handles_japanese_without_panic() {
    // chunk_by_sentence keeps a tail of `chunk_overlap` bytes from the
    // previous chunk. On Japanese text (3 bytes/char) a raw subtraction
    // can land mid-char, panicking with "not a char boundary".
    let text = "東京は首都です. 大阪は商都です. 名古屋は製造業です. ".repeat(30);
    let strategy = ChunkingStrategy::Sentence {
        max_chunk_size: 100,
        chunk_overlap: 25,
    };
    let chunks = chunk_markdown(&text, &strategy);
    assert!(chunks.len() > 1, "should produce multiple chunks");
    for c in &chunks {
        assert!(!c.text.is_empty());
        let _: &str = c.text.as_str();
    }
}

#[test]
fn sentence_chunker_overlap_handles_emoji_boundaries() {
    let text =
        "Hello world! 🌏 This is fine. Another sentence! 🎉 More text here. Final bit! 🚀 Done."
            .repeat(10);
    let strategy = ChunkingStrategy::Sentence {
        max_chunk_size: 60,
        chunk_overlap: 15,
    };
    let chunks = chunk_markdown(&text, &strategy);
    assert!(chunks.len() > 1);
    for c in &chunks {
        assert!(!c.text.is_empty());
    }
}

#[test]
fn sentence_chunker_splits_on_cjk_full_stop() {
    let text = "東京は首都です。大阪は商都です。名古屋は製造業の中心です。";
    let strategy = ChunkingStrategy::Sentence {
        max_chunk_size: 30,
        chunk_overlap: 0,
    };
    let chunks = chunk_markdown(text, &strategy);
    assert!(
        chunks.len() >= 3,
        "should split on CJK full stop, got {} chunks: {:?}",
        chunks.len(),
        chunks.iter().map(|c| &c.text).collect::<Vec<_>>()
    );
}

#[test]
fn sentence_chunker_splits_on_fullwidth_question_mark() {
    let text = "これは何ですか？はい、そうです。いいえ、違います！";
    let strategy = ChunkingStrategy::Sentence {
        max_chunk_size: 30,
        chunk_overlap: 0,
    };
    let chunks = chunk_markdown(text, &strategy);
    assert!(
        chunks.len() >= 3,
        "should split on fullwidth punctuation, got {} chunks",
        chunks.len()
    );
}
