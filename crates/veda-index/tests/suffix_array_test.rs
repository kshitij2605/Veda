use tempfile::NamedTempFile;
use veda_index::suffix_array::{PatternSearchConfig, SuffixArrayIndex};

// ────────────────────────────────────────────────────────────────────────────
// Helpers
// ────────────────────────────────────────────────────────────────────────────

fn sample_documents() -> Vec<(u32, String)> {
    (0..100)
        .map(|i| {
            let text = format!(
                "Document {} contains information about topic {} and electronics research.",
                i, i
            );
            (i as u32, text)
        })
        .collect()
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[test]
fn test_exact_search() {
    let docs = sample_documents();
    let doc_refs: Vec<(u32, &str)> = docs.iter().map(|(id, t)| (*id, t.as_str())).collect();
    let index = SuffixArrayIndex::build(&doc_refs);

    let matches = index.search_exact("Document 42");
    assert!(!matches.is_empty(), "Should find 'Document 42'");
    assert!(
        matches.iter().any(|m| m.doc_id == 42),
        "Should match doc_id 42"
    );
    // Verify the offset is 0 (phrase starts at beginning of the document text).
    let m42 = matches.iter().find(|m| m.doc_id == 42).unwrap();
    assert_eq!(m42.offset, 0);
}

#[test]
fn test_multiple_matches() {
    let docs = sample_documents();
    let doc_refs: Vec<(u32, &str)> = docs.iter().map(|(id, t)| (*id, t.as_str())).collect();
    let index = SuffixArrayIndex::build(&doc_refs);

    // "electronics" appears in every document.
    let matches = index.search_exact("electronics");
    let doc_ids: std::collections::HashSet<u32> = matches.iter().map(|m| m.doc_id).collect();
    assert_eq!(doc_ids.len(), 100, "Should find match in all 100 documents");
}

#[test]
fn test_no_matches() {
    let docs = sample_documents();
    let doc_refs: Vec<(u32, &str)> = docs.iter().map(|(id, t)| (*id, t.as_str())).collect();
    let index = SuffixArrayIndex::build(&doc_refs);

    let matches = index.search_exact("zzzznonexistent");
    assert!(matches.is_empty(), "Should find no matches");
}

#[test]
fn test_fuzzy_search_substitution() {
    let docs: Vec<(u32, String)> = vec![
        (0, "The electronics department is open.".to_string()),
        (1, "We sell fresh produce daily.".to_string()),
    ];
    let doc_refs: Vec<(u32, &str)> = docs.iter().map(|(id, t)| (*id, t.as_str())).collect();
    let index = SuffixArrayIndex::build(&doc_refs);

    let config = PatternSearchConfig {
        fuzzy: true,
        max_edits: 1,
        preserve_order: true,
        max_results: 100,
    };
    // "eloctronics" has one substitution (e->o at position 2).
    let matches = index.search_fuzzy("eloctronics", &config);
    assert!(
        matches.iter().any(|m| m.doc_id == 0),
        "Fuzzy substitution should find 'electronics'"
    );
}

#[test]
fn test_fuzzy_search_insertion() {
    let docs: Vec<(u32, String)> = vec![
        (0, "The electronics department is open.".to_string()),
        (1, "We sell fresh produce daily.".to_string()),
    ];
    let doc_refs: Vec<(u32, &str)> = docs.iter().map(|(id, t)| (*id, t.as_str())).collect();
    let index = SuffixArrayIndex::build(&doc_refs);

    let config = PatternSearchConfig {
        fuzzy: true,
        max_edits: 1,
        preserve_order: true,
        max_results: 100,
    };
    // "electroonics" has one extra 'o' — the correct word "electronics"
    // is a deletion variant of "electroonics".
    let matches = index.search_fuzzy("electroonics", &config);
    assert!(
        matches.iter().any(|m| m.doc_id == 0),
        "Fuzzy insertion should find 'electronics'"
    );
}

#[test]
fn test_fuzzy_search_deletion() {
    let docs: Vec<(u32, String)> = vec![
        (0, "The electronics department is open.".to_string()),
        (1, "We sell fresh produce daily.".to_string()),
    ];
    let doc_refs: Vec<(u32, &str)> = docs.iter().map(|(id, t)| (*id, t.as_str())).collect();
    let index = SuffixArrayIndex::build(&doc_refs);

    let config = PatternSearchConfig {
        fuzzy: true,
        max_edits: 1,
        preserve_order: true,
        max_results: 100,
    };
    // "electroncs" is missing the 'i' — "electronics" is an insertion variant.
    let matches = index.search_fuzzy("electroncs", &config);
    assert!(
        matches.iter().any(|m| m.doc_id == 0),
        "Fuzzy deletion should find 'electronics'"
    );
}

#[test]
fn test_word_order_preservation() {
    let docs: Vec<(u32, String)> = vec![
        (0, "I bought red shoes yesterday.".to_string()),
        (1, "The shoes are red and comfortable.".to_string()),
    ];
    let doc_refs: Vec<(u32, &str)> = docs.iter().map(|(id, t)| (*id, t.as_str())).collect();
    let index = SuffixArrayIndex::build(&doc_refs);

    let config = PatternSearchConfig {
        fuzzy: false,
        max_edits: 0,
        preserve_order: true,
        max_results: 100,
    };
    let matches = index.search_fuzzy("red shoes", &config);
    let doc_ids: Vec<u32> = matches.iter().map(|m| m.doc_id).collect();
    assert!(doc_ids.contains(&0), "Doc 0 has 'red' before 'shoes'");
    assert!(
        !doc_ids.contains(&1),
        "Doc 1 has 'shoes' before 'red', should be excluded"
    );
}

#[test]
fn test_search_doc_ids() {
    let docs = sample_documents();
    let doc_refs: Vec<(u32, &str)> = docs.iter().map(|(id, t)| (*id, t.as_str())).collect();
    let index = SuffixArrayIndex::build(&doc_refs);

    let config = PatternSearchConfig::default();
    let bitmap = index.search_doc_ids("electronics", &config);
    assert_eq!(bitmap.len(), 100, "All 100 docs contain 'electronics'");

    let bitmap2 = index.search_doc_ids("Document 7 ", &config);
    assert!(bitmap2.contains(7), "Bitmap should contain doc 7");
}

#[test]
fn test_save_load_roundtrip() {
    let docs: Vec<(u32, String)> = vec![
        (0, "hello world".to_string()),
        (1, "goodbye world".to_string()),
    ];
    let doc_refs: Vec<(u32, &str)> = docs.iter().map(|(id, t)| (*id, t.as_str())).collect();
    let index = SuffixArrayIndex::build(&doc_refs);

    let tmp = NamedTempFile::new().unwrap();
    index.save(tmp.path()).unwrap();

    let loaded = SuffixArrayIndex::load(tmp.path()).unwrap();

    let before = index.search_exact("world");
    let after = loaded.search_exact("world");
    assert_eq!(before.len(), after.len());
    for (a, b) in before.iter().zip(after.iter()) {
        assert_eq!(a.doc_id, b.doc_id);
        assert_eq!(a.offset, b.offset);
        assert_eq!(a.length, b.length);
    }
}

#[test]
fn test_large_corpus() {
    let docs: Vec<(u32, String)> = (0..10_000)
        .map(|i| (i as u32, format!("Short document number {}", i)))
        .collect();
    let doc_refs: Vec<(u32, &str)> = docs.iter().map(|(id, t)| (*id, t.as_str())).collect();
    let index = SuffixArrayIndex::build(&doc_refs);

    let matches = index.search_exact("document number 5000");
    assert!(
        matches.iter().any(|m| m.doc_id == 5000),
        "Should find doc 5000 in large corpus"
    );

    // "document" appears in every doc.
    let matches_all = index.search_exact("document");
    let unique_docs: std::collections::HashSet<u32> =
        matches_all.iter().map(|m| m.doc_id).collect();
    assert_eq!(unique_docs.len(), 10_000);
}

#[test]
fn test_case_sensitivity() {
    let docs: Vec<(u32, String)> = vec![
        (0, "Hello World".to_string()),
        (1, "hello world".to_string()),
    ];
    let doc_refs: Vec<(u32, &str)> = docs.iter().map(|(id, t)| (*id, t.as_str())).collect();
    let index = SuffixArrayIndex::build(&doc_refs);

    let upper = index.search_exact("Hello");
    let lower = index.search_exact("hello");

    assert_eq!(upper.len(), 1, "Exact uppercase match in doc 0 only");
    assert_eq!(upper[0].doc_id, 0);
    assert_eq!(lower.len(), 1, "Exact lowercase match in doc 1 only");
    assert_eq!(lower[0].doc_id, 1);
}

#[test]
fn test_empty_pattern() {
    let docs: Vec<(u32, String)> = vec![(0, "some text".to_string())];
    let doc_refs: Vec<(u32, &str)> = docs.iter().map(|(id, t)| (*id, t.as_str())).collect();
    let index = SuffixArrayIndex::build(&doc_refs);

    let matches = index.search_exact("");
    assert!(matches.is_empty(), "Empty pattern should return no results");

    let config = PatternSearchConfig::default();
    let matches_fuzzy = index.search_fuzzy("", &config);
    assert!(
        matches_fuzzy.is_empty(),
        "Empty pattern fuzzy should return no results"
    );
}
