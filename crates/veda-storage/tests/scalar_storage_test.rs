use tempfile::TempDir;
use veda_storage::bitmap_index::BitmapIndex;
use veda_storage::inverted_index::InvertedIndex;
use veda_storage::range_index::RangeIndex;

// ══════════════════════════════════════════════════════════════════════════════
// Inverted Index Tests
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_inverted_index_search_exact() {
    let mut idx = InvertedIndex::new();
    let categories = ["electronics", "books", "clothing"];

    for i in 0..100u32 {
        let cat = categories[(i % 3) as usize];
        idx.insert(i, cat);
    }

    // electronics: 0, 3, 6, 9, ..., 99
    let electronics = idx.search_exact("electronics").unwrap();
    assert_eq!(electronics.len(), 34); // 0,3,6,...,99 => ceil(100/3) = 34
    assert!(electronics.contains(0));
    assert!(electronics.contains(3));
    assert!(electronics.contains(99));
    assert!(!electronics.contains(1));
    assert!(!electronics.contains(2));

    // books: 1, 4, 7, ...
    let books = idx.search_exact("books").unwrap();
    assert_eq!(books.len(), 33);
    assert!(books.contains(1));
    assert!(books.contains(4));

    // clothing: 2, 5, 8, ...
    let clothing = idx.search_exact("clothing").unwrap();
    assert_eq!(clothing.len(), 33);
    assert!(clothing.contains(2));
    assert!(clothing.contains(5));

    // Non-existent term.
    assert!(idx.search_exact("furniture").is_none());
}

#[test]
fn test_inverted_index_search_prefix() {
    let mut idx = InvertedIndex::new();
    idx.insert(0, "electronics");
    idx.insert(1, "electrical");
    idx.insert(2, "books");
    idx.insert(3, "clothing");
    idx.insert(4, "electric");

    let result = idx.search_prefix("elect");
    assert_eq!(result.len(), 3); // electronics, electrical, electric
    assert!(result.contains(0));
    assert!(result.contains(1));
    assert!(result.contains(4));
    assert!(!result.contains(2));
    assert!(!result.contains(3));

    // Full-word prefix.
    let result = idx.search_prefix("electronics");
    assert_eq!(result.len(), 1);
    assert!(result.contains(0));

    // No match prefix.
    let result = idx.search_prefix("xyz");
    assert!(result.is_empty());
}

#[test]
fn test_inverted_index_search_in() {
    let mut idx = InvertedIndex::new();
    let categories = ["electronics", "books", "clothing"];

    for i in 0..100u32 {
        let cat = categories[(i % 3) as usize];
        idx.insert(i, cat);
    }

    let result = idx.search_in(&["books", "clothing"]);
    // books (33) + clothing (33) = 66
    assert_eq!(result.len(), 66);
    assert!(result.contains(1)); // books
    assert!(result.contains(2)); // clothing
    assert!(!result.contains(0)); // electronics

    // Search with non-existent term mixed in.
    let result = idx.search_in(&["books", "furniture"]);
    assert_eq!(result.len(), 33);
}

#[test]
fn test_inverted_index_contains() {
    let mut idx = InvertedIndex::new();
    idx.insert(5, "electronics");
    idx.insert(10, "books");

    assert!(idx.contains("electronics", 5));
    assert!(!idx.contains("electronics", 10));
    assert!(idx.contains("books", 10));
    assert!(!idx.contains("furniture", 5));
}

#[test]
fn test_inverted_index_all_doc_ids() {
    let mut idx = InvertedIndex::new();
    for i in 0..50u32 {
        idx.insert(i, "a");
    }
    for i in 25..75u32 {
        idx.insert(i, "b");
    }

    let all = idx.all_doc_ids();
    assert_eq!(all.len(), 75); // 0..75
}

#[test]
fn test_inverted_index_term_and_doc_count() {
    let mut idx = InvertedIndex::new();
    let categories = ["electronics", "books", "clothing"];

    for i in 0..100u32 {
        idx.insert(i, categories[(i % 3) as usize]);
    }

    assert_eq!(idx.term_count(), 3);
    assert_eq!(idx.doc_count(), 100);
}

#[test]
fn test_inverted_index_save_load_roundtrip() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("inverted.json");

    let mut idx = InvertedIndex::new();
    let categories = ["electronics", "books", "clothing"];
    for i in 0..100u32 {
        idx.insert(i, categories[(i % 3) as usize]);
    }

    idx.save(&path).unwrap();
    let loaded = InvertedIndex::load(&path).unwrap();

    assert_eq!(loaded.term_count(), 3);
    assert_eq!(loaded.doc_count(), 100);

    // Verify exact matches are preserved.
    let electronics = loaded.search_exact("electronics").unwrap();
    assert_eq!(electronics.len(), 34);
    assert!(electronics.contains(0));
    assert!(electronics.contains(99));

    let books = loaded.search_exact("books").unwrap();
    assert_eq!(books.len(), 33);
    assert!(books.contains(1));

    // Verify prefix search works after load.
    let prefix_result = loaded.search_prefix("elect");
    assert_eq!(prefix_result.len(), 34);
}

// ══════════════════════════════════════════════════════════════════════════════
// Range Index Tests
// ══════════════════════════════════════════════════════════════════════════════

/// Deterministic pseudo-random f64 in [lo, hi).
fn pseudo_random_f64(seed: u32, lo: f64, hi: f64) -> f64 {
    let state = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
    let frac = ((state >> 16) & 0x7FFF) as f64 / 32768.0;
    lo + frac * (hi - lo)
}

#[test]
fn test_range_index_search_range() {
    let mut idx = RangeIndex::new();
    let mut prices = Vec::new();

    for i in 0..100u32 {
        let price = pseudo_random_f64(i, 10.0, 500.0);
        prices.push(price);
        idx.insert(i, price);
    }
    idx.build();

    let result = idx.search_range(100.0, 200.0);
    // Verify every returned doc_id actually has a price in [100, 200].
    for doc_id in result.iter() {
        let price = prices[doc_id as usize];
        assert!(
            (100.0..=200.0).contains(&price),
            "doc {doc_id} price {price} not in [100, 200]"
        );
    }
    // Verify no docs in range were missed.
    for (i, &price) in prices.iter().enumerate() {
        if (100.0..=200.0).contains(&price) {
            assert!(
                result.contains(i as u32),
                "doc {i} price {price} should be in range result"
            );
        }
    }
}

#[test]
fn test_range_index_search_gt() {
    let mut idx = RangeIndex::new();
    let mut prices = Vec::new();

    for i in 0..100u32 {
        let price = pseudo_random_f64(i, 10.0, 500.0);
        prices.push(price);
        idx.insert(i, price);
    }
    idx.build();

    let result = idx.search_gt(400.0);
    for doc_id in result.iter() {
        assert!(
            prices[doc_id as usize] > 400.0,
            "doc {doc_id} should have price > 400"
        );
    }
    for (i, &price) in prices.iter().enumerate() {
        if price > 400.0 {
            assert!(result.contains(i as u32), "doc {i} price {price} missed");
        }
    }
}

#[test]
fn test_range_index_search_lt() {
    let mut idx = RangeIndex::new();
    let mut prices = Vec::new();

    for i in 0..100u32 {
        let price = pseudo_random_f64(i, 10.0, 500.0);
        prices.push(price);
        idx.insert(i, price);
    }
    idx.build();

    let result = idx.search_lt(50.0);
    for doc_id in result.iter() {
        assert!(
            prices[doc_id as usize] < 50.0,
            "doc {doc_id} should have price < 50"
        );
    }
    for (i, &price) in prices.iter().enumerate() {
        if price < 50.0 {
            assert!(result.contains(i as u32), "doc {i} price {price} missed");
        }
    }
}

#[test]
fn test_range_index_search_gte_lte() {
    let mut idx = RangeIndex::new();
    // Insert known values for precise testing.
    idx.insert(0, 10.0);
    idx.insert(1, 20.0);
    idx.insert(2, 30.0);
    idx.insert(3, 40.0);
    idx.insert(4, 50.0);
    idx.build();

    let gte = idx.search_gte(30.0);
    assert_eq!(gte.len(), 3); // 30, 40, 50
    assert!(gte.contains(2));
    assert!(gte.contains(3));
    assert!(gte.contains(4));

    let lte = idx.search_lte(30.0);
    assert_eq!(lte.len(), 3); // 10, 20, 30
    assert!(lte.contains(0));
    assert!(lte.contains(1));
    assert!(lte.contains(2));
}

#[test]
fn test_range_index_search_eq() {
    let mut idx = RangeIndex::new();
    idx.insert(0, 42.0);
    idx.insert(1, 42.0);
    idx.insert(2, 43.0);
    idx.insert(3, 41.0);
    idx.build();

    let result = idx.search_eq(42.0);
    assert_eq!(result.len(), 2);
    assert!(result.contains(0));
    assert!(result.contains(1));
    assert!(!result.contains(2));
    assert!(!result.contains(3));

    // Non-existent value.
    let result = idx.search_eq(99.0);
    assert!(result.is_empty());
}

#[test]
fn test_range_index_doc_count() {
    let mut idx = RangeIndex::new();
    for i in 0..50u32 {
        idx.insert(i, i as f64);
    }
    idx.build();
    assert_eq!(idx.doc_count(), 50);
}

#[test]
fn test_range_index_save_load_roundtrip() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("range.json");

    let mut idx = RangeIndex::new();
    let mut prices = Vec::new();
    for i in 0..100u32 {
        let price = pseudo_random_f64(i, 10.0, 500.0);
        prices.push(price);
        idx.insert(i, price);
    }
    idx.build();

    idx.save(&path).unwrap();
    let loaded = RangeIndex::load(&path).unwrap();

    assert_eq!(loaded.doc_count(), 100);

    // Verify range query works after load.
    let original_range = idx.search_range(100.0, 200.0);
    let loaded_range = loaded.search_range(100.0, 200.0);
    assert_eq!(original_range.len(), loaded_range.len());
    for doc_id in original_range.iter() {
        assert!(loaded_range.contains(doc_id));
    }

    // Verify gt query.
    let original_gt = idx.search_gt(400.0);
    let loaded_gt = loaded.search_gt(400.0);
    assert_eq!(original_gt.len(), loaded_gt.len());
}

// ══════════════════════════════════════════════════════════════════════════════
// Bitmap Index Tests
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_bitmap_index_search_eq() {
    let mut idx = BitmapIndex::new();

    for i in 0..100u32 {
        let val = if i % 2 == 0 { "true" } else { "false" };
        idx.insert(i, val);
    }

    let true_docs = idx.search_eq("true").unwrap();
    assert_eq!(true_docs.len(), 50);
    assert!(true_docs.contains(0));
    assert!(true_docs.contains(2));
    assert!(true_docs.contains(98));
    assert!(!true_docs.contains(1));

    let false_docs = idx.search_eq("false").unwrap();
    assert_eq!(false_docs.len(), 50);
    assert!(false_docs.contains(1));
    assert!(false_docs.contains(99));

    assert!(idx.search_eq("maybe").is_none());
}

#[test]
fn test_bitmap_index_search_not() {
    let mut idx = BitmapIndex::new();

    for i in 0..100u32 {
        let val = if i % 2 == 0 { "true" } else { "false" };
        idx.insert(i, val);
    }

    let not_true = idx.search_not("true");
    assert_eq!(not_true.len(), 50);
    assert!(not_true.contains(1)); // "false" doc
    assert!(!not_true.contains(0)); // "true" doc

    // NOT of a non-existent value should return all docs.
    let not_maybe = idx.search_not("maybe");
    assert_eq!(not_maybe.len(), 100);
}

#[test]
fn test_bitmap_index_search_in() {
    let mut idx = BitmapIndex::new();
    let colors = ["red", "green", "blue"];

    for i in 0..90u32 {
        idx.insert(i, colors[(i % 3) as usize]);
    }

    let result = idx.search_in(&["red", "blue"]);
    assert_eq!(result.len(), 60); // 30 red + 30 blue
    assert!(result.contains(0)); // red
    assert!(result.contains(2)); // blue
    assert!(!result.contains(1)); // green
}

#[test]
fn test_bitmap_index_distinct_values() {
    let mut idx = BitmapIndex::new();
    idx.insert(0, "red");
    idx.insert(1, "green");
    idx.insert(2, "blue");
    idx.insert(3, "red");

    let values = idx.distinct_values();
    assert_eq!(values.len(), 3);
    // distinct_values returns sorted
    assert_eq!(values, vec!["blue", "green", "red"]);
}

#[test]
fn test_bitmap_index_doc_count() {
    let mut idx = BitmapIndex::new();
    let colors = ["red", "green", "blue"];
    for i in 0..90u32 {
        idx.insert(i, colors[(i % 3) as usize]);
    }
    assert_eq!(idx.doc_count(), 90);
}

#[test]
fn test_bitmap_index_save_load_roundtrip() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("bitmap.json");

    let mut idx = BitmapIndex::new();
    let colors = ["red", "green", "blue"];
    for i in 0..90u32 {
        idx.insert(i, colors[(i % 3) as usize]);
    }

    idx.save(&path).unwrap();
    let loaded = BitmapIndex::load(&path).unwrap();

    assert_eq!(loaded.doc_count(), 90);
    assert_eq!(loaded.distinct_values().len(), 3);

    let red = loaded.search_eq("red").unwrap();
    assert_eq!(red.len(), 30);
    assert!(red.contains(0));
    assert!(red.contains(3));

    let not_green = loaded.search_not("green");
    assert_eq!(not_green.len(), 60);

    let result = loaded.search_in(&["red", "blue"]);
    assert_eq!(result.len(), 60);
}

// ══════════════════════════════════════════════════════════════════════════════
// Cross-Index Tests
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_cross_index_category_and_price_range() {
    // Simulate: category="electronics" AND price BETWEEN 100 AND 200
    let mut inverted = InvertedIndex::new();
    let mut range = RangeIndex::new();

    let categories = ["electronics", "books", "clothing"];
    for i in 0..100u32 {
        let cat = categories[(i % 3) as usize];
        inverted.insert(i, cat);

        let price = pseudo_random_f64(i, 10.0, 500.0);
        range.insert(i, price);
    }
    range.build();

    let electronics_bitmap = inverted.search_exact("electronics").unwrap().clone();
    let price_bitmap = range.search_range(100.0, 200.0);

    // AND: intersection
    let combined = electronics_bitmap & &price_bitmap;

    // Verify every doc in the result is actually electronics with price in [100, 200].
    for doc_id in combined.iter() {
        assert!(inverted.contains("electronics", doc_id));
        let price = pseudo_random_f64(doc_id, 10.0, 500.0);
        assert!(
            (100.0..=200.0).contains(&price),
            "doc {doc_id} price {price} not in range"
        );
    }
    // Ensure result is non-empty (probabilistically very likely with 100 docs).
    assert!(!combined.is_empty(), "expected some matching documents");
}

#[test]
fn test_cross_index_category_or_color() {
    // Simulate: category="electronics" OR color="red"
    let mut category_idx = InvertedIndex::new();
    let mut color_idx = BitmapIndex::new();

    let categories = ["electronics", "books", "clothing"];
    let colors = ["red", "green", "blue"];

    for i in 0..90u32 {
        category_idx.insert(i, categories[(i % 3) as usize]);
        color_idx.insert(i, colors[(i / 30) as usize]);
    }

    let electronics = category_idx.search_exact("electronics").unwrap().clone();
    let red = color_idx.search_eq("red").unwrap().clone();

    // OR: union
    let combined = electronics | red;

    // electronics: 0,3,6,...,87 (30 docs), red: 0..30 (30 docs)
    // Overlap: electronics in 0..30 = {0,3,6,...,27} = 10 docs
    // Union = 30 + 30 - 10 = 50
    assert_eq!(combined.len(), 50);
}

#[test]
fn test_cross_index_not_with_range() {
    // Simulate: color != "green" AND price > 300
    let mut color_idx = BitmapIndex::new();
    let mut range_idx = RangeIndex::new();

    let colors = ["red", "green", "blue"];
    for i in 0..90u32 {
        color_idx.insert(i, colors[(i % 3) as usize]);
        let price = pseudo_random_f64(i, 10.0, 500.0);
        range_idx.insert(i, price);
    }
    range_idx.build();

    let not_green = color_idx.search_not("green");
    let expensive = range_idx.search_gt(300.0);

    let combined = not_green & expensive;

    // Verify every doc is not green and has price > 300.
    for doc_id in combined.iter() {
        assert!(!color_idx.search_eq("green").unwrap().contains(doc_id));
        let price = pseudo_random_f64(doc_id, 10.0, 500.0);
        assert!(price > 300.0, "doc {doc_id} price {price} should be > 300");
    }
}
