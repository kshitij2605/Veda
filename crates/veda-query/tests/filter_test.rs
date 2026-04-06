use veda_query::filter::ast::{FilterExpr, FilterValue};
use veda_query::filter::evaluator::FilterContext;
use veda_query::filter::parser::FilterParser;
use veda_storage::inverted_index::InvertedIndex;
use veda_storage::range_index::RangeIndex;

// ══════════════════════════════════════════════════════════════════════════════
// Parser Tests
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_parse_eq_string() {
    let expr = FilterParser::parse("category = 'electronics'").unwrap();
    assert_eq!(
        expr,
        FilterExpr::Eq {
            field: "category".into(),
            value: FilterValue::String("electronics".into()),
        }
    );
}

#[test]
fn test_parse_gt_int() {
    let expr = FilterParser::parse("price > 100").unwrap();
    assert_eq!(
        expr,
        FilterExpr::Gt {
            field: "price".into(),
            value: FilterValue::Int(100),
        }
    );
}

#[test]
fn test_parse_gte_float() {
    let expr = FilterParser::parse("price >= 99.99").unwrap();
    assert_eq!(
        expr,
        FilterExpr::Gte {
            field: "price".into(),
            value: FilterValue::Float(99.99),
        }
    );
}

#[test]
fn test_parse_between() {
    let expr = FilterParser::parse("price BETWEEN 100 AND 500").unwrap();
    assert_eq!(
        expr,
        FilterExpr::Between {
            field: "price".into(),
            low: FilterValue::Int(100),
            high: FilterValue::Int(500),
        }
    );
}

#[test]
fn test_parse_in() {
    let expr = FilterParser::parse("category IN ('a', 'b', 'c')").unwrap();
    assert_eq!(
        expr,
        FilterExpr::In {
            field: "category".into(),
            values: vec![
                FilterValue::String("a".into()),
                FilterValue::String("b".into()),
                FilterValue::String("c".into()),
            ],
        }
    );
}

#[test]
fn test_parse_starts_with() {
    let expr = FilterParser::parse("title STARTS_WITH 'Red'").unwrap();
    assert_eq!(
        expr,
        FilterExpr::StartsWith {
            field: "title".into(),
            value: "Red".into(),
        }
    );
}

#[test]
fn test_parse_is_null() {
    let expr = FilterParser::parse("active IS NULL").unwrap();
    assert_eq!(expr, FilterExpr::IsNull { field: "active".into() });
}

#[test]
fn test_parse_is_not_null() {
    let expr = FilterParser::parse("active IS NOT NULL").unwrap();
    assert_eq!(expr, FilterExpr::IsNotNull { field: "active".into() });
}

#[test]
fn test_parse_and() {
    let expr = FilterParser::parse("category = 'electronics' AND price < 500").unwrap();
    assert_eq!(
        expr,
        FilterExpr::And(
            Box::new(FilterExpr::Eq {
                field: "category".into(),
                value: FilterValue::String("electronics".into()),
            }),
            Box::new(FilterExpr::Lt {
                field: "price".into(),
                value: FilterValue::Int(500),
            }),
        )
    );
}

#[test]
fn test_parse_or() {
    let expr = FilterParser::parse("a = 'x' OR b = 'y'").unwrap();
    assert_eq!(
        expr,
        FilterExpr::Or(
            Box::new(FilterExpr::Eq {
                field: "a".into(),
                value: FilterValue::String("x".into()),
            }),
            Box::new(FilterExpr::Eq {
                field: "b".into(),
                value: FilterValue::String("y".into()),
            }),
        )
    );
}

#[test]
fn test_parse_not() {
    let expr = FilterParser::parse("NOT category = 'books'").unwrap();
    assert_eq!(
        expr,
        FilterExpr::Not(Box::new(FilterExpr::Eq {
            field: "category".into(),
            value: FilterValue::String("books".into()),
        }))
    );
}

#[test]
fn test_parse_grouped_expr() {
    let expr = FilterParser::parse("(a = '1' OR b = '2') AND c > 3").unwrap();
    assert_eq!(
        expr,
        FilterExpr::And(
            Box::new(FilterExpr::Or(
                Box::new(FilterExpr::Eq {
                    field: "a".into(),
                    value: FilterValue::String("1".into()),
                }),
                Box::new(FilterExpr::Eq {
                    field: "b".into(),
                    value: FilterValue::String("2".into()),
                }),
            )),
            Box::new(FilterExpr::Gt {
                field: "c".into(),
                value: FilterValue::Int(3),
            }),
        )
    );
}

#[test]
fn test_parse_case_insensitive_keywords() {
    // "and" and "or" in lowercase should work.
    let expr = FilterParser::parse("category = 'X' and price > 100").unwrap();
    assert_eq!(
        expr,
        FilterExpr::And(
            Box::new(FilterExpr::Eq {
                field: "category".into(),
                value: FilterValue::String("X".into()),
            }),
            Box::new(FilterExpr::Gt {
                field: "price".into(),
                value: FilterValue::Int(100),
            }),
        )
    );
}

#[test]
fn test_parse_ne() {
    let expr = FilterParser::parse("status != 'deleted'").unwrap();
    assert_eq!(
        expr,
        FilterExpr::Ne {
            field: "status".into(),
            value: FilterValue::String("deleted".into()),
        }
    );
}

#[test]
fn test_parse_lt() {
    let expr = FilterParser::parse("age < 30").unwrap();
    assert_eq!(
        expr,
        FilterExpr::Lt {
            field: "age".into(),
            value: FilterValue::Int(30),
        }
    );
}

#[test]
fn test_parse_lte() {
    let expr = FilterParser::parse("age <= 65").unwrap();
    assert_eq!(
        expr,
        FilterExpr::Lte {
            field: "age".into(),
            value: FilterValue::Int(65),
        }
    );
}

#[test]
fn test_parse_bool_value() {
    let expr = FilterParser::parse("active = true").unwrap();
    assert_eq!(
        expr,
        FilterExpr::Eq {
            field: "active".into(),
            value: FilterValue::Bool(true),
        }
    );
}

#[test]
fn test_parse_contains() {
    let expr = FilterParser::parse("title CONTAINS 'rust'").unwrap();
    assert_eq!(
        expr,
        FilterExpr::Contains {
            field: "title".into(),
            value: "rust".into(),
        }
    );
}

#[test]
fn test_parse_dotted_field() {
    let expr = FilterParser::parse("metadata.color = 'red'").unwrap();
    assert_eq!(
        expr,
        FilterExpr::Eq {
            field: "metadata.color".into(),
            value: FilterValue::String("red".into()),
        }
    );
}

#[test]
fn test_parse_error_unterminated_string() {
    let result = FilterParser::parse("category = 'oops");
    assert!(result.is_err());
}

#[test]
fn test_parse_error_unexpected_token() {
    let result = FilterParser::parse("= 'value'");
    assert!(result.is_err());
}

// ══════════════════════════════════════════════════════════════════════════════
// Evaluator Tests
// ══════════════════════════════════════════════════════════════════════════════

/// Deterministic pseudo-random f64 in [lo, hi).
fn pseudo_random_f64(seed: u32, lo: f64, hi: f64) -> f64 {
    let state = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
    let frac = ((state >> 16) & 0x7FFF) as f64 / 32768.0;
    lo + frac * (hi - lo)
}

/// Build test indexes: 100 docs with categories and prices.
fn build_test_indexes() -> (InvertedIndex, RangeIndex, Vec<&'static str>, Vec<f64>) {
    let categories = ["electronics", "books", "clothing"];
    let mut inv = InvertedIndex::new();
    let mut rng = RangeIndex::new();
    let mut cats = Vec::new();
    let mut prices = Vec::new();

    for i in 0..100u32 {
        let cat = categories[(i % 3) as usize];
        inv.insert(i, cat);
        cats.push(cat);

        let price = pseudo_random_f64(i, 10.0, 500.0);
        rng.insert(i, price);
        prices.push(price);
    }
    rng.build();
    (inv, rng, cats, prices)
}

#[test]
fn test_eval_eq_string() {
    let (inv, rng, _cats, _prices) = build_test_indexes();
    let mut ctx = FilterContext::new(100);
    ctx.add_inverted("category", &inv);
    ctx.add_range("price", &rng);

    let expr = FilterParser::parse("category = 'electronics'").unwrap();
    let result = ctx.evaluate(&expr).unwrap();

    // electronics: 0, 3, 6, ..., 99 => 34 docs
    assert_eq!(result.len(), 34);
    assert!(result.contains(0));
    assert!(result.contains(3));
    assert!(result.contains(99));
    assert!(!result.contains(1));
}

#[test]
fn test_eval_between() {
    let (inv, rng, _cats, prices) = build_test_indexes();
    let mut ctx = FilterContext::new(100);
    ctx.add_inverted("category", &inv);
    ctx.add_range("price", &rng);

    let expr = FilterParser::parse("price BETWEEN 100 AND 200").unwrap();
    let result = ctx.evaluate(&expr).unwrap();

    // Verify every returned doc ID is in range.
    for doc_id in result.iter() {
        let price = prices[doc_id as usize];
        assert!(
            (100.0..=200.0).contains(&price),
            "doc {doc_id} price {price} should be in [100, 200]"
        );
    }
    // Verify no qualifying docs were missed.
    for (i, &price) in prices.iter().enumerate() {
        if (100.0..=200.0).contains(&price) {
            assert!(
                result.contains(i as u32),
                "doc {i} price {price} should be in result"
            );
        }
    }
}

#[test]
fn test_eval_and() {
    let (inv, rng, _cats, prices) = build_test_indexes();
    let mut ctx = FilterContext::new(100);
    ctx.add_inverted("category", &inv);
    ctx.add_range("price", &rng);

    let expr = FilterParser::parse("category = 'electronics' AND price < 200").unwrap();
    let result = ctx.evaluate(&expr).unwrap();

    for doc_id in result.iter() {
        assert!(inv.contains("electronics", doc_id));
        assert!(
            prices[doc_id as usize] < 200.0,
            "doc {doc_id} price {} should be < 200",
            prices[doc_id as usize]
        );
    }
    // Ensure we didn't miss any qualifying docs.
    for (i, &price) in prices.iter().enumerate() {
        let i = i as u32;
        if inv.contains("electronics", i) && price < 200.0 {
            assert!(result.contains(i), "doc {i} should be in result");
        }
    }
}

#[test]
fn test_eval_or() {
    let (inv, rng, _cats, _prices) = build_test_indexes();
    let mut ctx = FilterContext::new(100);
    ctx.add_inverted("category", &inv);
    ctx.add_range("price", &rng);

    let expr = FilterParser::parse("category = 'books' OR category = 'clothing'").unwrap();
    let result = ctx.evaluate(&expr).unwrap();

    // books: 33 docs, clothing: 33 docs, no overlap => 66 docs
    assert_eq!(result.len(), 66);
    assert!(result.contains(1)); // books
    assert!(result.contains(2)); // clothing
    assert!(!result.contains(0)); // electronics
}

#[test]
fn test_eval_not() {
    let (inv, rng, _cats, _prices) = build_test_indexes();
    let mut ctx = FilterContext::new(100);
    ctx.add_inverted("category", &inv);
    ctx.add_range("price", &rng);

    let expr = FilterParser::parse("NOT category = 'electronics'").unwrap();
    let result = ctx.evaluate(&expr).unwrap();

    // 100 total - 34 electronics = 66
    assert_eq!(result.len(), 66);
    assert!(!result.contains(0));
    assert!(result.contains(1)); // books
    assert!(result.contains(2)); // clothing
}

#[test]
fn test_eval_in() {
    let (inv, rng, _cats, _prices) = build_test_indexes();
    let mut ctx = FilterContext::new(100);
    ctx.add_inverted("category", &inv);
    ctx.add_range("price", &rng);

    let expr = FilterParser::parse("category IN ('books', 'clothing')").unwrap();
    let result = ctx.evaluate(&expr).unwrap();

    assert_eq!(result.len(), 66);
    assert!(!result.contains(0)); // electronics
    assert!(result.contains(1)); // books
    assert!(result.contains(2)); // clothing
}

#[test]
fn test_eval_starts_with() {
    let mut inv = InvertedIndex::new();
    inv.insert(0, "electronics");
    inv.insert(1, "electrical");
    inv.insert(2, "books");

    let mut ctx = FilterContext::new(3);
    ctx.add_inverted("category", &inv);

    let expr = FilterParser::parse("category STARTS_WITH 'elec'").unwrap();
    let result = ctx.evaluate(&expr).unwrap();

    assert_eq!(result.len(), 2);
    assert!(result.contains(0));
    assert!(result.contains(1));
    assert!(!result.contains(2));
}

#[test]
fn test_eval_contains() {
    let mut inv = InvertedIndex::new();
    inv.insert(0, "electronics");
    inv.insert(1, "electric_car");
    inv.insert(2, "books");

    let mut ctx = FilterContext::new(3);
    ctx.add_inverted("category", &inv);

    // "electric_car" contains "lectric" (l-e-c-t-r-i-c), but "electronics"
    // does not (it has "lectr-o-n-ics").  Use "lectr" to match both.
    let expr = FilterParser::parse("category CONTAINS 'lectr'").unwrap();
    let result = ctx.evaluate(&expr).unwrap();

    assert_eq!(result.len(), 2);
    assert!(result.contains(0)); // electronics contains "lectr"
    assert!(result.contains(1)); // electric_car contains "lectr"
    assert!(!result.contains(2)); // books does not
}

#[test]
fn test_eval_ne() {
    let (inv, rng, _cats, _prices) = build_test_indexes();
    let mut ctx = FilterContext::new(100);
    ctx.add_inverted("category", &inv);
    ctx.add_range("price", &rng);

    let expr = FilterParser::parse("category != 'electronics'").unwrap();
    let result = ctx.evaluate(&expr).unwrap();

    // 100 - 34 = 66
    assert_eq!(result.len(), 66);
    assert!(!result.contains(0));
    assert!(result.contains(1));
}

#[test]
fn test_eval_gt_lt() {
    let (inv, rng, _cats, prices) = build_test_indexes();
    let mut ctx = FilterContext::new(100);
    ctx.add_inverted("category", &inv);
    ctx.add_range("price", &rng);

    let expr = FilterParser::parse("price > 400").unwrap();
    let result = ctx.evaluate(&expr).unwrap();

    for doc_id in result.iter() {
        assert!(prices[doc_id as usize] > 400.0);
    }
    for (i, &price) in prices.iter().enumerate() {
        if price > 400.0 {
            assert!(result.contains(i as u32));
        }
    }
}

#[test]
fn test_eval_is_null_and_is_not_null() {
    // Build an inverted index that only covers docs 0..50 out of 100.
    let mut inv = InvertedIndex::new();
    for i in 0..50u32 {
        inv.insert(i, "present");
    }

    let mut ctx = FilterContext::new(100);
    ctx.add_inverted("optional_field", &inv);

    let expr_null = FilterParser::parse("optional_field IS NULL").unwrap();
    let result_null = ctx.evaluate(&expr_null).unwrap();
    assert_eq!(result_null.len(), 50); // docs 50..100
    assert!(result_null.contains(50));
    assert!(result_null.contains(99));
    assert!(!result_null.contains(0));

    let expr_not_null = FilterParser::parse("optional_field IS NOT NULL").unwrap();
    let result_not_null = ctx.evaluate(&expr_not_null).unwrap();
    assert_eq!(result_not_null.len(), 50); // docs 0..50
    assert!(result_not_null.contains(0));
    assert!(result_not_null.contains(49));
    assert!(!result_not_null.contains(50));
}

#[test]
fn test_eval_complex_filter() {
    let (inv, rng, _cats, prices) = build_test_indexes();
    let mut ctx = FilterContext::new(100);
    ctx.add_inverted("category", &inv);
    ctx.add_range("price", &rng);

    // (category = 'electronics' OR category = 'books') AND price BETWEEN 100 AND 300
    let expr = FilterParser::parse(
        "(category = 'electronics' OR category = 'books') AND price BETWEEN 100 AND 300",
    )
    .unwrap();
    let result = ctx.evaluate(&expr).unwrap();

    for doc_id in result.iter() {
        let is_elec = inv.contains("electronics", doc_id);
        let is_books = inv.contains("books", doc_id);
        assert!(is_elec || is_books);
        let price = prices[doc_id as usize];
        assert!(
            (100.0..=300.0).contains(&price),
            "doc {doc_id} price {price} not in [100, 300]"
        );
    }
}
