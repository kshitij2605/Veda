//! GAP-1 + UX-2 (2026-05-21 E2E campaign): LIKE operator + double-
//! quoted string literals in the filter DSL.
//!
//! Before:
//!   filter: "content LIKE '%foo%'"   → 500 "expected operator, got Ident(LIKE)"
//!   filter: "x = \"foo\""             → 500 "unexpected character: '\"'"
//!
//! After:
//!   - LIKE '%X%' translates to Contains{X}; 'X%' to StartsWith{X};
//!     'X' (no wildcards) to Eq{X}; '%X' (ends-with) returns a clear
//!     "not yet supported" error so the user knows to ask.
//!   - Double-quoted and single-quoted string literals are both
//!     accepted by the lexer.

use veda_query::filter::ast::{FilterExpr, FilterValue};
use veda_query::filter::parser::FilterParser;

#[test]
fn double_quoted_string_literal_parses() {
    let expr = FilterParser::parse(r#"category = "electronics""#).unwrap();
    assert_eq!(
        expr,
        FilterExpr::Eq {
            field: "category".into(),
            value: FilterValue::String("electronics".into())
        }
    );
}

#[test]
fn single_quoted_still_works() {
    let expr = FilterParser::parse("category = 'electronics'").unwrap();
    assert_eq!(
        expr,
        FilterExpr::Eq {
            field: "category".into(),
            value: FilterValue::String("electronics".into())
        }
    );
}

#[test]
fn double_quoted_with_japanese_works() {
    let expr = FilterParser::parse(r#"parent_doc_id = "提案資料""#).unwrap();
    assert_eq!(
        expr,
        FilterExpr::Eq {
            field: "parent_doc_id".into(),
            value: FilterValue::String("提案資料".into()),
        }
    );
}

#[test]
fn like_percent_X_percent_translates_to_contains() {
    let expr = FilterParser::parse("content LIKE '%バックアップ%'").unwrap();
    assert_eq!(
        expr,
        FilterExpr::Contains {
            field: "content".into(),
            value: "バックアップ".into()
        }
    );
}

#[test]
fn like_X_percent_translates_to_starts_with() {
    let expr = FilterParser::parse("title LIKE 'Adobe%'").unwrap();
    assert_eq!(
        expr,
        FilterExpr::StartsWith {
            field: "title".into(),
            value: "Adobe".into()
        }
    );
}

#[test]
fn like_no_wildcard_translates_to_eq() {
    let expr = FilterParser::parse("status LIKE 'active'").unwrap();
    assert_eq!(
        expr,
        FilterExpr::Eq {
            field: "status".into(),
            value: FilterValue::String("active".into())
        }
    );
}

#[test]
fn like_ends_with_returns_clear_error() {
    let err = FilterParser::parse("title LIKE '%pdf'").unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("ends-with") && msg.contains("not yet supported"),
        "should suggest CONTAINS / leading%: {msg}"
    );
}

#[test]
fn like_pattern_with_middle_percent_rejected() {
    let err = FilterParser::parse("title LIKE '%foo%bar%'").unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("middle"),
        "should mention middle-percent: {msg}"
    );
}

#[test]
fn ilike_accepted_as_alias_for_like() {
    let expr = FilterParser::parse("content ILIKE '%foo%'").unwrap();
    assert_eq!(
        expr,
        FilterExpr::Contains {
            field: "content".into(),
            value: "foo".into()
        }
    );
}

#[test]
fn double_quoted_and_like_compose() {
    let expr = FilterParser::parse(r#"content LIKE "%foo%" AND status = "active""#).unwrap();
    match expr {
        FilterExpr::And(_, _) => {}
        other => panic!("expected AND, got {other:?}"),
    }
}
