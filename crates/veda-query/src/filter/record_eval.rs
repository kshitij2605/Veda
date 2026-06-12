use veda_core::record::{FieldValue, Record};

use super::ast::{FilterExpr, FilterValue};

/// Evaluate a filter expression against a single record.
///
/// Returns `true` if the record satisfies the filter, `false` otherwise.
/// This is used for the growing segment where we don't have pre-built scalar
/// indexes — instead we scan each record and evaluate the filter directly.
pub fn evaluate_record(expr: &FilterExpr, record: &Record) -> bool {
    match expr {
        FilterExpr::Eq { field, value } => match_eq(record, field, value),
        FilterExpr::Ne { field, value } => !match_eq(record, field, value),
        FilterExpr::Gt { field, value } => match_cmp(record, field, value, CmpOp::Gt),
        FilterExpr::Gte { field, value } => match_cmp(record, field, value, CmpOp::Gte),
        FilterExpr::Lt { field, value } => match_cmp(record, field, value, CmpOp::Lt),
        FilterExpr::Lte { field, value } => match_cmp(record, field, value, CmpOp::Lte),
        FilterExpr::Between { field, low, high } => match_between(record, field, low, high),
        FilterExpr::In { field, values } => values.iter().any(|v| match_eq(record, field, v)),
        FilterExpr::Contains { field, value } => match_contains(record, field, value),
        FilterExpr::StartsWith { field, value } => match_starts_with(record, field, value),
        FilterExpr::IsNull { field } => record.get(field).is_none(),
        FilterExpr::IsNotNull { field } => record.get(field).is_some(),
        FilterExpr::And(left, right) => {
            evaluate_record(left, record) && evaluate_record(right, record)
        }
        FilterExpr::Or(left, right) => {
            evaluate_record(left, record) || evaluate_record(right, record)
        }
        FilterExpr::Not(inner) => !evaluate_record(inner, record),
    }
}

// ── Internal helpers ──────────────────────────────────────────────────────

enum CmpOp {
    Gt,
    Gte,
    Lt,
    Lte,
}

/// Resolve a (possibly dotted) field path to a `FieldValue`.
///
/// If `field` contains no `.`, this is equivalent to `record.get(field)`.
/// If `field` is dotted (e.g. `"meta.user.tier"`), the head (`"meta"`) is
/// looked up in the record; if that field is `FieldValue::Json`, the tail
/// segments are traversed via [`serde_json::Value::pointer`].  The leaf JSON
/// value is then coerced into a `FieldValue` (String, Int64, Float64, or Bool).
/// Returns `None` if the path doesn't exist or the head is not a Json field.
fn resolve_field<'a>(record: &'a Record, field: &str) -> Option<std::borrow::Cow<'a, FieldValue>> {
    match field.find('.') {
        None => record.get(field).map(std::borrow::Cow::Borrowed),
        Some(dot_pos) => {
            let head = &field[..dot_pos];
            let tail = &field[dot_pos + 1..];
            let fv = record.get(head)?;
            let FieldValue::Json(json_val) = fv else {
                return None;
            };
            // Build a JSON Pointer from the tail: "user.tier" → "/user/tier"
            let pointer: String = tail
                .split('.')
                .map(|seg| format!("/{seg}"))
                .collect();
            let leaf = json_val.pointer(&pointer)?;
            // Coerce the JSON leaf to a FieldValue.
            let coerced = match leaf {
                serde_json::Value::String(s) => FieldValue::String(s.clone()),
                serde_json::Value::Number(n) => {
                    if let Some(i) = n.as_i64() {
                        FieldValue::Int64(i)
                    } else if let Some(f) = n.as_f64() {
                        FieldValue::Float64(f)
                    } else {
                        return None;
                    }
                }
                serde_json::Value::Bool(b) => FieldValue::Bool(*b),
                _ => return None,
            };
            Some(std::borrow::Cow::Owned(coerced))
        }
    }
}

/// Check equality between a record's field value and a filter value.
///
/// For Array field values, returns true if any element equals the filter value
/// (membership semantics — `tags = 'foo'` matches a record whose `tags` array
/// contains `"foo"`).
fn match_eq(record: &Record, field: &str, value: &FilterValue) -> bool {
    let Some(fv) = resolve_field(record, field) else {
        return false;
    };
    if let FieldValue::Array(elems) = fv.as_ref() {
        return elems.iter().any(|elem| match_field_value(elem, value));
    }
    match_field_value(fv.as_ref(), value)
}

fn match_field_value(fv: &FieldValue, value: &FilterValue) -> bool {
    match (fv, value) {
        (FieldValue::String(s), FilterValue::String(v)) => s == v,
        (FieldValue::Int32(n), FilterValue::Int(v)) => *n as i64 == *v,
        (FieldValue::Int64(n), FilterValue::Int(v)) => *n == *v,
        (FieldValue::Float32(n), FilterValue::Float(v)) => (*n as f64 - *v).abs() < f64::EPSILON,
        (FieldValue::Float64(n), FilterValue::Float(v)) => (*n - *v).abs() < f64::EPSILON,
        (FieldValue::Float32(n), FilterValue::Int(v)) => {
            (*n as f64 - *v as f64).abs() < f64::EPSILON
        }
        (FieldValue::Float64(n), FilterValue::Int(v)) => (*n - *v as f64).abs() < f64::EPSILON,
        (FieldValue::Int32(n), FilterValue::Float(v)) => {
            (*n as f64 - *v).abs() < f64::EPSILON
        }
        (FieldValue::Int64(n), FilterValue::Float(v)) => {
            (*n as f64 - *v).abs() < f64::EPSILON
        }
        (FieldValue::Bool(b), FilterValue::Bool(v)) => *b == *v,
        _ => false,
    }
}

/// Extract an f64 from a record's field value for comparison.
fn field_as_f64(fv: &FieldValue) -> Option<f64> {
    match fv {
        FieldValue::Int32(n) => Some(*n as f64),
        FieldValue::Int64(n) => Some(*n as f64),
        FieldValue::Float32(n) => Some(*n as f64),
        FieldValue::Float64(n) => Some(*n),
        FieldValue::Timestamp(n) => Some(*n as f64),
        _ => None,
    }
}

/// Compare a record's field value against a filter value using the given operator.
fn match_cmp(record: &Record, field: &str, value: &FilterValue, op: CmpOp) -> bool {
    let Some(fv) = resolve_field(record, field) else {
        return false;
    };

    // Try numeric comparison first.
    if let Some(field_f64) = field_as_f64(fv.as_ref())
        && let Some(value_f64) = value.as_f64()
    {
        return match op {
            CmpOp::Gt => field_f64 > value_f64,
            CmpOp::Gte => field_f64 >= value_f64,
            CmpOp::Lt => field_f64 < value_f64,
            CmpOp::Lte => field_f64 <= value_f64,
        };
    }

    // String comparison fallback.
    if let (FieldValue::String(s), FilterValue::String(v)) = (fv.as_ref(), value) {
        return match op {
            CmpOp::Gt => s.as_str() > v.as_str(),
            CmpOp::Gte => s.as_str() >= v.as_str(),
            CmpOp::Lt => s.as_str() < v.as_str(),
            CmpOp::Lte => s.as_str() <= v.as_str(),
        };
    }

    false
}

/// Check if a record's field value is within [low, high] (inclusive).
fn match_between(record: &Record, field: &str, low: &FilterValue, high: &FilterValue) -> bool {
    let Some(fv) = resolve_field(record, field) else {
        return false;
    };
    let Some(field_f64) = field_as_f64(fv.as_ref()) else {
        return false;
    };
    let Some(lo) = low.as_f64() else {
        return false;
    };
    let Some(hi) = high.as_f64() else {
        return false;
    };
    field_f64 >= lo && field_f64 <= hi
}

/// Check if a record's string field contains the given substring.
fn match_contains(record: &Record, field: &str, substring: &str) -> bool {
    let Some(fv) = resolve_field(record, field) else {
        return false;
    };
    let FieldValue::String(s) = fv.as_ref() else {
        return false;
    };
    s.contains(substring)
}

/// Check if a record's string field starts with the given prefix.
fn match_starts_with(record: &Record, field: &str, prefix: &str) -> bool {
    let Some(fv) = resolve_field(record, field) else {
        return false;
    };
    let FieldValue::String(s) = fv.as_ref() else {
        return false;
    };
    s.starts_with(prefix)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter::parser::FilterParser;

    fn make_record(category: &str, price: f64) -> Record {
        let mut r = Record::new();
        r.set("category", FieldValue::String(category.to_owned()));
        r.set("price", FieldValue::Float64(price));
        r
    }

    #[test]
    fn test_eq_string() {
        let r = make_record("electronics", 99.99);
        let expr = FilterExpr::Eq {
            field: "category".into(),
            value: FilterValue::String("electronics".into()),
        };
        assert!(evaluate_record(&expr, &r));
    }

    #[test]
    fn test_eq_string_mismatch() {
        let r = make_record("books", 19.99);
        let expr = FilterExpr::Eq {
            field: "category".into(),
            value: FilterValue::String("electronics".into()),
        };
        assert!(!evaluate_record(&expr, &r));
    }

    #[test]
    fn test_gt_float() {
        let r = make_record("x", 150.0);
        let expr = FilterExpr::Gt {
            field: "price".into(),
            value: FilterValue::Float(100.0),
        };
        assert!(evaluate_record(&expr, &r));
    }

    #[test]
    fn test_between() {
        let r = make_record("x", 150.0);
        let expr = FilterExpr::Between {
            field: "price".into(),
            low: FilterValue::Float(100.0),
            high: FilterValue::Float(200.0),
        };
        assert!(evaluate_record(&expr, &r));
    }

    #[test]
    fn test_and() {
        let r = make_record("electronics", 150.0);
        let expr = FilterExpr::And(
            Box::new(FilterExpr::Eq {
                field: "category".into(),
                value: FilterValue::String("electronics".into()),
            }),
            Box::new(FilterExpr::Lt {
                field: "price".into(),
                value: FilterValue::Float(200.0),
            }),
        );
        assert!(evaluate_record(&expr, &r));
    }

    #[test]
    fn test_is_null() {
        let r = make_record("x", 10.0);
        let expr = FilterExpr::IsNull {
            field: "nonexistent".into(),
        };
        assert!(evaluate_record(&expr, &r));
    }

    #[test]
    fn test_is_not_null() {
        let r = make_record("x", 10.0);
        let expr = FilterExpr::IsNotNull {
            field: "category".into(),
        };
        assert!(evaluate_record(&expr, &r));
    }

    #[test]
    fn test_in() {
        let r = make_record("books", 25.0);
        let expr = FilterExpr::In {
            field: "category".into(),
            values: vec![
                FilterValue::String("electronics".into()),
                FilterValue::String("books".into()),
            ],
        };
        assert!(evaluate_record(&expr, &r));
    }

    #[test]
    fn test_not() {
        let r = make_record("books", 25.0);
        let expr = FilterExpr::Not(Box::new(FilterExpr::Eq {
            field: "category".into(),
            value: FilterValue::String("electronics".into()),
        }));
        assert!(evaluate_record(&expr, &r));
    }

    // ── JSON-path tests ───────────────────────────────────────────────────

    #[test]
    fn json_path_equality_matches_nested_value() {
        let mut r = Record::new();
        r.set(
            "meta",
            FieldValue::Json(serde_json::json!({
                "user": {"tier": "gold", "age": 30},
                "tags": ["a", "b"]
            })),
        );
        let f = FilterParser::parse(r#"meta.user.tier = "gold""#).unwrap();
        assert!(evaluate_record(&f, &r));
        let f2 = FilterParser::parse(r#"meta.user.tier = "silver""#).unwrap();
        assert!(!evaluate_record(&f2, &r));
    }

    #[test]
    fn json_path_numeric_comparison() {
        let mut r = Record::new();
        r.set(
            "meta",
            FieldValue::Json(serde_json::json!({"user": {"age": 30}})),
        );
        assert!(evaluate_record(
            &FilterParser::parse("meta.user.age > 25").unwrap(),
            &r
        ));
        assert!(!evaluate_record(
            &FilterParser::parse("meta.user.age > 40").unwrap(),
            &r
        ));
    }

    #[test]
    fn json_path_missing_is_false_not_error() {
        let mut r = Record::new();
        r.set(
            "meta",
            FieldValue::Json(serde_json::json!({"user": {}})),
        );
        // absent path → no match, no panic
        assert!(!evaluate_record(
            &FilterParser::parse(r#"meta.user.tier = "gold""#).unwrap(),
            &r
        ));
    }

    #[test]
    fn plain_scalar_field_unaffected() {
        let mut r = Record::new();
        r.set("category", FieldValue::String("docs".into()));
        assert!(evaluate_record(
            &FilterParser::parse(r#"category = "docs""#).unwrap(),
            &r
        ));
    }
}
