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

/// Check equality between a record's field value and a filter value.
///
/// For Array field values, returns true if any element equals the filter value
/// (membership semantics — `tags = 'foo'` matches a record whose `tags` array
/// contains `"foo"`).
fn match_eq(record: &Record, field: &str, value: &FilterValue) -> bool {
    let Some(fv) = record.get(field) else {
        return false;
    };
    if let FieldValue::Array(elems) = fv {
        return elems.iter().any(|elem| match_field_value(elem, value));
    }
    match_field_value(fv, value)
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
    let Some(fv) = record.get(field) else {
        return false;
    };

    // Try numeric comparison first.
    if let Some(field_f64) = field_as_f64(fv)
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
    if let (FieldValue::String(s), FilterValue::String(v)) = (fv, value) {
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
    let Some(fv) = record.get(field) else {
        return false;
    };
    let Some(field_f64) = field_as_f64(fv) else {
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
    let Some(FieldValue::String(s)) = record.get(field) else {
        return false;
    };
    s.contains(substring)
}

/// Check if a record's string field starts with the given prefix.
fn match_starts_with(record: &Record, field: &str, prefix: &str) -> bool {
    let Some(FieldValue::String(s)) = record.get(field) else {
        return false;
    };
    s.starts_with(prefix)
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
