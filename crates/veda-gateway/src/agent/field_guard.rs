//! Schema-declared field sensitivity, enforced at retrieval.
//!
//! `veda_core::schema::Sensitivity::Restricted` and the `pii` flag are
//! declared once on the collection schema by whoever designs it. This
//! module is where that declaration becomes enforcement.
//!
//! It is the counterpart to [`crate::agent::retrieval_guard`], and the
//! division of labour is worth stating because the two look similar and
//! are not: `retrieval_guard` asks *"is this chunk hostile?"* — a
//! per-record verdict produced by a detector at ingest. This asks
//! *"is the operator allowed to see this column?"* — a per-field fact the
//! operator declared, needing no detector and never wrong.
//!
//! Everything here is PURE. No state, no embedder, no LLM, so every rule
//! below is unit-testable, and the tests at the bottom are the reason to
//! believe the enforcement holds.
//!
//! # Why redaction alone is not enough
//!
//! Stripping restricted fields out of results closes the obvious hole and
//! leaves three that this module also closes, each of which is a real
//! bypass rather than a hypothetical one:
//!
//! 1. **`get_record`** returns a bare `{id, fields}` payload with no
//!    `hits`/`records` array, so a payload-shaped redactor misses it.
//!    Exactly the hole `guarded_record_outcome` was written to close for
//!    hostile content ([`crate::agent::tools::builtins`]).
//! 2. **`aggregate` with `group_by`** emits each group's key VERBATIM.
//!    `group_by=salary` returns every distinct salary without ever
//!    returning the field. `metric=min|max` on a restricted field leaks
//!    it a value at a time.
//! 3. **Filter predicates** turn any restricted field into an oracle:
//!    a caller who cannot *see* `salary` can still binary-search it with
//!    `salary > 100000`, reading the answer off the result count.
//!
//! The third is the subtle one, and it is why [`filter_mentions_withheld`]
//! rejects the query outright rather than trying to sanitise it.

use serde::{Deserialize, Serialize};
use veda_query::filter::ast::FilterExpr;

/// What was withheld from a payload, so a redaction is never silent.
///
/// Attached to the tool response under `field_guard`, mirroring how
/// `retrieval_guard` reports dropped chunks. A model that is told a field
/// was withheld can say "I can't see that"; one that simply finds the key
/// missing will keep searching for it.
#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
pub struct FieldGuardReport {
    /// Field names removed from this payload, sorted and deduplicated.
    pub withheld_fields: Vec<String>,
    /// How many records had at least one field removed.
    pub affected_records: usize,
}

impl FieldGuardReport {
    /// Nothing was withheld — the report is then omitted entirely.
    pub fn is_empty(&self) -> bool {
        self.withheld_fields.is_empty()
    }
}

/// Remove `withheld` keys from one record's `fields` map.
///
/// Returns the names actually removed, so the caller can report them.
fn redact_fields_map(
    fields: &mut serde_json::Map<String, serde_json::Value>,
    withheld: &[String],
) -> Vec<String> {
    let mut removed = Vec::new();
    for name in withheld {
        if fields.remove(name).is_some() {
            removed.push(name.clone());
        }
    }
    removed
}

/// Redact restricted fields from any tool payload shape.
///
/// Handles all three shapes the tools emit, because covering only the
/// array shapes would leave `get_record` as a one-call bypass:
///
/// - `{"hits": [{"id", "fields": {...}}, ...]}` — `search`
/// - `{"records": [...]}` — `filter_query`
/// - `{"id", "fields": {...}}` — `get_record`, bare and array-less
///
/// A payload with none of these is returned untouched, so tools that
/// return no record content (`upsert_record`, `route`, …) are unaffected.
pub fn redact_payload(payload: &mut serde_json::Value, withheld: &[String]) -> FieldGuardReport {
    let mut report = FieldGuardReport::default();
    if withheld.is_empty() {
        return report;
    }
    let Some(obj) = payload.as_object_mut() else {
        return report;
    };

    let mut all_removed: Vec<String> = Vec::new();

    // Array shapes: hits (search) and records (filter_query).
    for key in ["hits", "records"] {
        if let Some(arr) = obj.get_mut(key).and_then(|v| v.as_array_mut()) {
            for hit in arr.iter_mut() {
                if let Some(f) = hit.get_mut("fields").and_then(|f| f.as_object_mut()) {
                    let removed = redact_fields_map(f, withheld);
                    if !removed.is_empty() {
                        report.affected_records += 1;
                        all_removed.extend(removed);
                    }
                }
            }
        }
    }

    // Bare shape: get_record's `{id, fields}`, which carries no array and
    // would otherwise sail straight past the loop above.
    if let Some(f) = obj.get_mut("fields").and_then(|f| f.as_object_mut()) {
        let removed = redact_fields_map(f, withheld);
        if !removed.is_empty() {
            report.affected_records += 1;
            all_removed.extend(removed);
        }
    }

    all_removed.sort();
    all_removed.dedup();
    report.withheld_fields = all_removed;
    report
}

/// The first withheld field named anywhere in a filter expression, if any.
///
/// Walks every variant. The match is exhaustive on purpose — no wildcard
/// arm — so adding a `FilterExpr` variant fails to compile here rather
/// than silently opening a hole in the oracle defence.
pub fn filter_mentions_withheld(expr: &FilterExpr, withheld: &[String]) -> Option<String> {
    let hit = |f: &String| withheld.iter().find(|w| *w == f).cloned();
    match expr {
        FilterExpr::Eq { field, .. }
        | FilterExpr::Ne { field, .. }
        | FilterExpr::Gt { field, .. }
        | FilterExpr::Gte { field, .. }
        | FilterExpr::Lt { field, .. }
        | FilterExpr::Lte { field, .. }
        | FilterExpr::Between { field, .. }
        | FilterExpr::In { field, .. }
        | FilterExpr::Contains { field, .. }
        | FilterExpr::StartsWith { field, .. }
        | FilterExpr::IsNull { field }
        | FilterExpr::IsNotNull { field } => hit(field),
        FilterExpr::And(a, b) | FilterExpr::Or(a, b) => {
            filter_mentions_withheld(a, withheld).or_else(|| filter_mentions_withheld(b, withheld))
        }
        FilterExpr::Not(inner) => filter_mentions_withheld(inner, withheld),
    }
}

/// Error message for a rejected request naming a restricted field.
///
/// Deliberately says the field is restricted rather than pretending it
/// does not exist: the schema is already visible through
/// `describe_collection`, which marks the field `"restricted": true`, so
/// denying its existence here would be inconsistent without being any
/// more secret.
pub fn restricted_field_error(field: &str) -> String {
    format!(
        "field '{field}' is restricted by the collection schema and cannot be read, filtered, or aggregated by this caller"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use veda_query::filter::ast::{FilterExpr, FilterValue};

    fn withheld() -> Vec<String> {
        vec!["salary".to_owned(), "email".to_owned()]
    }

    fn hit(id: &str) -> serde_json::Value {
        serde_json::json!({
            "id": id,
            "distance": 0.1,
            "fields": { "name": "Ada", "salary": 120000, "email": "a@b.c" }
        })
    }

    #[test]
    fn search_hits_lose_restricted_fields_and_keep_the_rest() {
        let mut payload = serde_json::json!({ "hits": [hit("1"), hit("2")], "count": 2 });
        let report = redact_payload(&mut payload, &withheld());

        for h in payload["hits"].as_array().unwrap() {
            let f = h["fields"].as_object().unwrap();
            assert!(!f.contains_key("salary"));
            assert!(!f.contains_key("email"));
            assert_eq!(f["name"], "Ada", "non-restricted fields must survive");
        }
        assert_eq!(report.withheld_fields, vec!["email", "salary"]);
        assert_eq!(report.affected_records, 2);
    }

    #[test]
    fn filter_query_records_are_redacted_too() {
        let mut payload = serde_json::json!({ "records": [hit("1")], "count": 1 });
        let report = redact_payload(&mut payload, &withheld());
        let f = payload["records"][0]["fields"].as_object().unwrap();
        assert!(!f.contains_key("salary"));
        assert_eq!(report.affected_records, 1);
    }

    #[test]
    fn get_records_bare_shape_is_redacted() {
        // The bypass this test exists for: `get_record` returns no
        // `hits`/`records` array, so a redactor that only walked arrays
        // would return the restricted field verbatim.
        let mut payload = hit("1");
        let report = redact_payload(&mut payload, &withheld());
        let f = payload["fields"].as_object().unwrap();
        assert!(
            !f.contains_key("salary"),
            "bare {{id, fields}} must be redacted"
        );
        assert!(!f.contains_key("email"));
        assert_eq!(f["name"], "Ada");
        assert_eq!(report.affected_records, 1);
    }

    #[test]
    fn nothing_withheld_leaves_the_payload_identical() {
        // Every collection created before annotations existed has an
        // empty withheld list. Enforcement must be a total no-op there.
        let original = serde_json::json!({ "hits": [hit("1")], "count": 1 });
        let mut payload = original.clone();
        let report = redact_payload(&mut payload, &[]);
        assert_eq!(payload, original);
        assert!(report.is_empty());
    }

    #[test]
    fn payloads_without_record_content_are_untouched() {
        let original = serde_json::json!({ "ok": true, "doc_id_estimate": 7 });
        let mut payload = original.clone();
        let report = redact_payload(&mut payload, &withheld());
        assert_eq!(payload, original);
        assert!(report.is_empty());
    }

    #[test]
    fn a_filter_predicate_on_a_restricted_field_is_detected() {
        // The oracle attack: the caller never sees `salary`, but can read
        // it off the result count one comparison at a time.
        let expr = FilterExpr::Gt {
            field: "salary".to_owned(),
            value: FilterValue::Int(100_000),
        };
        assert_eq!(
            filter_mentions_withheld(&expr, &withheld()),
            Some("salary".to_owned())
        );
    }

    #[test]
    fn a_restricted_field_nested_deep_in_boolean_logic_is_still_detected() {
        let expr = FilterExpr::Or(
            Box::new(FilterExpr::Eq {
                field: "dept".to_owned(),
                value: FilterValue::String("eng".to_owned()),
            }),
            Box::new(FilterExpr::Not(Box::new(FilterExpr::And(
                Box::new(FilterExpr::IsNotNull {
                    field: "name".to_owned(),
                }),
                Box::new(FilterExpr::Between {
                    field: "salary".to_owned(),
                    low: FilterValue::Int(1),
                    high: FilterValue::Int(2),
                }),
            )))),
        );
        assert_eq!(
            filter_mentions_withheld(&expr, &withheld()),
            Some("salary".to_owned())
        );
    }

    #[test]
    fn an_ordinary_filter_is_allowed() {
        let expr = FilterExpr::And(
            Box::new(FilterExpr::Eq {
                field: "dept".to_owned(),
                value: FilterValue::String("eng".to_owned()),
            }),
            Box::new(FilterExpr::IsNotNull {
                field: "name".to_owned(),
            }),
        );
        assert_eq!(filter_mentions_withheld(&expr, &withheld()), None);
    }
}
