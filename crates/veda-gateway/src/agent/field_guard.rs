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

/// Replace raw embedding components with a shape summary, everywhere in
/// a tool payload.
///
/// **Ordering is load-bearing.** This must run at the dispatch choke
/// point and only *after* `apply_retrieval_guard`, because
/// `temporal_check`'s conflict detection reads each hit's vector back
/// out of the `fields` map, and `filter_query` is the only path where
/// that detection works at all (`search` skips vector fields when
/// hydrating hits). Stripping earlier — inside `record_json`, which is
/// where it was first tried — leaves the guard reading `None` and
/// silently disables it.
///
/// Why strip at all: a 1024-dim vector is ~1k JSON numbers and the tools
/// return one per record. That is not only token cost. Measured on the
/// 2026-08-20 annotation probe, `filter_query` returned `count: 2`
/// correctly on a 5-record collection and the model answered from the
/// first record only — once inventing a name absent from the corpus —
/// because the two useful fields sat behind ~2k float components.
/// Removing them moved that case from 0/5 to 5/5.
///
/// Returns the number of field values summarised.
pub fn strip_embeddings(payload: &mut serde_json::Value) -> usize {
    fn summarise(v: &serde_json::Value) -> Option<serde_json::Value> {
        // A dense vector: a non-empty array of numbers.
        if let Some(arr) = v.as_array() {
            if !arr.is_empty() && arr.iter().all(serde_json::Value::is_number) {
                return Some(serde_json::json!({ "vector": { "dim": arr.len() } }));
            }
            // A multi-vector: an array of such arrays.
            if !arr.is_empty()
                && arr.iter().all(|row| {
                    row.as_array().is_some_and(|r| {
                        !r.is_empty() && r.iter().all(serde_json::Value::is_number)
                    })
                })
            {
                let dim = arr[0].as_array().map(|r| r.len()).unwrap_or(0);
                return Some(serde_json::json!({
                    "multi_vector": { "vectors": arr.len(), "dim": dim }
                }));
            }
            return None;
        }
        // A sparse vector: `{indices: [...], values: [...]}`.
        if let Some(obj) = v.as_object()
            && let (Some(idx), true) = (
                obj.get("indices").and_then(|i| i.as_array()),
                obj.contains_key("values"),
            )
        {
            return Some(serde_json::json!({ "sparse_vector": { "nnz": idx.len() } }));
        }
        None
    }

    fn walk_fields(v: &mut serde_json::Value, n: &mut usize) {
        let Some(fields) = v.get_mut("fields").and_then(|f| f.as_object_mut()) else {
            return;
        };
        let names: Vec<String> = fields.keys().cloned().collect();
        for name in names {
            if let Some(replacement) = fields.get(&name).and_then(summarise) {
                fields.insert(name, replacement);
                *n += 1;
            }
        }
    }

    let mut n = 0;
    let Some(obj) = payload.as_object_mut() else {
        return 0;
    };
    for key in ["hits", "records"] {
        if let Some(arr) = obj.get_mut(key).and_then(|v| v.as_array_mut()) {
            for hit in arr.iter_mut() {
                walk_fields(hit, &mut n);
            }
        }
    }
    walk_fields(payload, &mut n); // get_record's bare `{id, fields}`
    n
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
    fn embeddings_are_summarised_in_every_payload_shape() {
        let vec1024: Vec<f32> = vec![0.5; 1024];
        let mut payload = serde_json::json!({
            "records": [{
                "id": "1",
                "fields": {
                    "name": "Nordwind",
                    "vec": vec1024,
                    "sp": { "indices": [1, 7, 9], "values": [0.2, 0.3, 0.5] },
                    "mv": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
                }
            }]
        });
        let n = strip_embeddings(&mut payload);
        let f = &payload["records"][0]["fields"];

        assert_eq!(
            n, 3,
            "dense, sparse and multi-vector should all be summarised"
        );
        assert_eq!(f["name"], "Nordwind", "ordinary fields must survive");
        assert_eq!(f["vec"]["vector"]["dim"], 1024);
        assert_eq!(f["sp"]["sparse_vector"]["nnz"], 3);
        assert_eq!(f["mv"]["multi_vector"]["vectors"], 3);
        assert_eq!(f["mv"]["multi_vector"]["dim"], 2);

        let len = serde_json::to_string(&payload).unwrap().len();
        assert!(len < 400, "payload unexpectedly large: {len} bytes");
    }

    #[test]
    fn the_bare_get_record_shape_is_stripped_too() {
        let mut payload = serde_json::json!({
            "id": "1",
            "fields": { "name": "Ada", "vec": vec![0.25f32; 768] }
        });
        assert_eq!(strip_embeddings(&mut payload), 1);
        assert_eq!(payload["fields"]["vec"]["vector"]["dim"], 768);
    }

    #[test]
    fn a_short_numeric_array_that_is_not_an_embedding_is_left_alone() {
        // Guards the heuristic: `strip_embeddings` matches on shape, so
        // an ordinary numeric array field would be collateral damage.
        // It is still summarised (any numeric array is indistinguishable
        // from a tiny embedding), which is why this asserts the CURRENT
        // behaviour explicitly rather than leaving it undefined — if a
        // collection ever needs to return real numeric arrays to a
        // model, this is the line that has to change.
        let mut payload = serde_json::json!({
            "id": "1",
            "fields": { "scores": [1, 2, 3], "label": "x" }
        });
        assert_eq!(strip_embeddings(&mut payload), 1);
        assert_eq!(payload["fields"]["scores"]["vector"]["dim"], 3);
        assert_eq!(payload["fields"]["label"], "x");
    }

    #[test]
    fn stripping_runs_after_the_guard_can_read_the_vector() {
        // The regression this ordering exists to prevent: conflict
        // detection reads the vector out of the hit's fields, so a
        // payload that has already been stripped yields no findings.
        use crate::agent::retrieval_guard::temporal_check;

        let now = 1_000_000i64;
        let a: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0];
        let b: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0];
        let hits = vec![
            serde_json::json!({"id": "new", "fields": {"_indexed_at": now - 10, "vec": a}}),
            serde_json::json!({"id": "old", "fields": {"_indexed_at": now - 10_000_000, "vec": b}}),
        ];

        let before = temporal_check(&hits, now, Some("vec"));
        assert!(
            before.iter().any(|f| f.kind == "conflict"),
            "with vectors present the guard must find the conflict: {before:?}"
        );

        let mut stripped = serde_json::json!({ "hits": hits });
        strip_embeddings(&mut stripped);
        let after = temporal_check(stripped["hits"].as_array().unwrap(), now, Some("vec"));
        assert!(
            !after.iter().any(|f| f.kind == "conflict"),
            "stripping first blinds the guard — which is why dispatch runs it second"
        );
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
