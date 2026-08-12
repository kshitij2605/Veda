//! Query-time retrieval guardrails: enforcement policy and the temporal
//! checks, kept separate from `guardrails.rs` (ingest-time detection) because
//! the concerns differ — detection asks "is this chunk hostile?", enforcement
//! asks "what should this request do about it?".
//!
//! Everything here is PURE and operates on the JSON hit shape the agent tools
//! emit (`{ id, distance, fields }`), so it is unit-testable with no state,
//! no embedder and no LLM.

use serde::{Deserialize, Serialize};

use veda_core::record::{
    GUARDRAIL_CATEGORY_FIELD, GUARDRAIL_FLAGGED_FIELD, GUARDRAIL_SCORE_FIELD, INDEXED_AT_FIELD,
};

/// How a retrieved hit was classified for enforcement.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum HitClass {
    /// Screened, nothing found.
    ScreenedClean,
    /// Screened and flagged — the only class that is ever dropped.
    ScreenedFlagged,
    /// Screening did not apply (media placeholder, no text embedding).
    ScreenedSkipped,
    /// No verdict at all — indexed before retrieval guardrails shipped.
    Unscreened,
}

/// One dropped chunk, reported so the drop is never silent.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct DroppedChunk {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub category: Option<String>,
    pub score: f64,
}

/// A temporal anomaly found in the retained set. Populated in Task 5.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct TemporalFinding {
    pub kind: String,
    pub id: String,
    pub age_secs: i64,
}

/// What enforcement did to one retrieval, attached to LLM-facing responses.
///
/// `unscreened` and `skipped` are always serialized, even at zero: an
/// operator must be able to tell "nothing was hostile" from "nothing was
/// checked".
#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
pub struct RetrievalGuardReport {
    pub kept: usize,
    pub dropped: usize,
    pub unscreened: usize,
    pub skipped: usize,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub dropped_chunks: Vec<DroppedChunk>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub temporal: Vec<TemporalFinding>,
}

/// Classify one hit from its stored fields.
pub fn classify_hit(fields: &serde_json::Map<String, serde_json::Value>) -> HitClass {
    match fields
        .get(GUARDRAIL_FLAGGED_FIELD)
        .and_then(|v| v.as_bool())
    {
        Some(true) => HitClass::ScreenedFlagged,
        Some(false) => HitClass::ScreenedClean,
        // No verdict. `_indexed_at` present means a write path ran but
        // screening did not apply; absent means pre-feature data.
        None if fields.contains_key(INDEXED_AT_FIELD) => HitClass::ScreenedSkipped,
        None => HitClass::Unscreened,
    }
}

/// Partition hits by policy. Only [`HitClass::ScreenedFlagged`] is ever
/// dropped, and only when `drop_flagged` is true.
pub fn enforce_retrieved(
    hits: Vec<serde_json::Value>,
    drop_flagged: bool,
) -> (Vec<serde_json::Value>, RetrievalGuardReport) {
    let mut kept = Vec::with_capacity(hits.len());
    let mut report = RetrievalGuardReport::default();

    for hit in hits {
        let class = hit
            .get("fields")
            .and_then(|f| f.as_object())
            .map(classify_hit)
            .unwrap_or(HitClass::Unscreened);

        match class {
            HitClass::ScreenedSkipped => report.skipped += 1,
            HitClass::Unscreened => report.unscreened += 1,
            _ => {}
        }

        if class == HitClass::ScreenedFlagged && drop_flagged {
            let fields = hit.get("fields").and_then(|f| f.as_object());
            report.dropped += 1;
            report.dropped_chunks.push(DroppedChunk {
                id: hit
                    .get("id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_owned(),
                category: fields
                    .and_then(|f| f.get(GUARDRAIL_CATEGORY_FIELD))
                    .and_then(|v| v.as_str())
                    .map(str::to_owned),
                score: fields
                    .and_then(|f| f.get(GUARDRAIL_SCORE_FIELD))
                    .and_then(serde_json::Value::as_f64)
                    .unwrap_or(0.0),
            });
            continue;
        }
        kept.push(hit);
    }

    report.kept = kept.len();
    (kept, report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn hit(
        id: &str,
        flagged: Option<bool>,
        score: f64,
        indexed_at: Option<i64>,
    ) -> serde_json::Value {
        let mut fields = serde_json::Map::new();
        fields.insert("content".into(), json!("text"));
        if let Some(f) = flagged {
            fields.insert("_guardrail_flagged".into(), json!(f));
            fields.insert("_guardrail_score".into(), json!(score));
            if f {
                fields.insert("_guardrail_category".into(), json!("indirect_injection"));
            }
        }
        if let Some(ts) = indexed_at {
            fields.insert("_indexed_at".into(), json!(ts));
        }
        json!({ "id": id, "distance": 0.1, "fields": fields })
    }

    #[test]
    fn classifies_each_hit_kind() {
        assert_eq!(
            classify_hit(
                hit("a", Some(false), 0.1, Some(1))["fields"]
                    .as_object()
                    .unwrap()
            ),
            HitClass::ScreenedClean
        );
        assert_eq!(
            classify_hit(
                hit("b", Some(true), 0.8, Some(1))["fields"]
                    .as_object()
                    .unwrap()
            ),
            HitClass::ScreenedFlagged
        );
        // No _guardrail_flagged but has _indexed_at → skipped (a write path
        // that stamped age but could not screen).
        assert_eq!(
            classify_hit(hit("c", None, 0.0, Some(1))["fields"].as_object().unwrap()),
            HitClass::ScreenedSkipped
        );
        // Neither field → indexed before this feature existed.
        assert_eq!(
            classify_hit(hit("d", None, 0.0, None)["fields"].as_object().unwrap()),
            HitClass::Unscreened
        );
    }

    #[test]
    fn drops_only_flagged_hits_and_reports_them() {
        let hits = vec![
            hit("a", Some(false), 0.1, Some(1)),
            hit("b", Some(true), 0.81, Some(1)),
            hit("c", None, 0.0, None),
        ];
        let (kept, report) = enforce_retrieved(hits, true);
        assert_eq!(kept.len(), 2, "only the flagged hit is dropped");
        assert_eq!(report.kept, 2);
        assert_eq!(report.dropped, 1);
        assert_eq!(report.unscreened, 1);
        assert_eq!(report.dropped_chunks.len(), 1);
        assert_eq!(report.dropped_chunks[0].id, "b");
        assert_eq!(
            report.dropped_chunks[0].category.as_deref(),
            Some("indirect_injection")
        );
    }

    #[test]
    fn annotate_only_mode_drops_nothing() {
        let hits = vec![hit("a", Some(true), 0.9, Some(1))];
        let (kept, report) = enforce_retrieved(hits, false);
        assert_eq!(kept.len(), 1, "drop_flagged=false must keep every hit");
        assert_eq!(report.dropped, 0);
        assert_eq!(report.kept, 1);
    }

    #[test]
    fn unscreened_hits_are_never_dropped() {
        let hits = vec![hit("a", None, 0.0, None), hit("b", None, 0.0, None)];
        let (kept, report) = enforce_retrieved(hits, true);
        assert_eq!(kept.len(), 2, "unscreened content passes through by design");
        assert_eq!(report.unscreened, 2);
        assert_eq!(report.dropped, 0);
    }

    #[test]
    fn counts_are_present_even_when_zero() {
        let (_, report) = enforce_retrieved(vec![], true);
        let v = serde_json::to_value(&report).unwrap();
        for k in ["kept", "dropped", "unscreened", "skipped"] {
            assert!(v.get(k).is_some(), "{k} must always be serialized: {v}");
        }
    }
}
