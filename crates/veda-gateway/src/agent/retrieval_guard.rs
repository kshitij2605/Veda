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
    EXPIRES_AT_FIELD, GUARDRAIL_CATEGORY_FIELD, GUARDRAIL_FLAGGED_FIELD, GUARDRAIL_SCORE_FIELD,
    INDEXED_AT_FIELD,
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

/// A temporal anomaly in the retained set.
///
/// `kind` is `"stale"` (age far above the set median), `"expired"` (past
/// `_expires_at`), or `"conflict"` (a near-duplicate that is materially older
/// than its twin — the newer is preferred and this one is reported).
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

/// A hit older than this multiple of the retained set's median age is stale.
pub const STALENESS_MEDIAN_MULTIPLE: f64 = 3.0;
/// Cosine at or above which two hits are treated as near-duplicates.
pub const DUPLICATE_COSINE_THRESHOLD: f32 = 0.95;
/// Age ratio between near-duplicates that constitutes a temporal conflict.
pub const CONFLICT_AGE_RATIO: f64 = 2.0;

fn hit_id(hit: &serde_json::Value) -> String {
    hit.get("id")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_owned()
}

fn hit_age_secs(hit: &serde_json::Value, now_secs: i64) -> Option<i64> {
    hit.get("fields")
        .and_then(|f| f.get(INDEXED_AT_FIELD))
        .and_then(serde_json::Value::as_i64)
        .map(|ts| now_secs.saturating_sub(ts))
}

fn hit_vector(hit: &serde_json::Value) -> Option<Vec<f32>> {
    let fields = hit.get("fields")?.as_object()?;
    for value in fields.values() {
        if let Some(arr) = value.as_array()
            && arr.len() > 1
            && arr.iter().all(serde_json::Value::is_number)
        {
            return Some(
                arr.iter()
                    .filter_map(|v| v.as_f64())
                    .map(|f| f as f32)
                    .collect(),
            );
        }
    }
    None
}

fn cosine_f32(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (mut dot, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
    for i in 0..n {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

/// Staleness + conflict findings over the retained hits.
///
/// Returns an empty vector when fewer than one hit carries `_indexed_at` —
/// with no ages there is nothing to compare, and guessing would be worse than
/// silence.
pub fn temporal_check(hits: &[serde_json::Value], now_secs: i64) -> Vec<TemporalFinding> {
    let mut findings = Vec::new();

    // Expired is per-hit and needs no comparison set.
    for hit in hits {
        let expired = hit
            .get("fields")
            .and_then(|f| f.get(EXPIRES_AT_FIELD))
            .and_then(serde_json::Value::as_i64)
            .is_some_and(|ts| ts <= now_secs);
        if expired {
            findings.push(TemporalFinding {
                kind: "expired".to_owned(),
                id: hit_id(hit),
                age_secs: hit_age_secs(hit, now_secs).unwrap_or(0),
            });
        }
    }

    let mut ages: Vec<(usize, i64)> = hits
        .iter()
        .enumerate()
        .filter_map(|(i, h)| hit_age_secs(h, now_secs).map(|a| (i, a)))
        .collect();
    if ages.is_empty() {
        return findings;
    }

    // Staleness: age far above the median.
    let mut sorted: Vec<i64> = ages.iter().map(|(_, a)| *a).collect();
    sorted.sort_unstable();
    let median = sorted[sorted.len() / 2] as f64;
    if median > 0.0 {
        for (i, age) in &ages {
            if *age as f64 > median * STALENESS_MEDIAN_MULTIPLE {
                findings.push(TemporalFinding {
                    kind: "stale".to_owned(),
                    id: hit_id(&hits[*i]),
                    age_secs: *age,
                });
            }
        }
    }

    // Conflict: near-duplicate content whose ages differ materially. Report
    // the OLDER of each pair — the newer is the one to prefer.
    ages.sort_unstable_by_key(|(_, a)| *a);
    for x in 0..ages.len() {
        for y in (x + 1)..ages.len() {
            let (i, newer) = ages[x];
            let (j, older) = ages[y];
            if newer <= 0 || (older as f64) < newer as f64 * CONFLICT_AGE_RATIO {
                continue;
            }
            let (Some(vi), Some(vj)) = (hit_vector(&hits[i]), hit_vector(&hits[j])) else {
                continue;
            };
            if cosine_f32(&vi, &vj) >= DUPLICATE_COSINE_THRESHOLD {
                findings.push(TemporalFinding {
                    kind: "conflict".to_owned(),
                    id: hit_id(&hits[j]),
                    age_secs: older,
                });
            }
        }
    }

    findings
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

    fn aged_hit(id: &str, indexed_at: i64, vec: Vec<f32>) -> serde_json::Value {
        let mut fields = serde_json::Map::new();
        fields.insert("content".into(), json!("text"));
        fields.insert("_guardrail_flagged".into(), json!(false));
        fields.insert("_indexed_at".into(), json!(indexed_at));
        fields.insert("vec".into(), json!(vec));
        json!({ "id": id, "distance": 0.1, "fields": fields })
    }

    const NOW: i64 = 1_000_000;

    #[test]
    fn flags_a_stale_outlier() {
        // Three recent hits, one ~10x older than the median age.
        let hits = vec![
            aged_hit("a", NOW - 100, vec![1.0, 0.0]),
            aged_hit("b", NOW - 110, vec![0.0, 1.0]),
            aged_hit("c", NOW - 120, vec![0.0, 0.0]),
            aged_hit("old", NOW - 5_000, vec![0.5, 0.5]),
        ];
        let findings = temporal_check(&hits, NOW);
        assert!(
            findings.iter().any(|f| f.id == "old" && f.kind == "stale"),
            "the outlier must be reported: {findings:?}"
        );
        assert!(
            !findings.iter().any(|f| f.id == "a"),
            "recent hits must not be reported: {findings:?}"
        );
    }

    #[test]
    fn flags_an_expired_hit() {
        let mut h = aged_hit("exp", NOW - 100, vec![1.0, 0.0]);
        h["fields"]["_expires_at"] = json!(NOW - 10);
        let findings = temporal_check(&[h], NOW);
        assert!(
            findings
                .iter()
                .any(|f| f.kind == "expired" && f.id == "exp")
        );
    }

    #[test]
    fn flags_a_near_duplicate_conflict_across_ages() {
        // Same direction (cosine 1.0) but ages differ by >2x.
        let hits = vec![
            aged_hit("new", NOW - 100, vec![1.0, 0.0]),
            aged_hit("stale_dup", NOW - 400, vec![1.0, 0.0]),
        ];
        let findings = temporal_check(&hits, NOW);
        assert!(
            findings
                .iter()
                .any(|f| f.kind == "conflict" && f.id == "stale_dup"),
            "the older near-duplicate is the one reported: {findings:?}"
        );
        assert!(
            !findings.iter().any(|f| f.id == "new"),
            "the newer of a conflicting pair is preferred, not reported: {findings:?}"
        );
    }

    #[test]
    fn distinct_content_is_not_a_conflict() {
        let hits = vec![
            aged_hit("a", NOW - 100, vec![1.0, 0.0]),
            aged_hit("b", NOW - 400, vec![0.0, 1.0]),
        ];
        let findings = temporal_check(&hits, NOW);
        assert!(
            !findings.iter().any(|f| f.kind == "conflict"),
            "orthogonal vectors are not near-duplicates: {findings:?}"
        );
    }

    #[test]
    fn reports_nothing_without_age_data() {
        let hits = vec![
            hit("a", Some(false), 0.0, None),
            hit("b", Some(false), 0.0, None),
        ];
        assert!(
            temporal_check(&hits, NOW).is_empty(),
            "no ages → no guessing"
        );
    }
}
