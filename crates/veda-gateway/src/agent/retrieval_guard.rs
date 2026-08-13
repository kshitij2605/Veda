//! Query-time retrieval guardrails: enforcement policy and the temporal
//! checks, kept separate from `guardrails.rs` (ingest-time detection) because
//! the concerns differ — detection asks "is this chunk hostile?", enforcement
//! asks "what should this request do about it?".
//!
//! Everything here is PURE and operates on the JSON hit shape the agent tools
//! emit (`{ id, distance, fields }`), so it is unit-testable with no state,
//! no embedder and no LLM. The two exceptions are [`record_is_hostile`],
//! which applies the same policy to a `Record` for the paths that never build
//! a JSON hit (the `aggregate` tool), and [`enforcement_enabled`], which
//! reads the operator kill switch out of the environment.

use serde::{Deserialize, Serialize};

use veda_core::record::{
    EXPIRES_AT_FIELD, GUARDRAIL_CATEGORY_FIELD, GUARDRAIL_FLAGGED_FIELD, GUARDRAIL_SCORE_FIELD,
    INDEXED_AT_FIELD, Record,
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
/// `age_secs` is `None` when the anomaly was detected without a known age
/// (e.g. a hit past `_expires_at` that carries no `_indexed_at`) — guessing
/// an age would be worse than reporting none.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct TemporalFinding {
    pub kind: String,
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub age_secs: Option<i64>,
}

/// What enforcement did to one retrieval, attached to LLM-facing responses.
///
/// `unscreened` and `skipped` are always serialized, even at zero: an
/// operator must be able to tell "nothing was hostile" from "nothing was
/// checked". `flagged_not_dropped` is the same idea applied to category
/// policy (2026-08-13 ruling): a flagged hit whose category is exempt from
/// dropping (currently only `Pii` — see [`category_is_hostile`]), or one
/// that stayed flagged-but-kept because `drop_flagged` was false, must be
/// just as visible as an outright drop. "Detected but allowed" must never
/// be invisible.
#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
pub struct RetrievalGuardReport {
    pub kept: usize,
    pub dropped: usize,
    pub unscreened: usize,
    pub skipped: usize,
    pub flagged_not_dropped: usize,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub dropped_chunks: Vec<DroppedChunk>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub temporal: Vec<TemporalFinding>,
}

impl RetrievalGuardReport {
    /// Fold another retrieval's report into this one: counts add, findings
    /// concatenate. Used to roll one turn's several tool calls up into the
    /// single turn-level report `/converse` returns.
    fn merge(&mut self, other: RetrievalGuardReport) {
        self.kept += other.kept;
        self.dropped += other.dropped;
        self.unscreened += other.unscreened;
        self.skipped += other.skipped;
        self.flagged_not_dropped += other.flagged_not_dropped;
        self.dropped_chunks.extend(other.dropped_chunks);
        self.temporal.extend(other.temporal);
    }
}

/// Roll every retrieving tool call in one agent turn up into a single
/// turn-level report, for the top-level `retrieval_guard` field on
/// `/converse` and the `/converse/stream` `done` frame.
///
/// Returns `None` when the turn made no retrieving tool call at all, so the
/// field is *omitted* rather than reported as an all-zero report — "the agent
/// never retrieved" and "the agent retrieved and everything was clean" are
/// different facts and the design (§6) rests on an operator being able to
/// tell them apart.
///
/// Reads the per-call reports back out of the recorded steps rather than
/// threading a second channel out of the executor: `dispatch` already wrote
/// one into every tool result, and the steps are the same data the transports
/// serialize, so there is no way for the two to disagree.
pub fn collect_from_steps(
    steps: &[crate::agent::model::AgentStep],
) -> Option<RetrievalGuardReport> {
    let mut out: Option<RetrievalGuardReport> = None;
    for step in steps {
        let crate::agent::model::AgentStep::ToolResult {
            result, is_error, ..
        } = step
        else {
            continue;
        };
        if *is_error {
            continue;
        }
        let Some(value) = result.get("retrieval_guard") else {
            continue;
        };
        let Ok(report) = serde_json::from_value::<RetrievalGuardReport>(value.clone()) else {
            continue;
        };
        out.get_or_insert_with(RetrievalGuardReport::default)
            .merge(report);
    }
    out
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

/// Whether a flagged hit's guardrail category is hostile — content an LLM
/// must never see — as opposed to exempt-from-dropping.
///
/// 2026-08-13 ruling: live calibration on a clean 149-chunk business corpus
/// found `_guardrail_flagged` chunks were being dropped purely because they
/// contained a PII pattern (an email/phone/etc. in an otherwise-benign
/// vendor contact deck) — `screen_chunk` reuses the existing PII regex
/// layer, and dropping never distinguished "hostile instructions" from
/// "contact details present". This feature targets OWASP's
/// injection/poisoning/temporal vectors; PII in retrieved content is a
/// different concern, and a user asking "who do I contact about X" getting
/// contact details silently stripped is the worse failure. So: `Pii` is
/// detected and reported (`RetrievalGuardReport::flagged_not_dropped`) but
/// passed through; every other category is hostile and dropped.
///
/// An unrecognised or missing category on a flagged hit is treated as
/// hostile — fails closed, since an unfamiliar category is more likely a
/// new threat than a benign one. Matches on the raw string (rather than
/// depending on `agent::guardrails::GuardrailCategory`) to keep this module
/// decoupled from the detection layer, per its module doc.
pub fn category_is_hostile(category: Option<&str>) -> bool {
    !matches!(category, Some("pii"))
}

/// Environment variable that turns query-time *dropping* off.
pub const ENFORCE_ENV: &str = "VEDA_RETRIEVAL_GUARD";

/// Whether query-time enforcement (dropping) is on. **Defaults to on** — the
/// switch can only ever weaken enforcement, never enable it by accident.
///
/// The choke point in `agent::catalog::dispatch` is deliberately always-on
/// rather than routed through the agent's `guardrails` flag (design §4.9):
/// per-agent opt-in would mean the default agent ships unprotected. But
/// always-on with no override leaves an operator whose corpus trips a
/// false-positive pattern with no remedy short of editing and redeploying
/// the code, which is not an acceptable production posture for a control
/// that silently removes content from answers.
///
/// So: `VEDA_RETRIEVAL_GUARD=off` (also `0`/`false`/`no`/`disabled`, any
/// case) degrades every LLM-facing path to annotate-only — hits are still
/// classified and still reported in `RetrievalGuardReport`
/// (`flagged_not_dropped` counts what *would* have been dropped), nothing is
/// removed. Any other value, or an unset variable, enforces.
///
/// Read per call rather than cached: this is one `getenv` per tool dispatch,
/// far below the cost of the search it accompanies, and a cached value would
/// make the switch require a restart — the opposite of an incident remedy.
pub fn enforcement_enabled() -> bool {
    enforcement_from_env_value(std::env::var(ENFORCE_ENV).ok().as_deref())
}

/// Pure half of [`enforcement_enabled`], separated so the parsing is testable
/// without mutating process-wide environment state.
pub fn enforcement_from_env_value(value: Option<&str>) -> bool {
    match value {
        Some(v) => !matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "off" | "0" | "false" | "no" | "disabled"
        ),
        None => true,
    }
}

/// Whether a stored [`Record`] must be withheld from an LLM, applying the
/// exact same policy [`enforce_retrieved`] applies to a JSON hit.
///
/// Needed because not every model-reachable read builds a `{id, fields}` hit.
/// The `aggregate` tool returns `groups[].key`, which with
/// `group_by=<text field>` is the field's *verbatim value* — so
/// `group_by=content, filter=_guardrail_flagged = true` would hand a model
/// exactly the text every other path withholds, and the array-shaped guard
/// would never see it (the payload has neither `hits` nor `records`). The
/// aggregation walks `Record`s, which carry the verdict, so the skip belongs
/// there; this is the shared predicate so the rule is not written twice.
pub fn record_is_hostile(rec: &Record) -> bool {
    use veda_core::record::FieldValue;

    if !matches!(
        rec.get(GUARDRAIL_FLAGGED_FIELD),
        Some(FieldValue::Bool(true))
    ) {
        return false;
    }
    let category = match rec.get(GUARDRAIL_CATEGORY_FIELD) {
        Some(FieldValue::String(s)) => Some(s.as_str()),
        _ => None,
    };
    category_is_hostile(category)
}

/// Partition hits by policy. Only [`HitClass::ScreenedFlagged`] hits with a
/// hostile category (see [`category_is_hostile`]) are ever dropped, and
/// only when `drop_flagged` is true. A flagged hit that is kept anyway
/// (PII-exempt, or `drop_flagged` is false) still counts toward
/// `RetrievalGuardReport::flagged_not_dropped` — flagged-but-allowed must
/// never be silent.
pub fn enforce_retrieved(
    hits: Vec<serde_json::Value>,
    drop_flagged: bool,
) -> (Vec<serde_json::Value>, RetrievalGuardReport) {
    let mut kept = Vec::with_capacity(hits.len());
    let mut report = RetrievalGuardReport::default();

    for hit in hits {
        let fields = hit.get("fields").and_then(|f| f.as_object());
        let class = fields.map(classify_hit).unwrap_or(HitClass::Unscreened);

        match class {
            HitClass::ScreenedSkipped => report.skipped += 1,
            HitClass::Unscreened => report.unscreened += 1,
            _ => {}
        }

        if class == HitClass::ScreenedFlagged {
            let category = fields
                .and_then(|f| f.get(GUARDRAIL_CATEGORY_FIELD))
                .and_then(|v| v.as_str());
            if category_is_hostile(category) && drop_flagged {
                report.dropped += 1;
                report.dropped_chunks.push(DroppedChunk {
                    id: hit
                        .get("id")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_owned(),
                    category: category.map(str::to_owned),
                    score: fields
                        .and_then(|f| f.get(GUARDRAIL_SCORE_FIELD))
                        .and_then(serde_json::Value::as_f64)
                        .unwrap_or(0.0),
                });
                continue;
            }
            report.flagged_not_dropped += 1;
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

/// Read `field` from the hit's fields and return it as a vector, but only if
/// it is present and is a JSON array of numbers. No scanning for "the field
/// that looks like a vector": a bare JSON array is indistinguishable from a
/// bounding box, page range, or sparse index, so the caller must name the
/// field explicitly.
fn hit_vector(hit: &serde_json::Value, field: &str) -> Option<Vec<f32>> {
    let arr = hit.get("fields")?.get(field)?.as_array()?;
    if !arr.iter().all(serde_json::Value::is_number) {
        return None;
    }
    Some(
        arr.iter()
            .filter_map(|v| v.as_f64())
            .map(|f| f as f32)
            .collect(),
    )
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

/// Staleness, expiry, and conflict findings over the retained hits.
///
/// Expiry is checked unconditionally — `_expires_at` needs no comparison
/// set. Staleness and conflict need age data: if no hit carries
/// `_indexed_at`, both are skipped, since with no ages there is nothing to
/// compare and guessing would be worse than silence. Conflict detection
/// additionally requires `vector_field`, the name of the field holding each
/// hit's embedding; pass `None` to skip conflict detection entirely (e.g.
/// when the caller has not resolved the collection's vector field).
///
/// ## Known limitation — conflict detection is inert on the `search` path
///
/// Conflict detection needs each hit's stored vector to be present inside
/// `hit.fields`. `filter_query` satisfies that (its `record_json` copies
/// every field, vectors included), so conflicts are detected there. `search`
/// does not: `builtins::search` deliberately skips vector fields when
/// hydrating hits, and the reserved-field top-up does not add them back — so
/// `resolve_vector_field` names a field that is never present and
/// [`hit_vector`] always returns `None`. Staleness and expiry are unaffected
/// and work on both paths.
///
/// This is documented rather than fixed: hydrating raw embeddings into every
/// search hit would bloat every LLM-facing payload by (dimensions × top-k)
/// floats to serve a secondary signal. A cheaper hit-to-hit comparison
/// (content hash, or reusing the distances the index already computed) is the
/// eventual fix; until then the honest statement is that conflict findings
/// come from `filter_query` only.
pub fn temporal_check(
    hits: &[serde_json::Value],
    now_secs: i64,
    vector_field: Option<&str>,
) -> Vec<TemporalFinding> {
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
                age_secs: hit_age_secs(hit, now_secs),
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
                    age_secs: Some(*age),
                });
            }
        }
    }

    // Conflict: near-duplicate content whose ages differ materially. Report
    // the OLDER of each pair — the newer is the one to prefer. A hit can be
    // the older member of several pairs (e.g. 3+ mutual near-duplicates), so
    // track which indices are already reported and report each at most once.
    if let Some(field) = vector_field {
        ages.sort_unstable_by_key(|(_, a)| *a);
        let mut reported: std::collections::HashSet<usize> = std::collections::HashSet::new();
        for x in 0..ages.len() {
            for y in (x + 1)..ages.len() {
                let (i, newer) = ages[x];
                let (j, older) = ages[y];
                if reported.contains(&j) {
                    continue;
                }
                if newer <= 0 || (older as f64) < newer as f64 * CONFLICT_AGE_RATIO {
                    continue;
                }
                let (Some(vi), Some(vj)) =
                    (hit_vector(&hits[i], field), hit_vector(&hits[j], field))
                else {
                    continue;
                };
                if cosine_f32(&vi, &vj) >= DUPLICATE_COSINE_THRESHOLD {
                    reported.insert(j);
                    findings.push(TemporalFinding {
                        kind: "conflict".to_owned(),
                        id: hit_id(&hits[j]),
                        age_secs: Some(older),
                    });
                }
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
        assert_eq!(
            report.flagged_not_dropped, 0,
            "a hostile-category hit that was dropped must not also count as flagged_not_dropped"
        );
    }

    #[test]
    fn annotate_only_mode_drops_nothing() {
        let hits = vec![hit("a", Some(true), 0.9, Some(1))];
        let (kept, report) = enforce_retrieved(hits, false);
        assert_eq!(kept.len(), 1, "drop_flagged=false must keep every hit");
        assert_eq!(report.dropped, 0);
        assert_eq!(report.kept, 1);
        assert_eq!(
            report.flagged_not_dropped, 1,
            "flagged-but-kept must be visible even in annotate-only mode"
        );
    }

    /// 2026-08-13 ruling: PII is detected and reported, never dropped —
    /// silently stripping contact details from an answer is a worse
    /// failure than the injection/poisoning vectors this feature targets.
    #[test]
    fn pii_flagged_hits_are_kept_and_counted_not_dropped() {
        let mut fields = serde_json::Map::new();
        fields.insert("content".into(), json!("call me at 555-123-4567"));
        fields.insert(GUARDRAIL_FLAGGED_FIELD.to_owned(), json!(true));
        fields.insert(GUARDRAIL_SCORE_FIELD.to_owned(), json!(1.0));
        fields.insert(GUARDRAIL_CATEGORY_FIELD.to_owned(), json!("pii"));
        fields.insert(INDEXED_AT_FIELD.to_owned(), json!(1));
        let hit = json!({ "id": "p", "distance": 0.1, "fields": fields });

        let (kept, report) = enforce_retrieved(vec![hit], true);
        assert_eq!(
            kept.len(),
            1,
            "PII must pass through even with drop_flagged=true"
        );
        assert_eq!(report.dropped, 0);
        assert_eq!(report.dropped_chunks.len(), 0);
        assert_eq!(
            report.flagged_not_dropped, 1,
            "PII must still be reported as flagged"
        );
    }

    /// Fail closed: an unrecognised (or missing) category on a flagged hit
    /// is treated as hostile, since an unfamiliar category is more likely a
    /// new threat than a benign one.
    #[test]
    fn unknown_or_missing_category_on_flagged_hit_is_dropped() {
        let mut fields = serde_json::Map::new();
        fields.insert("content".into(), json!("text"));
        fields.insert(GUARDRAIL_FLAGGED_FIELD.to_owned(), json!(true));
        fields.insert(GUARDRAIL_SCORE_FIELD.to_owned(), json!(0.9));
        // No `_guardrail_category` at all.
        fields.insert(INDEXED_AT_FIELD.to_owned(), json!(1));
        let no_category = json!({ "id": "nc", "distance": 0.1, "fields": fields.clone() });

        fields.insert(
            GUARDRAIL_CATEGORY_FIELD.to_owned(),
            json!("some_future_category"),
        );
        let unknown_category = json!({ "id": "uc", "distance": 0.1, "fields": fields });

        let (kept, report) = enforce_retrieved(vec![no_category, unknown_category], true);
        assert_eq!(kept.len(), 0, "both must be dropped, failing closed");
        assert_eq!(report.dropped, 2);
        assert_eq!(report.flagged_not_dropped, 0);
    }

    /// The kill switch can only ever weaken enforcement: absent or
    /// unrecognised values must still enforce.
    #[test]
    fn kill_switch_defaults_to_enforcing() {
        assert!(enforcement_from_env_value(None));
        assert!(enforcement_from_env_value(Some("on")));
        assert!(enforcement_from_env_value(Some("1")));
        assert!(enforcement_from_env_value(Some("")));
        assert!(enforcement_from_env_value(Some("something-unexpected")));
        for off in ["off", "OFF", " off ", "0", "false", "no", "disabled"] {
            assert!(
                !enforcement_from_env_value(Some(off)),
                "{off} must disable dropping"
            );
        }
    }

    /// With the switch off, a hostile hit is kept but still reported — the
    /// operator remedy must degrade to annotate-only, never to blind.
    #[test]
    fn enforcement_off_keeps_and_still_reports() {
        let hits = vec![hit("b", Some(true), 0.9, Some(1))];
        let (kept, report) = enforce_retrieved(hits, enforcement_from_env_value(Some("off")));
        assert_eq!(kept.len(), 1);
        assert_eq!(report.dropped, 0);
        assert_eq!(report.flagged_not_dropped, 1);
    }

    #[test]
    fn record_is_hostile_matches_the_hit_policy() {
        use veda_core::record::FieldValue;

        let mut clean = Record::new();
        clean.set(GUARDRAIL_FLAGGED_FIELD, FieldValue::Bool(false));
        assert!(!record_is_hostile(&clean));

        let unscreened = Record::new();
        assert!(!record_is_hostile(&unscreened));

        let mut pii = Record::new();
        pii.set(GUARDRAIL_FLAGGED_FIELD, FieldValue::Bool(true));
        pii.set(
            GUARDRAIL_CATEGORY_FIELD,
            FieldValue::String("pii".to_owned()),
        );
        assert!(!record_is_hostile(&pii), "PII is exempt here too");

        let mut hostile = Record::new();
        hostile.set(GUARDRAIL_FLAGGED_FIELD, FieldValue::Bool(true));
        hostile.set(
            GUARDRAIL_CATEGORY_FIELD,
            FieldValue::String("indirect_injection".to_owned()),
        );
        assert!(record_is_hostile(&hostile));

        // Missing category on a flagged record fails closed.
        let mut no_category = Record::new();
        no_category.set(GUARDRAIL_FLAGGED_FIELD, FieldValue::Bool(true));
        assert!(record_is_hostile(&no_category));
    }

    #[test]
    fn collect_from_steps_sums_counts_and_concatenates_findings() {
        use crate::agent::model::AgentStep;

        let step = |dropped: usize, id: &str| AgentStep::ToolResult {
            tool_call_id: id.to_owned(),
            result: json!({
                "hits": [],
                "retrieval_guard": {
                    "kept": 2, "dropped": dropped, "unscreened": 1, "skipped": 0,
                    "flagged_not_dropped": 1,
                    "dropped_chunks": [{ "id": id, "category": "jailbreak", "score": 0.9 }],
                    "temporal": [{ "kind": "stale", "id": id, "age_secs": 10 }],
                }
            }),
            is_error: false,
            duration_ms: None,
        };

        let steps = vec![
            AgentStep::Reasoning {
                text: "thinking".to_owned(),
            },
            step(1, "a"),
            step(2, "b"),
        ];
        let report = collect_from_steps(&steps).expect("a retrieving turn must report");
        assert_eq!(report.kept, 4);
        assert_eq!(report.dropped, 3);
        assert_eq!(report.unscreened, 2);
        assert_eq!(report.flagged_not_dropped, 2);
        assert_eq!(report.dropped_chunks.len(), 2);
        assert_eq!(report.temporal.len(), 2);
    }

    #[test]
    fn collect_from_steps_is_none_without_a_retrieving_call() {
        use crate::agent::model::AgentStep;

        let steps = vec![
            AgentStep::Reasoning {
                text: "no tools".to_owned(),
            },
            AgentStep::ToolResult {
                tool_call_id: "c1".to_owned(),
                result: json!({ "ok": true }),
                is_error: false,
                duration_ms: None,
            },
        ];
        assert!(
            collect_from_steps(&steps).is_none(),
            "no retrieval must omit the field, not report zeros"
        );
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
        for k in [
            "kept",
            "dropped",
            "unscreened",
            "skipped",
            "flagged_not_dropped",
        ] {
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
        let findings = temporal_check(&hits, NOW, None);
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
        let findings = temporal_check(&[h], NOW, None);
        assert!(
            findings
                .iter()
                .any(|f| f.kind == "expired" && f.id == "exp")
        );
    }

    #[test]
    fn flags_an_expired_hit_without_indexed_at() {
        // Expired but no `_indexed_at` at all: still reported, but the age
        // must not be fabricated.
        let mut fields = serde_json::Map::new();
        fields.insert("content".into(), json!("text"));
        fields.insert("_guardrail_flagged".into(), json!(false));
        fields.insert("_expires_at".into(), json!(NOW - 10));
        let h = json!({ "id": "exp_no_age", "distance": 0.1, "fields": fields });

        let findings = temporal_check(&[h], NOW, None);
        let finding = findings
            .iter()
            .find(|f| f.kind == "expired" && f.id == "exp_no_age")
            .expect("expired hit must be reported even without _indexed_at");
        assert_eq!(
            finding.age_secs, None,
            "age must not be guessed when _indexed_at is absent"
        );
    }

    #[test]
    fn flags_a_near_duplicate_conflict_across_ages() {
        // Same direction (cosine 1.0) but ages differ by >2x.
        let hits = vec![
            aged_hit("new", NOW - 100, vec![1.0, 0.0]),
            aged_hit("stale_dup", NOW - 400, vec![1.0, 0.0]),
        ];
        let findings = temporal_check(&hits, NOW, Some("vec"));
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
        let findings = temporal_check(&hits, NOW, Some("vec"));
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
            temporal_check(&hits, NOW, None).is_empty(),
            "no ages → no guessing"
        );
    }

    #[test]
    fn conflict_findings_are_deduplicated_across_pairs() {
        // Three mutual near-duplicates (same direction, cosine ≈ 1.0) at
        // ages 100/300/900. The oldest qualifies as a conflict against both
        // of the others, but must be reported only once.
        let hits = vec![
            aged_hit("a", NOW - 100, vec![1.0, 0.0]),
            aged_hit("b", NOW - 300, vec![1.0, 0.0]),
            aged_hit("c", NOW - 900, vec![1.0, 0.0]),
        ];
        let findings = temporal_check(&hits, NOW, Some("vec"));
        let c_conflicts = findings
            .iter()
            .filter(|f| f.kind == "conflict" && f.id == "c")
            .count();
        assert_eq!(
            c_conflicts, 1,
            "the oldest hit must be reported at most once per kind: {findings:?}"
        );
    }

    #[test]
    fn no_vector_field_skips_conflict_detection() {
        // Clear near-duplicates by age and content, but no vector_field is
        // given, so conflict detection must not run at all.
        let hits = vec![
            aged_hit("new", NOW - 100, vec![1.0, 0.0]),
            aged_hit("stale_dup", NOW - 400, vec![1.0, 0.0]),
        ];
        let findings = temporal_check(&hits, NOW, None);
        assert!(
            !findings.iter().any(|f| f.kind == "conflict"),
            "without a named vector field, conflict detection must be skipped: {findings:?}"
        );
    }
}
