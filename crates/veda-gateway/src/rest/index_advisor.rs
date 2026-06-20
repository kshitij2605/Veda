//! ANN index-advisor admin API (`GET /v1/admin/index-advisor`).
//!
//! Admin-only — registered under the protected router with an extra
//! `rbac::require_admin` route layer, mirroring [`crate::rest::correlations`]
//! and [`crate::rest::depmap`].
//!
//! Surfaces the recall-vs-latency picture the advisor already assembles
//! (Phase 6a's `OpsSnapshot::index_health`) per collection, paired with the
//! index-advisor rule recommendations (Phase 6b) and any open `index_*`
//! advisor events. This is the data the Niriksha index-advisor panel reads.
//!
//! Recall is a **proxy** measurement (sampled queries + capped brute force),
//! so `is_proxy` is surfaced per row and the recommendations are directional
//! and advisory only — nothing here mutates an index. The Niriksha panel's
//! opt-in "Apply reindex" button POSTs the existing
//! `/v1/collections/{name}/reindex` endpoint separately.

use axum::Json;
use axum::extract::State;
use serde::Serialize;

use crate::advisor::config::AdvisorConfig;
use crate::advisor::event::OpsEvent;
use crate::advisor::recall_probe::IndexParams;
use crate::advisor::rules::{RuleHit, catalog, run_rules};
use crate::advisor::snapshot::IndexSnapshot;
use crate::rest::state::AppState;

/// One recommendation derived from an index-advisor rule for a collection.
#[derive(Debug, Clone, Serialize)]
pub struct Recommendation {
    /// Rule id that produced it (e.g. `index_recall_degraded`).
    pub rule_id: String,
    pub severity: crate::advisor::event::Severity,
    pub title: String,
    pub detail: String,
    pub recommendation: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub runbook: Option<String>,
}

impl From<&RuleHit> for Recommendation {
    fn from(h: &RuleHit) -> Self {
        Recommendation {
            rule_id: h.rule_id.clone(),
            severity: h.severity,
            title: h.title.clone(),
            detail: h.detail.clone(),
            recommendation: h.recommendation.clone(),
            runbook: h.runbook.clone(),
        }
    }
}

/// Per-collection index-advisor row.
#[derive(Debug, Clone, Serialize)]
pub struct IndexAdvisorRow {
    pub collection: String,
    pub index_type: String,
    pub params: IndexParams,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p95_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub recall: Option<f64>,
    /// Whether the recall figure is a proxy measurement (always true when
    /// a recall sample exists; absent when no sample yet).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_proxy: Option<bool>,
    /// Suggested reindex body for the Niriksha opt-in button. Directional —
    /// index params aren't on the schema yet, so today this is a re-embed
    /// straight-reindex hint; absent when there's nothing to recommend.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suggested_reindex: Option<serde_json::Value>,
    pub recommendations: Vec<Recommendation>,
    pub open_events: Vec<OpsEvent>,
}

#[derive(Debug, Serialize)]
pub struct IndexAdvisorResponse {
    pub rows: Vec<IndexAdvisorRow>,
}

/// `GET /v1/admin/index-advisor` — per-collection recall-vs-latency rows
/// with index-advisor recommendations + open index events.
pub async fn index_advisor(State(state): State<AppState>) -> Json<IndexAdvisorResponse> {
    let advisor = state.advisor();

    // Latest assembled index_health + index-advisor rule hits, both from
    // the most recent snapshot under a single brief history lock. Empty
    // when the advisor hasn't ticked yet → an empty response (not an error).
    let cfg: AdvisorConfig = advisor.config.read().await.clone();
    let (index_health, hits_by_coll): (
        Vec<IndexSnapshot>,
        std::collections::HashMap<String, Vec<RuleHit>>,
    ) = {
        let history = advisor.history.lock().await;
        match history.latest() {
            Some(snap) => {
                let ih = snap.index_health.clone();
                // run_rules needs the history for rate-based rules; the
                // index rules ignore it, but pass the live one anyway.
                let all = run_rules(&catalog(), snap, &history, &cfg);
                let mut by: std::collections::HashMap<String, Vec<RuleHit>> =
                    std::collections::HashMap::new();
                for h in all {
                    if let crate::advisor::event::Scope::Index(c) = &h.scope {
                        by.entry(c.clone()).or_default().push(h);
                    }
                }
                (ih, by)
            }
            None => (Vec::new(), std::collections::HashMap::new()),
        }
    };

    // `is_proxy` per collection from the latest recall sample.
    let proxy_by_coll: std::collections::HashMap<String, bool> = {
        let results = advisor.recall_results.lock().await;
        results
            .latest_all()
            .into_iter()
            .map(|s| (s.collection, s.is_proxy))
            .collect()
    };

    // Open index_* events, bucketed by collection (Scope::Index target).
    let open_by_coll: std::collections::HashMap<String, Vec<OpsEvent>> = {
        let events = advisor.events.lock().await;
        let mut by: std::collections::HashMap<String, Vec<OpsEvent>> =
            std::collections::HashMap::new();
        for ev in events.open_events() {
            if let crate::advisor::event::Scope::Index(c) = &ev.scope {
                by.entry(c.clone()).or_default().push(ev);
            }
        }
        by
    };

    let mut rows: Vec<IndexAdvisorRow> = index_health
        .into_iter()
        .map(|ih| {
            let recommendations: Vec<Recommendation> = hits_by_coll
                .get(&ih.collection)
                .map(|hits| hits.iter().map(Recommendation::from).collect())
                .unwrap_or_default();
            let open_events = open_by_coll
                .get(&ih.collection)
                .cloned()
                .unwrap_or_default();
            // Suggest a concrete reindex body only when there's a
            // recommendation to act on. Directional: re-embed straight
            // reindex (index params aren't schema-surfaced yet).
            let suggested_reindex = if recommendations.is_empty() {
                None
            } else {
                Some(serde_json::json!({ "re_embed": true }))
            };
            IndexAdvisorRow {
                collection: ih.collection.clone(),
                index_type: ih.index_type,
                params: ih.params,
                p95_ms: ih.p95_ms,
                recall: ih.recall,
                is_proxy: proxy_by_coll.get(&ih.collection).copied(),
                suggested_reindex,
                recommendations,
                open_events,
            }
        })
        .collect();
    rows.sort_by(|a, b| a.collection.cmp(&b.collection));

    Json(IndexAdvisorResponse { rows })
}
