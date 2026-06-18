//! Semantic incident-correlation admin API (`GET /v1/admin/incidents`).
//!
//! Admin-only — registered under the protected router with an extra
//! `rbac::require_admin` route layer, mirroring
//! [`crate::rest::index_advisor`] and [`crate::rest::correlations`].
//!
//! Returns the latest [`Incident`](crate::advisor::incidents::Incident)
//! clusters the off-tick incident correlator produced: recurring /
//! duplicate ops events collapsed into one row per distinct problem
//! (representative title, member event ids, count, severity, first/last
//! seen). This is the data the Niriksha "Incidents" panel reads. Empty
//! until the correlator has run at least once.

use axum::extract::State;
use axum::Json;
use serde::Serialize;

use crate::advisor::incidents::Incident;
use crate::rest::state::AppState;

/// Alert-storm noise-reduction summary: how many raw events collapsed into
/// how many distinct incidents on the latest correlation pass.
#[derive(Debug, Serialize)]
pub struct NoiseReduction {
    /// Total member events across all incidents (raw alert count).
    pub event_count: u32,
    /// Number of distinct incidents.
    pub incident_count: u32,
    /// Alerts suppressed by correlation (`event_count − incident_count`).
    pub suppressed: u32,
}

#[derive(Debug, Serialize)]
pub struct IncidentsResponse {
    pub incidents: Vec<Incident>,
    pub noise_reduction: NoiseReduction,
}

/// Compute the noise-reduction summary from a set of incidents. Pure.
fn noise_reduction(incidents: &[Incident]) -> NoiseReduction {
    let event_count: u32 = incidents.iter().map(|i| i.count).sum();
    let incident_count = incidents.len() as u32;
    NoiseReduction {
        event_count,
        incident_count,
        suppressed: event_count.saturating_sub(incident_count),
    }
}

/// `GET /v1/admin/incidents` — the latest incident clusters + the
/// alert-storm noise-reduction summary. Reads the stored result of the
/// off-tick correlator under a brief lock; never embeds or clusters on the
/// request path.
pub async fn list_incidents(State(state): State<AppState>) -> Json<IncidentsResponse> {
    let advisor = state.advisor();
    let incidents = {
        let store = advisor.incidents.lock().await;
        store.latest().to_vec()
    };
    let noise_reduction = noise_reduction(&incidents);
    Json(IncidentsResponse {
        incidents,
        noise_reduction,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::advisor::event::Severity;

    fn inc(id: &str, count: u32) -> Incident {
        Incident {
            id: id.into(),
            representative: format!("rep {id}"),
            member_event_ids: (0..count).map(|i| format!("{id}-{i}")).collect(),
            rule_ids: vec!["r".into()],
            count,
            max_severity: Severity::High,
            first_seen_ms: 1,
            last_seen_ms: 2,
            draft: None,
            is_novel: false,
            novelty_score: None,
        }
    }

    #[test]
    fn noise_reduction_is_events_minus_incidents() {
        // 5 + 3 = 8 events collapsed into 2 incidents → 6 suppressed.
        let nr = noise_reduction(&[inc("a", 5), inc("b", 3)]);
        assert_eq!(nr.event_count, 8);
        assert_eq!(nr.incident_count, 2);
        assert_eq!(nr.suppressed, 6);
    }

    #[test]
    fn noise_reduction_empty_is_zero() {
        let nr = noise_reduction(&[]);
        assert_eq!((nr.event_count, nr.incident_count, nr.suppressed), (0, 0, 0));
    }
}
