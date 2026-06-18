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

#[derive(Debug, Serialize)]
pub struct IncidentsResponse {
    pub incidents: Vec<Incident>,
}

/// `GET /v1/admin/incidents` — the latest incident clusters. Reads the
/// stored result of the off-tick correlator under a brief lock; never
/// embeds or clusters on the request path.
pub async fn list_incidents(State(state): State<AppState>) -> Json<IncidentsResponse> {
    let advisor = state.advisor();
    let incidents = {
        let store = advisor.incidents.lock().await;
        store.latest().to_vec()
    };
    Json(IncidentsResponse { incidents })
}
