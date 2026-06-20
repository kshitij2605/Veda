//! Admin REST API for the cron-style scheduled-job scheduler.
//!
//! Endpoints (all admin-only via `rbac::require_admin` route layer):
//!   POST   /v1/admin/jobs         — register a new job (or overwrite by name)
//!   GET    /v1/admin/jobs         — list all registered jobs
//!   DELETE /v1/admin/jobs/{name}  — remove a job by name

use axum::Json;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use serde::{Deserialize, Serialize};

use crate::rest::collections::ErrorResponse;
use crate::rest::state::AppState;
use crate::scheduler::{JobKind, ScheduledJob};

/// Request body for `POST /v1/admin/jobs`.
#[derive(Debug, Deserialize)]
pub struct CreateJobRequest {
    pub name: String,
    pub cron: String,
    pub kind: JobKind,
}

/// Response body for list and create.
#[derive(Debug, Serialize)]
pub struct JobResponse {
    pub name: String,
    pub cron: String,
    pub kind: JobKind,
    pub last_run: Option<i64>,
}

impl From<ScheduledJob> for JobResponse {
    fn from(j: ScheduledJob) -> Self {
        Self {
            name: j.name,
            cron: j.cron,
            kind: j.kind,
            last_run: j.last_run,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct ListJobsResponse {
    pub jobs: Vec<JobResponse>,
}

/// `POST /v1/admin/jobs` — register or overwrite a scheduled job.
///
/// Returns 400 if the cron expression is invalid.
pub async fn create_job(
    State(state): State<AppState>,
    Json(req): Json<CreateJobRequest>,
) -> Result<(StatusCode, Json<JobResponse>), (StatusCode, Json<ErrorResponse>)> {
    if req.name.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "job name must not be empty".into(),
            }),
        ));
    }
    let job = ScheduledJob {
        name: req.name,
        cron: req.cron,
        kind: req.kind,
        last_run: None,
    };
    state
        .inner
        .job_registry
        .register(job.clone())
        .map_err(|e| (StatusCode::BAD_REQUEST, Json(ErrorResponse { error: e })))?;
    Ok((StatusCode::CREATED, Json(JobResponse::from(job))))
}

/// `GET /v1/admin/jobs` — list all registered jobs.
pub async fn list_jobs(State(state): State<AppState>) -> Json<ListJobsResponse> {
    let mut jobs: Vec<JobResponse> = state
        .inner
        .job_registry
        .list()
        .into_iter()
        .map(JobResponse::from)
        .collect();
    jobs.sort_by(|a, b| a.name.cmp(&b.name));
    Json(ListJobsResponse { jobs })
}

/// `DELETE /v1/admin/jobs/{name}` — remove a job by name (idempotent).
pub async fn delete_job(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> Result<StatusCode, (StatusCode, Json<ErrorResponse>)> {
    state.inner.job_registry.remove(&name).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse { error: e }),
        )
    })?;
    Ok(StatusCode::NO_CONTENT)
}
